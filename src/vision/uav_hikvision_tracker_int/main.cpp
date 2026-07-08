#include <iostream>
#include <thread>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cstdint>
#include <cctype>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "yolo_decoder.h"
#include "bytetrack_lite.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <poll.h>
#include <unistd.h>

using namespace std;
using namespace cv;

// ============================================================
// 1. 基础配置：海康 RTSP + RKNN YOLO
// ============================================================

static const string DEFAULT_MODEL_PATH =
    "models/best_uav_headless_i8.rknn";
static string g_model_path = DEFAULT_MODEL_PATH;
static int g_rknn_output_count = 1;

static bool load_model_bytes(const string& path, vector<unsigned char>& data) {
    ifstream file(path, ios::binary | ios::ate);
    if (!file) {
        cerr << "[RKNN] failed to open model: " << path << endl;
        return false;
    }

    streamsize size = file.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > UINT32_MAX) {
        cerr << "[RKNN] invalid model size: " << size << " path=" << path << endl;
        return false;
    }

    data.resize(static_cast<size_t>(size));
    file.seekg(0, ios::beg);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        cerr << "[RKNN] failed to read model: " << path << endl;
        data.clear();
        return false;
    }

    return true;
}

static string shell_quote_path(const string& path) {
    string out = "'";
    for (char ch : path) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
    out += "'";
    return out;
}

static string trim_command_output_copy(const string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static string sha256sum_file(const string& path) {
    const string command = "sha256sum -- " + shell_quote_path(path) + " 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "unavailable";
    char buffer[256] = {0};
    string output;
    if (fgets(buffer, sizeof(buffer), pipe)) output = buffer;
    pclose(pipe);
    output = trim_command_output_copy(output);
    const size_t space = output.find_first_of(" \t");
    if (space != string::npos) output = output.substr(0, space);
    return output.empty() ? "unavailable" : output;
}

static string format_time_iso(time_t value) {
    char buf[64] = {0};
    struct tm tm_value;
    if (!localtime_r(&value, &tm_value)) return "unknown";
    if (strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm_value) == 0) {
        return "unknown";
    }
    return string(buf);
}

static void log_model_file_metadata(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        cerr << "[RKNN_MODEL] stat failed path=" << path
             << " errno=" << errno << endl;
        return;
    }
    cout << "[RKNN_MODEL] path=" << path << endl;
    cout << "[RKNN_MODEL] size_bytes=" << static_cast<long long>(st.st_size)
         << " mtime=" << format_time_iso(st.st_mtime)
         << " sha256=" << sha256sum_file(path) << endl;
}

// 默认不内置摄像头凭据；通过 ANTI_UAV_RTSP_URL 或 --rtsp 传入完整 RTSP。
const string DEFAULT_RTSP_URL = "";

const int IMG_SIZE = 640;
const float IOU_THRESH = 0.40f;

// 摄像头倒装时保持 true；如果后面物理正装，改成 false。
const bool ROTATE_IMAGE_180 = false;

// 是否写入共享内存，兼容你之前的 Qt / 上位机显示逻辑。
const bool ENABLE_SHM = true;
const char* SHM_NAME = "/rk3588_vision";

// Qt 是 ELF2 的主界面，默认不额外创建 OpenCV 窗口；需要调试时可显式开启。
const bool ENABLE_IMSHOW = []() {
    const char* value = getenv("ANTI_UAV_YOLO_IMSHOW");
    if (!value) return false;
    string normalized(value);
    transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    return normalized == "1" || normalized == "true" ||
           normalized == "yes" || normalized == "on";
}();

static string redact_url_credentials(const string& url) {
    const size_t scheme = url.find("://");
    if (scheme == string::npos) return url;
    const size_t authority_start = scheme + 3;
    const size_t at = url.find('@', authority_start);
    if (at == string::npos) return url;
    return url.substr(0, authority_start) + "<credentials>@" + url.substr(at + 1);
}

// UDP 输出给本机 Qt 程序；支持启动参数和环境变量覆盖，避免本机 IP 变化时改源码。
static const string DEFAULT_UDP_IP = "127.0.0.1";
static const int DEFAULT_UDP_PORT = 5005;
static string g_udp_ip = DEFAULT_UDP_IP;
static int g_udp_port = DEFAULT_UDP_PORT;
static const int DEFAULT_FRAME_STREAM_PORT = 5010;
static int g_frame_stream_port = DEFAULT_FRAME_STREAM_PORT;
static int g_frame_stream_max_width = 1024;
static int g_frame_stream_jpeg_quality = 62;
static int g_frame_stream_send_buffer = 262144;
static int g_frame_stream_send_timeout_ms = 180;
static bool g_frame_stream_letterbox640 = true;
static const int DEFAULT_AUDIO_FUSION_PORT = 5007;
static int g_audio_fusion_port = DEFAULT_AUDIO_FUSION_PORT;
static const int DEFAULT_QT_CONTROL_PORT = 5011;
static int g_qt_control_port = DEFAULT_QT_CONTROL_PORT;
static std::atomic<long long> g_local_ui_lease_until_ms{0};
static std::atomic<bool> g_audio_guidance_enabled{true};
static std::atomic<bool> g_audio_calibration_reset_requested{false};
static string g_fusion_calibration_path =
    "models/best_uav_headless_i8.rknn";
static string g_learning_profile_path =
    "models/best_uav_headless_i8.rknn";
static bool g_runtime_learning_enabled = true;
static bool g_load_audio_offset_prior = false;
static string g_tracker_config_path =
    "models/best_uav_headless_i8.rknn";
static string g_control_params_path =
    "models/best_uav_headless_i8.rknn";
static string g_control_params_session_snapshot_path;
// METRICS INT8 PATCH
static std::atomic<double> g_latest_yolo_infer_ms{0.0};
static std::atomic<double> g_latest_vision_fps{0.0};

static long long metric_now_ms_cpp() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
}

static std::string metric_model_mode_cpp() {
    std::string p = g_model_path;
    std::string lower = p;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("int8") != std::string::npos || lower.find("_i8") != std::string::npos || lower.find("i8.") != std::string::npos) {
        return "INT8";
    }
    return "FP";
}

// VISION METRICS PATCH



atomic<bool> is_running(true);
atomic<bool> g_detection_enabled(true);

static bool env_flag_or_default_early(const char* name, bool default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    string normalized(value);
    transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    if (normalized == "1" || normalized == "true" ||
        normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" ||
        normalized == "no" || normalized == "off") {
        return false;
    }
    return default_value;
}

// ============================================================
// 1.1 FP 低频推理 + 海康 PTZ 粗闭环控制配置
// ============================================================

// INT8 raw-head 模型实时性优先：默认每帧推理。负载过高时可用 --infer-stride 临时降频。
static int g_infer_frame_stride = 1;

// 按 C 开/关云台自动跟踪。实测任务默认进入自动闭环，可用环境变量临时关闭。
atomic<bool> ptz_control_enabled(
    env_flag_or_default_early("ANTI_UAV_AUTO_PTZ", true)
);

// 按 B 开/关自动变焦。自动变焦只根据目标框大小粗略调整，不追求精准。
// + / - 保留手动连续变焦，手动变焦优先级高于自动变焦。
atomic<bool> auto_zoom_enabled(
    env_flag_or_default_early("ANTI_UAV_AUTO_ZOOM", true)
);
atomic<int> manual_zoom_user(0);
atomic<long long> manual_zoom_until_ms(0);

// WSAD 手动云台控制：优先级高于自动跟踪，但不关闭自动模式。
// 松开/停止按键后一小段时间自动恢复自动跟踪。
atomic<int> manual_pan_user(0);
atomic<int> manual_tilt_user(0);
atomic<long long> manual_ptz_until_ms(0);
// 手动 WSAD/+/- 按键后，立即唤醒 PTZ 线程发送一次命令，避免等自动控制周期造成卡顿。
atomic<bool> manual_force_send(false);
atomic<long long> manual_auto_resume_block_until_ms(0);
atomic<int> manual_owner_code(0); // 0 none, 1 windows_qt, 2 board_qt, 3 local_keyboard, 4 other
atomic<long long> g_qt_windows_last_seq(0);
atomic<long long> g_qt_board_last_seq(0);
atomic<long long> g_qt_other_last_seq(0);
atomic<long long> g_stale_command_dropped_count(0);
atomic<long long> g_last_stale_command_drop_ms(0);

// 海康球机控制参数
static string env_string_or_default_early(const char* name, const string& default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    return string(value);
}

string HIK_IP = env_string_or_default_early("ANTI_UAV_HIK_IP", "CAMERA_IP");
string HIK_USER = env_string_or_default_early("ANTI_UAV_CAMERA_USER", "YOUR_USERNAME");
string HIK_PASS = env_string_or_default_early("ANTI_UAV_CAMERA_PASSWORD", "");
const int HIK_CHANNEL = 1;

string PTZ_STATUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/status";
string PTZ_ABSOLUTE_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/absolute";
string PTZ_CONTINUOUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/continuous";

static void refresh_hik_urls() {
    PTZ_STATUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" + to_string(HIK_CHANNEL) + "/status";
    PTZ_ABSOLUTE_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" + to_string(HIK_CHANNEL) + "/absolute";
    PTZ_CONTINUOUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" + to_string(HIK_CHANNEL) + "/continuous";
}

// 海康特殊预置点用于日夜模式切换。阴天测试时默认强制白天彩色，
// 避免球机自动进入红外/黑白导致 RGB YOLO 模型漏检。
const bool FORCE_DAY_MODE_ON_START =
    env_flag_or_default_early("ANTI_UAV_FORCE_CAMERA_PROFILE_ON_START", false);
const bool DISABLE_AUTO_IR_FOR_TEST =
    env_flag_or_default_early("ANTI_UAV_DISABLE_AUTO_IR_FOR_TEST", false);
const int PRESET_DAY_MODE = 39;
const int PRESET_NIGHT_MODE = 40;
const int PRESET_AUTO_DAYNIGHT = 46;
atomic<int> g_camera_mode_preset_cmd(0);
atomic<bool> g_camera_is_grayscale(false);
atomic<double> g_camera_brightness_mean(-1.0);
atomic<double> g_camera_contrast_std(-1.0);
atomic<double> g_camera_saturation_mean(-1.0);
atomic<double> g_camera_blur_laplacian_var(-1.0);
atomic<int> g_camera_grayscale_frames(0);
atomic<long long> g_last_force_day_retry_ms(0);
atomic<long long> g_day_retry_holdoff_until_ms(0);
atomic<bool> g_camera_safe_shutdown_done(false);

enum class CameraProfile {
    DAY_NORMAL = 0,
    DAY_CLOUDY_UAV = 1,
    FAST_TRACKING = 2,
    LOW_LIGHT_KEEP_RGB = 3,
    NIGHT_IR_TEST = 4
};

const CameraProfile DEFAULT_CAMERA_PROFILE = CameraProfile::DAY_CLOUDY_UAV;
const bool AUTO_PROFILE_ENABLED_DEFAULT = false;
const bool ALLOW_AUTO_NIGHT_IR = false;
const long long CAMERA_PROFILE_MIN_EVAL_MS = 2000;
const long long CAMERA_PROFILE_MIN_HOLD_MS = 15000;
const long long CAMERA_PROFILE_MANUAL_HOLD_MS = 60000;
const int CAMERA_PROFILE_CONFIRM_STATS = 5;

atomic<int> g_camera_profile((int)CameraProfile::DAY_CLOUDY_UAV);
atomic<bool> g_auto_camera_profile_enabled(AUTO_PROFILE_ENABLED_DEFAULT);
atomic<long long> g_manual_profile_override_until_ms(0);
atomic<long long> g_last_profile_eval_ms(0);
atomic<long long> g_last_profile_switch_ms(0);
atomic<int> g_profile_candidate((int)CameraProfile::DAY_CLOUDY_UAV);
atomic<int> g_profile_candidate_count(0);
atomic<int> g_camera_detected(0);
atomic<long long> g_camera_lost_since_ms(0);
atomic<double> g_camera_zoom_ratio(1.0);
atomic<bool> g_camera_ptz_moving(false);
static mutex g_camera_quality_mtx;
static string g_camera_track_state = "SEARCH";
static string g_image_quality_warning = "OK";
static string g_last_profile_switch_reason = "startup_default";

// 你前面实测的 absolute 方向
const int ABS_PAN_DIR = -1;
const int ABS_TILT_DIR =1;

// 粗闭环参数：按 C 后使用 continuous 小速度脉冲控制，反应比 absolute + 查询状态快得多。
// 目标：只要把无人机拉回中心小框附近，不做像素级精准对中。
const float PTZ_CONTROL_PERIOD_SEC = 0.06f;     // 控制周期，越小反应越快；声学引导阶段优先降低停顿
const float PTZ_TARGET_MAX_AGE_SEC = 0.75f;     // FP 识别较慢，允许稍旧一点的目标，防止一顿一停
const float AUDIO_CUE_MAX_AGE_SEC = 1.80f;      // 音频 cue 采用短时保持，避免 0.5s 音频窗抖动导致停-走-停
const float AUDIO_VISION_SYNC_SEC = 0.50f;      // 音视频共同样本用于在线校准的最大时间差
const float AUDIO_GUIDE_DEAD_ZONE_DEG = 10.0f;  // ReSpeaker4 硬件 DOA 直通后缩小提前停止区，提高粗定位贴近度
const float AUDIO_GUIDE_MIN_STABILITY = 0.78f;  // 低于该稳定度的 DOA 不用于强引导和校准
const float AUDIO_GUIDE_MIN_SCORE_PERCENT = 20.0f; // 声学引导阈值低于报警阈值，远距离弱候选先引导搜索
const float AUDIO_GUIDE_MIN_SNR_DB = 1.20f;      // 声学引导最低信噪比，按 2026-06-15 实飞日志下调以缩短确认长尾
const float AUDIO_OFFSET_HIGH_CONF = 0.75f;     // offset 达到该置信度后允许强音频引导
const float AUDIO_OFFSET_BOOTSTRAP_CONF = 0.55f; // 视觉锁定且声学稳定时，先用短时主簇建立现场 offset
const float AUDIO_OFFSET_GUIDE_CONF = 0.65f;     // offset 退到自举下限时只保留显示，不再拉动云台
const int AUDIO_OFFSET_BOOTSTRAP_MIN_SAMPLES = 12; // 约 0.5s 以上稳定同步样本后允许自举
const int AUDIO_OFFSET_BOOTSTRAP_WINDOW = 28;       // 自举只看最近短窗口，避免旧方向污染新方向
const float AUDIO_OFFSET_BOOTSTRAP_MAX_STD_DEG = 8.0f; // offset 观测短时离散度小于该值才自举
const int AUDIO_OFFSET_OUTLIER_GRACE_SAMPLES = 18;  // 已有 offset 后，连续多次坏样本才缓慢降置信度
const float AUDIO_HIGH_CONF_MIN_STABILITY = 0.85f; // 高置信声学转向必须同时满足较高 DOA 稳定度
const int AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES = 24; // 防止少量错误视觉-声学样本把 offset 学成高置信
const float AUDIO_GUIDE_SWEEP_ZONE_DEG = 25.0f; // 接近音频角后叠加局部上下扫描，避免只盯一个点
const float AUDIO_LOW_CONF_DEAD_ZONE_DEG = 28.0f; // 未完成校准时只做粗扇区引导，避免错角频繁拉云台
const float AUDIO_BOOTSTRAP_DEAD_ZONE_DEG = 14.0f; // 自举成功但未高置信时，允许大角度但提前减速停靠
const float AUDIO_LOW_CONF_MAX_TURN_ERROR_DEG = 42.0f; // 未完成现场校准前不允许跨大角度追声，避免旧 offset 或回声带偏
const float VISION_AUDIO_SUPPRESS_AFTER_TRACK_SEC = 1.20f; // 视觉刚锁定或短暂丢帧时不让音频抢控制权
const float AUDIO_GUIDE_CONFIRM_SEC = 0.35f;    // 硬件 DOA 稳定，缩短确认等待以提高声学接管响应
const float AUDIO_GUIDE_CONFIRM_DRIFT_DEG = 18.0f; // 候选方向允许的短时漂移
const float AUDIO_GUIDE_REJECT_JUMP_DEG = 180.0f; // 允许真实大角度换向，不再粘住上一声源方向
const float AUDIO_GUIDE_ACCEPT_HOLD_SEC = 1.10f; // 稳定 cue 接受后的保持时间
const float AUDIO_GUIDE_SETTLE_SEC = 0.20f;     // 到达音频角后短暂停稳，再开始局部搜索
const float AUDIO_LOCAL_SWEEP_FLIP_SEC = 1.20f; // 到位后的上下搜索翻转周期，降低抽搐感
const float AUDIO_LOCAL_PAN_DITHER_FLIP_SEC = 2.80f; // 左右小幅补扫明显慢于上下搜索
const float AUDIO_RECOVERY_SWEEP_SEC = 0.60f;   // 音频刚丢失后只做短时局部重捕，避免无目标时长时间乱扫
const float AUDIO_LOCAL_SEARCH_MAX_SEC = 0.95f; // 到达声源方向后最多局部搜索时长
const float AUDIO_LOW_CONF_LOCAL_SEARCH_MAX_SEC = 0.75f; // 未标定时只做短促确认，不持续扫
const int AUDIO_APPROACH_TILT_SWEEP_SPEED = 24; // 预留：低速接近扫描，当前转向阶段不启用上下扫
const int AUDIO_SEARCH_TILT_SWEEP_SPEED = 0;    // 转向阶段禁止上下扫，避免方向没到位就乱扫
const int AUDIO_LOCAL_SWEEP_TILT_SPEED = 12;    // 到达音频角后的上下搜索速度
const int AUDIO_LOW_CONF_LOCAL_SWEEP_TILT_SPEED = 6; // 未标定时降低俯仰扫速，减少手动接管需求
const int AUDIO_LOCAL_SWEEP_PAN_SPEED = 4;      // 到达音频角后的左右小幅补扫速度
const int AUDIO_HIGH_CONF_MIN_PAN_SPEED = 22;    // 已完成校准时的最小声源转向速度
const int AUDIO_HIGH_CONF_MAX_PAN_SPEED = 56;    // 已完成校准时的最大声源转向速度
const int AUDIO_BOOTSTRAP_MIN_PAN_SPEED = 16;     // 自举 offset 后的大角度重捕速度
const int AUDIO_BOOTSTRAP_MAX_PAN_SPEED = 42;     // 自举 offset 未高置信前仍限制最高速度
const int AUDIO_LOW_CONF_MIN_PAN_SPEED = 8;      // 未完成校准时的最小声源转向速度
const int AUDIO_LOW_CONF_MAX_PAN_SPEED = 24;     // 未完成校准时的最大声源转向速度
const int AUTO_PTZ_MAX_SPEED_STEP = 36;          // 自动连续控制每周期最大速度变化，提高远离中心时的追踪响应
const int AUDIO_PTZ_MAX_SPEED_STEP = 16;         // 硬件 DOA 稳定时允许更快提速，提高声学引导响应
const float AUDIO_VERTICAL_SWEEP_MIN_DELTA_DEG = 0.8f; // 高倍率时的最小垂直扇扫覆盖
const float AUDIO_VERTICAL_SWEEP_MAX_DELTA_DEG = 3.5f; // 低倍率时的最大垂直扇扫覆盖
const float AUDIO_VERTICAL_SWEEP_VFOV_FACTOR = 0.35f;  // 根据当前 VFOV 自动估计上下搜索范围

// 中心允许范围：你说框太大，这里已经缩小。
// 默认死区偏小，优先保证目标偏离中心后能快速起动；高倍率限速另行抑制过冲。
const float CENTER_DEAD_ZONE_X_RATIO = 0.040f;
const float CENTER_DEAD_ZONE_Y_RATIO = 0.040f;
const float CENTER_DEAD_ZONE_X_MIN_PX = 28.0f;
const float CENTER_DEAD_ZONE_Y_MIN_PX = 24.0f;

// continuous 控制方向：沿用你之前 Python 控制脚本里的方向。
const int CONT_PAN_DIR = 1;
const int CONT_TILT_DIR = 1;
const int CONT_ZOOM_DIR = 1;

// continuous 速度。越大反应越快，也越容易过冲。
const int PTZ_MIN_SPEED = 16;
const int PTZ_MAX_SPEED = 90;
const int PTZ_MAX_SPEED_HIGH_ZOOM = 62;
// 速度曲线：>1 表示靠近中心时更温柔，远离中心时仍然足够快，减少过冲。
const float PTZ_SPEED_CURVE = 1.12f;


// 自动变焦参数：用检测框最大边占画面最大边的比例控制。
// 做成“区间保持 + 冷却 + 回差”，减少自动变焦和自动对焦互相拉扯。
// 不再追固定 0.16；目标最大边稳定在 10%~25%，12%~22% 内完全保持。
const float ZOOM_IN_NEAR_EDGE_BLOCK_RATIO = 0.25f; // 目标离画面中心太远时禁止继续放大，防止放大后丢失
const float ZOOM_RANGE_MIN_RATIO = 0.10f;       // 小于 10% 才允许分步放大
const float ZOOM_HOLD_LOW_RATIO = 0.12f;        // 12%~22% 是稳定保持区
const float ZOOM_HOLD_HIGH_RATIO = 0.22f;
const float ZOOM_RANGE_MAX_RATIO = 0.25f;       // 大于 25% 才允许分步缩小
const int ZOOM_MIN_SPEED = 15;
const int ZOOM_MAX_SPEED = 60;
// 缩小时通常是在目标过大或丢失后扩视角，允许比放大更快。
const int ZOOM_OUT_MIN_SPEED = 25;
const int ZOOM_OUT_MAX_SPEED = 125;

const int ZOOM_SEARCH_OUT_SPEED = 160;           // 丢目标或声学搜索时快速缩小扩大视角
const float ZOOM_TARGET_MAX_AGE_SEC = 1.20f;
const float LOST_ZOOM_OUT_AFTER_SEC = 0.35f;
const float FAST_ZOOM_OUT_MAX_DURATION_SEC = 4.0f;
const float ZOOM_CMD_PERIOD_SEC = 0.12f;          // 自动变焦最小命令间隔，越大越稳
const float AUTO_ZOOM_HOLD_SEC = 0.14f;           // 自动变焦短保持，避免 zoom-out 断续太明显
const float ZOOM_FLIP_COOLDOWN_SEC = 1.80f;       // 放大/缩小方向切换冷却，减少来回拉扯
const float ZOOM_EMA_ALPHA = 0.12f;               // 框大小平滑系数，减小检测框抖动导致的误放大
const float AUTO_FOCUS_AFTER_ZOOM_SETTLE_SEC = 0.90f;
const float AUTO_FOCUS_AFTER_ZOOM_MIN_INTERVAL_SEC = 2.50f;
const int AUTO_FOCUS_BLUR_STABLE_FRAMES = 15;
const int AUTO_FOCUS_CONF_DROP_STABLE_FRAMES = 10;

// 手动 + / - 变焦每次按键维持的时间，避免必须一直长按。
const int MANUAL_ZOOM_SPEED = 120;
const int MANUAL_ZOOM_HOLD_MS = 220;   // 手动 +/- 死人开关保持时间：松手后很快停

// WSAD 手动云台速度与按键保持时间。
const int MANUAL_PTZ_SPEED = 52;
const int MANUAL_PTZ_HOLD_MS = 190;    // WSAD 死人开关保持时间：兼顾连续按住的丝滑和松手快停

// 手动模式持续补发周期：比自动 PTZ 周期更快，WSAD 手感会接近“持续按住”。
const float MANUAL_PTZ_SEND_PERIOD_SEC = 0.065f;  // 手动云台补发更快，按住更丝滑；松手靠 deadman stop 兜底
const float MANUAL_ZOOM_SEND_PERIOD_SEC = 0.060f; // 手动变焦补发周期

// 手动脉冲结束后必须强制多次 STOP。海康 continuous 如果没收到 0/0/0，
// 会继续执行上一条非零速度指令，这是“按一下停不住”的根因。
const int MANUAL_STOP_RETRY_COUNT = 3;
const int MANUAL_STOP_RETRY_INTERVAL_MS = 35;
const float MANUAL_AUTO_RESUME_DELAY_SEC = 0.35f;  // 短 hold：先让 STOP 落到球机，再快速恢复自动控制
const int QT_MANUAL_PTZ_SPEED = 55;
// Qt 端只发 zoom 方向，倍率速度在这里统一转换成海康 continuous 的有效速度。
// 旧值 20/35 对当前球机几乎不可见；本地键盘 +/- 一直使用 120，因此 Qt 路径保持同一速度单位。
const int QT_MANUAL_ZOOM_SPEED = 120;
const int QT_MANUAL_ZOOM_SPEED_MAX = 180;
const int QT_MANUAL_TIMEOUT_MS = 300;


// ============================================================
// 1.2 在线调参参数
// ============================================================
// 飞行时可以直接按键调整下面这些参数，不需要重新编译。
// 这比“自动自学习”更安全：你可以一边看无人机跟踪，一边逐步把速度调到刚好够快且不过冲。

atomic<int> g_ptz_min_speed(PTZ_MIN_SPEED);
atomic<int> g_ptz_max_speed(PTZ_MAX_SPEED);
atomic<int> g_ptz_max_speed_high_zoom(PTZ_MAX_SPEED_HIGH_ZOOM);
atomic<float> g_ptz_speed_curve(PTZ_SPEED_CURVE);

atomic<float> g_center_dead_zone_x_ratio(CENTER_DEAD_ZONE_X_RATIO);
atomic<float> g_center_dead_zone_y_ratio(CENTER_DEAD_ZONE_Y_RATIO);
atomic<float> g_ptz_control_period_sec(PTZ_CONTROL_PERIOD_SEC);

atomic<int> g_zoom_min_speed(ZOOM_MIN_SPEED);
atomic<int> g_zoom_max_speed(ZOOM_MAX_SPEED);
atomic<int> g_zoom_out_min_speed(ZOOM_OUT_MIN_SPEED);
atomic<int> g_zoom_out_max_speed(ZOOM_OUT_MAX_SPEED);
atomic<int> g_zoom_search_out_speed(ZOOM_SEARCH_OUT_SPEED);
atomic<float> g_zoom_cmd_period_sec(ZOOM_CMD_PERIOD_SEC);
atomic<float> g_zoom_flip_cooldown_sec(ZOOM_FLIP_COOLDOWN_SEC);
atomic<float> g_zoom_target_min_ratio(0.10f);
atomic<float> g_zoom_target_ideal_low_ratio(0.12f);
atomic<float> g_zoom_target_ideal_high_ratio(0.22f);
atomic<float> g_zoom_target_max_ratio(0.25f);
atomic<float> g_zoom_min_ratio(1.0f);
atomic<float> g_zoom_max_ratio(25.0f);
atomic<float> g_lost_search_zoom_ratio(1.2f);
atomic<float> g_search_min_box_ratio(0.04f);
atomic<float> g_track_min_box_ratio(0.06f);
atomic<float> g_alarm_min_box_ratio(0.08f);
atomic<float> g_search_conf_thresh(0.35f);
atomic<float> g_track_conf_thresh(0.45f);
atomic<float> g_alarm_conf_thresh(0.45f);
atomic<int> g_search_confirm_frames(3);
atomic<int> g_track_confirm_frames(3);
atomic<int> g_alarm_confirm_frames(2);
atomic<int> g_detection_policy_mode(1);
atomic<long long> g_control_params_version{0};
atomic<float> g_tracking_pan_gain(1.25f);
atomic<float> g_tracking_tilt_gain(1.15f);
atomic<float> g_tracking_feedforward_gain(0.0f);
atomic<int> g_tracking_mode(1);
atomic<int> g_zoom_step_in_speed(35);
atomic<int> g_zoom_step_out_speed(80);
atomic<float> g_autofocus_cooldown_sec(5.0f);
atomic<bool> g_autofocus_enabled{true};
atomic<long long> g_last_autofocus_request_ms{0};
atomic<long long> g_autofocus_cooldown_until_ms{0};
atomic<int> g_last_autofocus_reason{0};

atomic<int> g_manual_ptz_speed(MANUAL_PTZ_SPEED);
atomic<int> g_manual_zoom_speed(MANUAL_ZOOM_SPEED);

// 你说打开了镜像，所以显示画面的上下控制需要反一下。
// 如果后面发现上下又反了，把 true 改成 false。
const bool INVERT_DISPLAY_Y_FOR_CONTROL = true;

// 保留 absolute 相关参数，后续如果要改回 absolute 闭环还能用。
const float PTZ_PAN_GAIN = 0.80f;
const float PTZ_TILT_GAIN = 0.76f;
const float MAX_PAN_STEP_DEG_LOW_ZOOM = 3.5f;
const float MAX_TILT_STEP_DEG_LOW_ZOOM = 2.2f;
const float MAX_PAN_STEP_DEG_HIGH_ZOOM = 0.90f;
const float MAX_TILT_STEP_DEG_HIGH_ZOOM = 0.70f;

struct FovPoint {
    float zoom;
    float hfov;
    float vfov;
};

const vector<FovPoint> FOV_TABLE = {
    {1.0f, 60.000f, 36.000f},
    {1.2f, 52.740f, 32.327f},
    {2.4f, 28.300f, 17.000f},
    {5.0f, 14.642f, 8.582f},
    {10.0f, 7.564f, 4.419f},
    {13.1f, 5.852f, 3.445f},
    {20.0f, 4.057f, 2.419f},
    {37.0f, 2.478f, 1.261f},
};


// ============================================================
// 2. 跟踪参数：取消环境学习，只保留 YOLO + Kalman
// ============================================================

// ============================================================
// 2. 跟踪参数：取消环境学习，只保留严格 YOLO + 状态机 + Kalman
// ============================================================

// 首次捕获阈值：室内杂乱环境误检多，建议 0.50~0.65。
// 只有 conf >= ACQUIRE_CONF 的检测，才允许进入候选态。
const float ACQUIRE_CONF = 0.55f;

// 已确认目标后的维持阈值：可以比首次捕获略低，但不能低到 0.01。
const float TRACK_CONF = 0.30f;

// 连续多少帧同一位置附近检测到，才确认目标。
const int CONFIRM_FRAMES = 3;

// 候选阶段允许短暂漏检多少帧。
const int CANDIDATE_MISS_MAX = 2;

// 已确认后连续丢失多少帧，才退出跟踪。
const int MAX_LOST_FRAMES = 10;

// 搜索/确认阶段目标中心的最大跳变距离。
const float CANDIDATE_GATE_PX = 220.0f;
const float TRACK_GATE_PX = 280.0f;

// 检测框几何过滤。根据 704x576 子码流做保守约束。
const float MIN_BOX_AREA_RATIO = 0.000015f;
const float MAX_BOX_AREA_RATIO = 0.060f;
const float MIN_BOX_ASPECT = 0.20f;
const float MAX_BOX_ASPECT = 5.00f;
const float MAX_SIZE_JUMP_RATIO = 3.0f;

// ============================================================
// 3. 数据结构
// ============================================================

struct FrameContext {
    int frame_id = 0;
    long long camera_estimated_mono_ms = -1;
    long long rtsp_grab_start_mono_ms = -1;
    long long rtsp_grab_done_mono_ms = -1;
    long long decode_done_mono_ms = -1;
    long long infer_start_mono_ms = -1;
    long long infer_end_mono_ms = -1;
    long long postprocess_done_mono_ms = -1;

    // 原始海康画面尺寸，旋转之后的尺寸。
    int src_w = 0;
    int src_h = 0;

    // letterbox 参数：src -> 640x640
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;

    // 给 NPU 的 RGB 640x640 图。
    Mat infer_rgb;

    // 给显示 / 共享内存的 BGR 640x640 图。
    Mat ui_bgr;

    // 给 Qt 同帧流的原始比例画面，避免把视频界面压成 640x640 方形。
    Mat stream_bgr;

    vector<ObjectBox> decoded_boxes;
};

struct DetBox {
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float conf = 0.0f;

    // 640 letterbox 坐标，兼容旧 UI。
    float ui_cx = 0.0f;
    float ui_cy = 0.0f;
    float ui_w = 0.0f;
    float ui_h = 0.0f;
};

template <typename T>
class LatestQueue {
private:
    queue<T> q;
    mutex mtx;
    condition_variable cv;
    size_t max_size;

public:
    explicit LatestQueue(size_t size) : max_size(size) {}

    void push_latest(T item) {
        unique_lock<mutex> lock(mtx);

        // 低延迟原则：队列满了就丢旧帧，不阻塞采集线程。
        while (q.size() >= max_size) {
            q.pop();
        }

        q.push(std::move(item));
        cv.notify_one();
    }

    bool pop(T& item) {
        unique_lock<mutex> lock(mtx);
        cv.wait_for(lock, chrono::milliseconds(100), [this]() {
            return !q.empty() || !is_running.load();
        });

        if (!is_running.load() || q.empty()) return false;

        item = std::move(q.front());
        q.pop();
        return true;
    }
};

LatestQueue<FrameContext> input_queue(2);
LatestQueue<FrameContext> output_queue(4);

mutex latest_ui_mtx;
Mat latest_ui_frame;
string latest_status_text = "starting";

// RK 到 Qt 的同帧视频流：只保留最新后处理帧，避免编码或网络慢时拖住 YOLO。
mutex latest_stream_mtx;
condition_variable latest_stream_cv;
string latest_stream_meta;
vector<uchar> latest_stream_jpg;
uint64_t latest_stream_seq = 0;

// 显示线程用：采集线程每帧更新，不依赖 YOLO 推理频率，避免 FP 推理慢导致窗口卡成几 fps。
mutex latest_camera_mtx;
Mat latest_camera_frame;
int latest_camera_id = -1;

struct SharedDetection {
    bool valid = false;
    bool confirmed = false;
    int frame_id = -1;
    int fw = 0;
    int fh = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float conf = 0.0f;
    double ts = 0.0;
};

struct PTZState {
    bool ok = false;
    int azimuth = 0;
    int elevation = 0;
    int absolute_zoom = 10;
};

atomic<bool> g_latest_ptz_state_ok{false};
atomic<int> g_latest_ptz_azimuth_unit{0};
atomic<int> g_latest_ptz_elevation_unit{0};
atomic<int> g_latest_ptz_absolute_zoom{10};
mutex g_ptz_home_mtx;
PTZState g_session_initial_ptz;
PTZState g_saved_home_ptz;
bool g_session_initial_ptz_valid = false;
bool g_saved_home_ptz_valid = false;
bool g_config_home_valid = false;
PTZState g_config_home_ptz;
int g_config_home_preset = 0;

mutex latest_det_mtx;
SharedDetection latest_det;

struct SharedAudioCue {
    bool detected = false;
    double raw_doa_deg = -1.0;
    double smooth_doa_deg = -1.0;
    double score = 0.0;
    double score_ema = 0.0;
    double stability = 0.0;
    double rms_dbfs = -120.0;
    double snr_db = -99.0;
    double ts = 0.0;
    long long ts_mono_ms = 0;
    string state = "IDLE";
};

struct FusionCalibration {
    double offset_deg = 0.0;
    double confidence = 0.0;
    int total_samples = 0;
    int stable_samples = 0;
    double last_error_deg = 99999.0;
    double last_visual_azimuth_deg = 99999.0;
    double last_audio_world_deg = 99999.0;
};

struct FusionTelemetry {
    string mode = "IDLE";
    string fusion_state = "IDLE";
    string control_owner = "idle";
    string control_source = "none";
    string manual_owner = "none";
    string search_state = "none";
    string audio_guidance_state = "disabled";
    string audio_reject_reason = "none";
    string audio_calibration_state = "uncalibrated";
    bool audio_guided = false;
    double camera_azimuth_deg = 99999.0;
    double audio_pan_err_deg = 99999.0;
    double audio_visual_angle_error_deg = 99999.0;
    double visual_azimuth_deg = 99999.0;
    double audio_world_azimuth_deg = 99999.0;
    double mic_to_camera_offset_deg = 0.0;
    double offset_confidence = 0.0;
    string offset_confidence_label = "low";
    int calibration_samples = 0;
    double doa_stability = 0.0;
    int pan_speed = 0;
    int tilt_speed = 0;
    int zoom_speed = 0;
    bool auto_track_enabled = false;
    bool auto_zoom_enabled = false;
    bool auto_focus_enabled = false;
    bool manual_override_active = false;
    long long manual_hold_remaining_ms = 0;
    bool emergency_stop_active = false;
    long long target_age_ms = -1;
    string target_state = "NO_TARGET";
    string ptz_block_reason = "none";
    long long stale_command_dropped = 0;
    bool high_zoom_limited = false;
    int max_speed_cap = 0;
    int actual_pan_speed = 0;
    int actual_tilt_speed = 0;
    double zoom_ratio = -1.0;
    double target_max_side = 0.0;
    double target_max_side_ema = 0.0;
    string zoom_state = "IDLE";
    string zoom_action = "hold";
    string zoom_reason = "none";
    bool target_near_edge = false;
    bool target_stable = false;
    bool target_lost = false;
    double lost_duration_sec = 0.0;
    double search_zoom_level = -1.0;
    bool focus_triggered = false;
    string focus_reason = "none";
    string focus_state = "HOLD";
    long long last_focus_request_ms = 0;
    long long focus_cooldown_remaining_ms = 0;
    double vertical_sweep_delta_deg = 0.0;
};

struct RuntimeLearningProfile {
    bool loaded = false;
    double best_offset_deg = 0.0;
    double best_audio_error_ema = 99999.0;
    int audio_samples = 0;
    double best_center_error_ema = 99999.0;
    int track_samples = 0;
    int ptz_max_speed = PTZ_MAX_SPEED;
    int ptz_max_speed_high_zoom = PTZ_MAX_SPEED_HIGH_ZOOM;
    double center_dead_zone_x_ratio = CENTER_DEAD_ZONE_X_RATIO;
    double center_dead_zone_y_ratio = CENTER_DEAD_ZONE_Y_RATIO;
    long long updated_ts_ms = 0;
};

struct RuntimeLearningState {
    double audio_error_ema = 99999.0;
    int audio_samples = 0;
    int audio_bad_samples = 0;
    double center_error_ema = 99999.0;
    int track_samples = 0;
    int stable_track_samples = 0;
    int track_bad_samples = 0;
    int last_dx_sign = 0;
    int sign_flips = 0;
    double last_track_adjust_time = -10.0;
    string last_action = "init";
};

mutex latest_audio_mtx;
SharedAudioCue latest_audio;

mutex fusion_mtx;
FusionCalibration fusion_calib;
FusionTelemetry fusion_telemetry;

mutex runtime_learning_mtx;
RuntimeLearningProfile runtime_profile;
RuntimeLearningState runtime_learning;



// ============================================================
// 3.1 PTZ 控制工具函数：curl + Digest Auth，无需修改 CMake 链接库
// ============================================================

double now_sec() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::steady_clock::now().time_since_epoch()
    ).count() / 1000000.0;
}

long long now_ms() {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch()
    ).count();
}

struct SystemResourceSnapshot {
    double cpu_percent = -1.0;
    double mem_used_percent = -1.0;
    double mem_used_mb = -1.0;
    double mem_total_mb = -1.0;
    double npu_freq_mhz = -1.0;
    double npu_load_percent = -1.0;
    string npu_governor;
    string temp_summary;
    string cpu_freq_summary;
    double process_rss_mb = -1.0;
    int process_threads = -1;
};

static mutex g_system_resource_mtx;
static SystemResourceSnapshot g_system_resource_cache;
static long long g_system_resource_last_ms = -5000;
static long long g_prev_cpu_total = -1;
static long long g_prev_cpu_idle = -1;

static string read_text_file_trimmed(const string& path) {
    ifstream f(path);
    if (!f.good()) return "";
    stringstream ss;
    ss << f.rdbuf();
    string s = ss.str();
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    size_t start = 0;
    while (start < s.size() && isspace((unsigned char)s[start])) start++;
    return s.substr(start);
}

static vector<string> list_dir_names(const string& path, const string& prefix = "") {
    vector<string> names;
    DIR* dir = opendir(path.c_str());
    if (!dir) return names;
    while (dirent* ent = readdir(dir)) {
        string name = ent->d_name;
        if (name == "." || name == "..") continue;
        if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
        names.push_back(name);
    }
    closedir(dir);
    sort(names.begin(), names.end());
    return names;
}

static string json_escape_string(const string& input) {
    string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if ((unsigned char)c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                out += buf;
            } else {
                out += c;
            }
            break;
        }
    }
    return out;
}

static void sample_cpu_percent(SystemResourceSnapshot& s) {
    ifstream f("/proc/stat");
    string cpu_label;
    long long user = 0, nice = 0, sys = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
    if (!(f >> cpu_label >> user >> nice >> sys >> idle >> iowait >> irq >> softirq)) {
        s.cpu_percent = -1.0;
        return;
    }
    long long idle_all = idle + iowait;
    long long total = user + nice + sys + idle + iowait + irq + softirq;
    if (g_prev_cpu_total > 0) {
        long long dt = total - g_prev_cpu_total;
        long long di = idle_all - g_prev_cpu_idle;
        s.cpu_percent = dt > 0 ? 100.0 * (double)(dt - di) / (double)dt : -1.0;
    }
    g_prev_cpu_total = total;
    g_prev_cpu_idle = idle_all;
}

static void sample_mem_info(SystemResourceSnapshot& s) {
    ifstream f("/proc/meminfo");
    string key, unit;
    double value = 0.0;
    double total_kb = 0.0, avail_kb = 0.0;
    while (f >> key >> value >> unit) {
        if (key == "MemTotal:") total_kb = value;
        if (key == "MemAvailable:") avail_kb = value;
    }
    if (total_kb > 0.0) {
        s.mem_total_mb = total_kb / 1024.0;
        s.mem_used_mb = (total_kb - avail_kb) / 1024.0;
        s.mem_used_percent = 100.0 * (total_kb - avail_kb) / total_kb;
    }
}

static void sample_process_info(SystemResourceSnapshot& s) {
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            string key, unit;
            double kb = 0.0;
            stringstream ss(line);
            ss >> key >> kb >> unit;
            s.process_rss_mb = kb / 1024.0;
        } else if (line.rfind("Threads:", 0) == 0) {
            string key;
            int threads = -1;
            stringstream ss(line);
            ss >> key >> threads;
            s.process_threads = threads;
        }
    }
}

static void sample_npu_info(SystemResourceSnapshot& s) {
    vector<string> entries = list_dir_names("/sys/class/devfreq");
    entries.insert(entries.begin(), "fdab0000.npu");
    for (const string& e : entries) {
        string lower = e;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("npu") == string::npos && e != "fdab0000.npu") continue;
        string base = "/sys/class/devfreq/" + e + "/";
        string freq = read_text_file_trimmed(base + "cur_freq");
        if (freq.empty()) freq = read_text_file_trimmed(base + "scaling_cur_freq");
        if (!freq.empty()) {
            s.npu_freq_mhz = atof(freq.c_str()) / 1000000.0;
        }
        s.npu_governor = read_text_file_trimmed(base + "governor");
        string load = read_text_file_trimmed(base + "load");
        if (!load.empty()) {
            string num;
            for (char c : load) {
                if (isdigit((unsigned char)c) || c == '.') num.push_back(c);
                else if (!num.empty()) break;
            }
            if (!num.empty()) s.npu_load_percent = atof(num.c_str());
        }
        if (s.npu_freq_mhz >= 0.0 || s.npu_load_percent >= 0.0 || !s.npu_governor.empty()) {
            break;
        }
    }
}

static void sample_thermal_info(SystemResourceSnapshot& s) {
    vector<string> zones = list_dir_names("/sys/class/thermal", "thermal_zone");
    vector<string> out;
    for (const string& z : zones) {
        string base = "/sys/class/thermal/" + z + "/";
        string type = read_text_file_trimmed(base + "type");
        string temp = read_text_file_trimmed(base + "temp");
        if (temp.empty()) continue;
        char buf[128];
        snprintf(buf, sizeof(buf), "%s:%.1fC", (type.empty() ? z : type).c_str(), atof(temp.c_str()) / 1000.0);
        out.push_back(buf);
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (i) s.temp_summary += "|";
        s.temp_summary += out[i];
    }
}

static void sample_cpu_freq_info(SystemResourceSnapshot& s) {
    vector<string> policies = list_dir_names("/sys/devices/system/cpu/cpufreq", "policy");
    vector<string> out;
    for (const string& p : policies) {
        string v = read_text_file_trimmed("/sys/devices/system/cpu/cpufreq/" + p + "/scaling_cur_freq");
        if (v.empty()) continue;
        char buf[128];
        snprintf(buf, sizeof(buf), "%s:%.0fMHz", p.c_str(), atof(v.c_str()) / 1000.0);
        out.push_back(buf);
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (i) s.cpu_freq_summary += "|";
        s.cpu_freq_summary += out[i];
    }
}

static SystemResourceSnapshot get_system_resource_snapshot() {
    lock_guard<mutex> lock(g_system_resource_mtx);
    long long t = now_ms();
    if (t - g_system_resource_last_ms < 1000) {
        return g_system_resource_cache;
    }

    SystemResourceSnapshot s;
    sample_cpu_percent(s);
    sample_mem_info(s);
    sample_npu_info(s);
    sample_thermal_info(s);
    sample_cpu_freq_info(s);
    sample_process_info(s);
    g_system_resource_cache = s;
    g_system_resource_last_ms = t;
    return g_system_resource_cache;
}

static string system_resource_json_fragment() {
    SystemResourceSnapshot s = get_system_resource_snapshot();
    ostringstream os;
    os << fixed << setprecision(2)
       << "\"cpu_percent\":" << s.cpu_percent << ","
       << "\"mem_used_percent\":" << s.mem_used_percent << ","
       << "\"mem_used_mb\":" << s.mem_used_mb << ","
       << "\"mem_total_mb\":" << s.mem_total_mb << ","
       << "\"npu_freq_mhz\":" << s.npu_freq_mhz << ","
       << "\"npu_load_percent\":" << s.npu_load_percent << ","
       << "\"npu_governor\":\"" << json_escape_string(s.npu_governor) << "\","
       << "\"temp_summary\":\"" << json_escape_string(s.temp_summary) << "\","
       << "\"cpu_freq_summary\":\"" << json_escape_string(s.cpu_freq_summary) << "\","
       << "\"process_rss_mb\":" << s.process_rss_mb << ","
       << "\"process_threads\":" << s.process_threads << ",";
    return os.str();
}

float clampf(float v, float lo, float hi) {
    return max(lo, min(hi, v));
}

template <typename T>
T clamp_val(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

float zoom_to_ratio(int absolute_zoom);
pair<float, float> interp_fov(float zoom_ratio);

double wrap_deg360(double angle) {
    double r = fmod(angle, 360.0);
    if (r < 0.0) r += 360.0;
    return r;
}

double circular_error_deg(double target, double current) {
    double e = fmod(target - current + 540.0, 360.0) - 180.0;
    return e;
}

bool valid_deg360(double angle) {
    return isfinite(angle) && angle >= 0.0 && angle < 360.0;
}

double circular_mean_deg(const vector<double>& angles) {
    double s = 0.0;
    double c = 0.0;
    int n = 0;
    for (double a : angles) {
        if (!valid_deg360(a)) continue;
        double r = a * CV_PI / 180.0;
        s += sin(r);
        c += cos(r);
        n++;
    }
    if (n <= 0) return 99999.0;
    return wrap_deg360(atan2(s / n, c / n) * 180.0 / CV_PI);
}

double circular_std_deg(const vector<double>& angles) {
    double s = 0.0;
    double c = 0.0;
    int n = 0;
    for (double a : angles) {
        if (!valid_deg360(a)) continue;
        double r = a * CV_PI / 180.0;
        s += sin(r);
        c += cos(r);
        n++;
    }
    if (n <= 1) return n == 1 ? 0.0 : 99999.0;
    double r = sqrt((s / n) * (s / n) + (c / n) * (c / n));
    r = max(1e-6, min(1.0, r));
    return sqrt(max(0.0, -2.0 * log(r))) * 180.0 / CV_PI;
}

double hik_azimuth_to_deg(int azimuth_unit) {
    return wrap_deg360(azimuth_unit / 10.0);
}

string fusion_confidence_label(double confidence) {
    if (confidence >= AUDIO_OFFSET_HIGH_CONF) return "high";
    if (confidence >= AUDIO_OFFSET_BOOTSTRAP_CONF) return "bootstrap";
    return "uncalibrated";
}

string fusion_calibration_state(double confidence, int samples) {
    if (confidence >= AUDIO_OFFSET_HIGH_CONF &&
        samples >= AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES) {
        return "high_confidence_ready";
    }
    if (confidence >= AUDIO_OFFSET_BOOTSTRAP_CONF &&
        samples >= AUDIO_OFFSET_BOOTSTRAP_MIN_SAMPLES) {
        return "bootstrap_ready";
    }
    if (samples > 0 || confidence > 0.0) {
        return "auto_calibrating";
    }
    return "uncalibrated";
}

double visual_target_azimuth_deg(const SharedDetection& det, const PTZState& st) {
    int fw = max(1, det.fw);
    float dx = det.cx - fw * 0.5f;
    float zoom_ratio = zoom_to_ratio(st.absolute_zoom);
    auto fov = interp_fov(zoom_ratio);
    double pan_err_deg = (double)dx / (double)fw * (double)fov.first;
    return wrap_deg360(hik_azimuth_to_deg(st.azimuth) + ABS_PAN_DIR * pan_err_deg);
}

double json_get_double(const string& json, const string& key, double default_value) {
    string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == string::npos) return default_value;
    p = json.find(':', p + needle.size());
    if (p == string::npos) return default_value;
    p++;
    while (p < json.size() && isspace((unsigned char)json[p])) p++;
    if (p < json.size() && json[p] == '"') p++;
    char* end = nullptr;
    double value = strtod(json.c_str() + p, &end);
    if (end == json.c_str() + p || !isfinite(value)) return default_value;
    return value;
}

bool json_has_key(const string& json, const string& key) {
    string needle = "\"" + key + "\"";
    return json.find(needle) != string::npos;
}

bool json_get_bool(const string& json, const string& key, bool default_value) {
    string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == string::npos) return default_value;
    p = json.find(':', p + needle.size());
    if (p == string::npos) return default_value;
    p++;
    while (p < json.size() && isspace((unsigned char)json[p])) p++;
    if (p >= json.size()) return default_value;
    if (json.compare(p, 4, "true") == 0) return true;
    if (json.compare(p, 5, "false") == 0) return false;
    double v = json_get_double(json, key, default_value ? 1.0 : 0.0);
    return v != 0.0;
}

string json_get_string(const string& json, const string& key, const string& default_value) {
    string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == string::npos) return default_value;
    p = json.find(':', p + needle.size());
    if (p == string::npos) return default_value;
    p++;
    while (p < json.size() && isspace((unsigned char)json[p])) p++;
    if (p >= json.size() || json[p] != '"') return default_value;
    p++;
    size_t end = json.find('"', p);
    if (end == string::npos) return default_value;
    return json.substr(p, end - p);
}

static int manual_owner_code_from_source(const string& source) {
    if (source == "windows" || source == "windows_qt") return 1;
    if (source == "elf2" || source == "board_qt" || source == "local_ui") return 2;
    if (source == "keyboard") return 3;
    return source.empty() ? 4 : 4;
}

static string manual_owner_name(int code) {
    if (code == 1) return "windows_qt";
    if (code == 2) return "board_qt";
    if (code == 3) return "local_keyboard";
    if (code == 4) return "other";
    return "none";
}

static atomic<long long>& last_seq_for_control_source(const string& source) {
    if (source == "windows" || source == "windows_qt") return g_qt_windows_last_seq;
    if (source == "elf2" || source == "board_qt" || source == "local_ui") return g_qt_board_last_seq;
    return g_qt_other_last_seq;
}

bool ensure_parent_dir(const string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == string::npos) return true;

    string dir = path.substr(0, slash);
    if (dir.empty()) return true;

    string current;
    size_t start = 0;
    if (dir[0] == '/') {
        current = "/";
        start = 1;
    }

    while (start <= dir.size()) {
        size_t pos = dir.find('/', start);
        string part = dir.substr(start, pos == string::npos ? string::npos : pos - start);
        if (!part.empty()) {
            if (!current.empty() && current != "/") current += "/";
            current += part;

            struct stat st {};
            if (stat(current.c_str(), &st) != 0) {
                if (mkdir(current.c_str(), 0755) != 0) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (pos == string::npos) break;
        start = pos + 1;
    }
    return true;
}

static string control_params_json(const string& source) {
    ostringstream ofs;
    ofs << "{\n"
        << "  \"created_by\": \"Codex\",\n"
        << "  \"updated_ts_ms\": " << metric_now_ms_cpp() << ",\n"
        << "  \"source\": \"" << json_escape_string(source) << "\",\n"
        << "  \"detection_mode\": " << g_detection_policy_mode.load() << ",\n"
        << "  \"search_conf\": " << fixed << setprecision(3) << g_search_conf_thresh.load() << ",\n"
        << "  \"track_conf\": " << fixed << setprecision(3) << g_track_conf_thresh.load() << ",\n"
        << "  \"alarm_conf\": " << fixed << setprecision(3) << g_alarm_conf_thresh.load() << ",\n"
        << "  \"search_min_box\": " << fixed << setprecision(3) << g_search_min_box_ratio.load() << ",\n"
        << "  \"track_min_box\": " << fixed << setprecision(3) << g_track_min_box_ratio.load() << ",\n"
        << "  \"alarm_min_box\": " << fixed << setprecision(3) << g_alarm_min_box_ratio.load() << ",\n"
        << "  \"confirm_frames\": " << g_search_confirm_frames.load() << ",\n"
        << "  \"track_confirm_frames\": " << g_track_confirm_frames.load() << ",\n"
        << "  \"alarm_confirm_frames\": " << g_alarm_confirm_frames.load() << ",\n"
        << "  \"target_box_min\": " << fixed << setprecision(3) << g_zoom_target_min_ratio.load() << ",\n"
        << "  \"target_box_ideal_low\": " << fixed << setprecision(3) << g_zoom_target_ideal_low_ratio.load() << ",\n"
        << "  \"target_box_ideal_high\": " << fixed << setprecision(3) << g_zoom_target_ideal_high_ratio.load() << ",\n"
        << "  \"target_box_max\": " << fixed << setprecision(3) << g_zoom_target_max_ratio.load() << ",\n"
        << "  \"min_zoom\": " << fixed << setprecision(2) << g_zoom_min_ratio.load() << ",\n"
        << "  \"max_zoom\": " << fixed << setprecision(2) << g_zoom_max_ratio.load() << ",\n"
        << "  \"lost_search_zoom\": " << fixed << setprecision(2) << g_lost_search_zoom_ratio.load() << ",\n"
        << "  \"auto_track_enabled\": " << (ptz_control_enabled.load() ? 1 : 0) << ",\n"
        << "  \"auto_zoom_enabled\": " << (auto_zoom_enabled.load() ? 1 : 0) << ",\n"
        << "  \"auto_focus_enabled\": " << (g_autofocus_enabled.load() ? 1 : 0) << ",\n"
        << "  \"tracking_mode\": " << g_tracking_mode.load() << ",\n"
        << "  \"pan_gain\": " << fixed << setprecision(3) << g_tracking_pan_gain.load() << ",\n"
        << "  \"tilt_gain\": " << fixed << setprecision(3) << g_tracking_tilt_gain.load() << ",\n"
        << "  \"feedforward_gain\": " << fixed << setprecision(3) << g_tracking_feedforward_gain.load() << ",\n"
        << "  \"max_speed\": " << g_ptz_max_speed.load() << ",\n"
        << "  \"high_zoom_max_speed\": " << g_ptz_max_speed_high_zoom.load() << ",\n"
        << "  \"center_dead_zone\": " << fixed << setprecision(3) << g_center_dead_zone_x_ratio.load() << ",\n"
        << "  \"zoom_step_in\": " << g_zoom_step_in_speed.load() << ",\n"
        << "  \"zoom_step_out\": " << g_zoom_step_out_speed.load() << ",\n"
        << "  \"zoom_cooldown_ms\": " << (int)round(g_zoom_cmd_period_sec.load() * 1000.0f) << ",\n"
        << "  \"lost_zoom_out_step\": " << g_zoom_search_out_speed.load() << ",\n"
        << "  \"focus_cooldown_ms\": " << (int)round(g_autofocus_cooldown_sec.load() * 1000.0f) << "\n"
        << "}\n";
    return ofs.str();
}

static bool write_text_atomic(const string& path, const string& text) {
    if (path.empty()) return false;
    if (!ensure_parent_dir(path)) return false;
    string tmp = path + ".tmp";
    ofstream ofs(tmp);
    if (!ofs.good()) return false;
    ofs << text;
    ofs.close();
    return rename(tmp.c_str(), path.c_str()) == 0;
}

static void save_control_params_snapshot(const string& source) {
    string text = control_params_json(source);
    if (!g_control_params_path.empty() && !write_text_atomic(g_control_params_path, text)) {
        cerr << "[CTRL_PARAMS] failed to save " << g_control_params_path << endl;
    }
    if (!g_control_params_session_snapshot_path.empty() &&
        !write_text_atomic(g_control_params_session_snapshot_path, text)) {
        cerr << "[CTRL_PARAMS] failed to save session snapshot "
             << g_control_params_session_snapshot_path << endl;
    }
}

void save_fusion_calibration() {
    FusionCalibration c;
    {
        lock_guard<mutex> lock(fusion_mtx);
        c = fusion_calib;
    }
    if (!ensure_parent_dir(g_fusion_calibration_path)) {
        cerr << "[FUSION] cannot create calibration directory: " << g_fusion_calibration_path << endl;
        return;
    }
    string tmp = g_fusion_calibration_path + ".tmp";
    ofstream ofs(tmp);
    if (!ofs.good()) return;
    ofs << "{\n"
        << "  \"mic_to_camera_offset_deg\": " << c.offset_deg << ",\n"
        << "  \"offset_confidence\": " << c.confidence << ",\n"
        << "  \"total_samples\": " << c.total_samples << ",\n"
        << "  \"stable_samples\": " << c.stable_samples << ",\n"
        << "  \"updated_ts_ms\": " << metric_now_ms_cpp() << "\n"
        << "}\n";
    ofs.close();
    rename(tmp.c_str(), g_fusion_calibration_path.c_str());
}

void load_fusion_calibration() {
    ifstream ifs(g_fusion_calibration_path);
    if (!ifs.good()) {
        cout << "[FUSION] no previous calibration file: " << g_fusion_calibration_path << endl;
        return;
    }
    string text((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    double offset = json_get_double(text, "mic_to_camera_offset_deg", 0.0);
    double prev_conf = json_get_double(text, "offset_confidence", 0.0);
    if (!g_load_audio_offset_prior) {
        cout << "[FUSION] ignored previous audio offset prior: "
             << wrap_deg360(offset) << " deg, previous confidence=" << prev_conf
             << ". Live calibration is required for this run." << endl;
        return;
    }
    {
        lock_guard<mutex> lock(fusion_mtx);
        fusion_calib.offset_deg = wrap_deg360(offset);
        fusion_calib.confidence = prev_conf >= 0.75 ? 0.25 : 0.0;
        fusion_calib.total_samples = 0;
        fusion_calib.stable_samples = 0;
    }
    cout << "[FUSION] loaded previous offset as low-confidence prior: "
         << wrap_deg360(offset) << " deg, previous confidence=" << prev_conf << endl;
}

void save_runtime_learning_profile_locked() {
    if (!g_runtime_learning_enabled) return;
    if (!ensure_parent_dir(g_learning_profile_path)) {
        cerr << "[LEARN] cannot create profile directory: " << g_learning_profile_path << endl;
        return;
    }

    runtime_profile.updated_ts_ms = metric_now_ms_cpp();
    string tmp = g_learning_profile_path + ".tmp";
    ofstream ofs(tmp);
    if (!ofs.good()) return;
    ofs << "{\n"
        << "  \"best_offset_deg\": " << runtime_profile.best_offset_deg << ",\n"
        << "  \"best_audio_error_ema\": " << runtime_profile.best_audio_error_ema << ",\n"
        << "  \"audio_samples\": " << runtime_profile.audio_samples << ",\n"
        << "  \"best_center_error_ema\": " << runtime_profile.best_center_error_ema << ",\n"
        << "  \"track_samples\": " << runtime_profile.track_samples << ",\n"
        << "  \"ptz_max_speed\": " << runtime_profile.ptz_max_speed << ",\n"
        << "  \"ptz_max_speed_high_zoom\": " << runtime_profile.ptz_max_speed_high_zoom << ",\n"
        << "  \"center_dead_zone_x_ratio\": " << runtime_profile.center_dead_zone_x_ratio << ",\n"
        << "  \"center_dead_zone_y_ratio\": " << runtime_profile.center_dead_zone_y_ratio << ",\n"
        << "  \"updated_ts_ms\": " << runtime_profile.updated_ts_ms << "\n"
        << "}\n";
    ofs.close();
    rename(tmp.c_str(), g_learning_profile_path.c_str());
}

void load_runtime_learning_profile() {
    if (!g_runtime_learning_enabled) return;
    ifstream ifs(g_learning_profile_path);
    if (!ifs.good()) {
        cout << "[LEARN] no previous runtime profile: " << g_learning_profile_path << endl;
        return;
    }

    string text((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    RuntimeLearningProfile p;
    p.loaded = true;
    p.best_offset_deg = wrap_deg360(json_get_double(text, "best_offset_deg", 0.0));
    p.best_audio_error_ema = json_get_double(text, "best_audio_error_ema", 99999.0);
    p.audio_samples = (int)json_get_double(text, "audio_samples", 0.0);
    p.best_center_error_ema = json_get_double(text, "best_center_error_ema", 99999.0);
    p.track_samples = (int)json_get_double(text, "track_samples", 0.0);
    p.ptz_max_speed = clamp_val((int)json_get_double(text, "ptz_max_speed", PTZ_MAX_SPEED), 20, 95);
    p.ptz_max_speed_high_zoom = clamp_val((int)json_get_double(text, "ptz_max_speed_high_zoom", PTZ_MAX_SPEED_HIGH_ZOOM), 10, 75);
    p.center_dead_zone_x_ratio = clampf((float)json_get_double(text, "center_dead_zone_x_ratio", CENTER_DEAD_ZONE_X_RATIO), 0.035f, 0.095f);
    p.center_dead_zone_y_ratio = clampf((float)json_get_double(text, "center_dead_zone_y_ratio", CENTER_DEAD_ZONE_Y_RATIO), 0.035f, 0.095f);
    p.updated_ts_ms = (long long)json_get_double(text, "updated_ts_ms", 0.0);

    double loaded_audio_offset = p.best_offset_deg;
    double loaded_audio_error = p.best_audio_error_ema;
    int loaded_audio_samples = p.audio_samples;
    if (!g_load_audio_offset_prior) {
        p.best_offset_deg = 0.0;
        p.best_audio_error_ema = 99999.0;
        p.audio_samples = 0;
    }

    {
        lock_guard<mutex> lock(runtime_learning_mtx);
        runtime_profile = p;
        runtime_learning.last_action = "loaded_profile";
    }

    // 历史 offset 只能作为低置信先验；硬件移动后必须靠新共同观测重新收敛。
    if (g_load_audio_offset_prior && p.best_audio_error_ema <= 12.0 && p.audio_samples >= 12) {
        lock_guard<mutex> lock(fusion_mtx);
        fusion_calib.offset_deg = p.best_offset_deg;
        fusion_calib.confidence = max(fusion_calib.confidence, 0.30);
        cout << "[LEARN] loaded best audio offset prior: " << p.best_offset_deg
             << " deg, best_error_ema=" << p.best_audio_error_ema << endl;
    } else if (!g_load_audio_offset_prior && loaded_audio_error <= 12.0 && loaded_audio_samples >= 12) {
        cout << "[LEARN] ignored saved audio offset prior: " << loaded_audio_offset
             << " deg, best_error_ema=" << loaded_audio_error
             << ", samples=" << loaded_audio_samples
             << ". Live visual-acoustic calibration is required." << endl;
    }

    if (p.best_center_error_ema <= 0.10 && p.track_samples >= 60) {
        g_ptz_max_speed.store(p.ptz_max_speed);
        g_ptz_max_speed_high_zoom.store(p.ptz_max_speed_high_zoom);
        g_center_dead_zone_x_ratio.store((float)p.center_dead_zone_x_ratio);
        g_center_dead_zone_y_ratio.store((float)p.center_dead_zone_y_ratio);
        cout << "[LEARN] loaded best tracking profile: center_error_ema="
             << p.best_center_error_ema
             << ", ptz_max=" << p.ptz_max_speed
             << ", high_zoom=" << p.ptz_max_speed_high_zoom << endl;
    }
}

void update_audio_runtime_learning(double abs_err,
                                   double current_offset,
                                   double current_confidence) {
    if (!g_runtime_learning_enabled || !isfinite(abs_err)) return;

    bool restore_best = false;
    {
        lock_guard<mutex> lock(runtime_learning_mtx);
        double& ema = runtime_learning.audio_error_ema;
        if (ema > 90000.0) ema = abs_err;
        else ema = 0.90 * ema + 0.10 * abs_err;
        runtime_learning.audio_samples++;

        if (abs_err <= 18.0) {
            runtime_learning.audio_bad_samples = 0;
        } else if (abs_err >= 32.0) {
            runtime_learning.audio_bad_samples++;
        }

        bool better_profile = current_confidence >= AUDIO_OFFSET_HIGH_CONF &&
                              runtime_learning.audio_samples >= 12 &&
                              ema <= 11.0 &&
                              (!runtime_profile.loaded ||
                               runtime_profile.best_audio_error_ema > 90000.0 ||
                               ema + 0.8 < runtime_profile.best_audio_error_ema);
        if (better_profile) {
            runtime_profile.loaded = true;
            runtime_profile.best_offset_deg = wrap_deg360(current_offset);
            runtime_profile.best_audio_error_ema = ema;
            runtime_profile.audio_samples = runtime_learning.audio_samples;
            runtime_profile.ptz_max_speed = g_ptz_max_speed.load();
            runtime_profile.ptz_max_speed_high_zoom = g_ptz_max_speed_high_zoom.load();
            runtime_profile.center_dead_zone_x_ratio = g_center_dead_zone_x_ratio.load();
            runtime_profile.center_dead_zone_y_ratio = g_center_dead_zone_y_ratio.load();
            runtime_learning.last_action = "saved_best_audio";
            save_runtime_learning_profile_locked();
            cout << "\n[LEARN] saved best audio offset: offset="
                 << runtime_profile.best_offset_deg
                 << " error_ema=" << runtime_profile.best_audio_error_ema << endl;
        }

        restore_best = runtime_profile.loaded &&
                       runtime_profile.best_audio_error_ema <= 12.0 &&
                       runtime_learning.audio_bad_samples >= 8 &&
                       ema >= runtime_profile.best_audio_error_ema + 12.0;
        if (restore_best) {
            runtime_learning.audio_bad_samples = 0;
            runtime_learning.last_action = "restore_best_audio";
        }
    }

    if (restore_best) {
        double best_offset = 0.0;
        {
            lock_guard<mutex> lock(runtime_learning_mtx);
            best_offset = runtime_profile.best_offset_deg;
        }
        lock_guard<mutex> lock(fusion_mtx);
        fusion_calib.offset_deg = wrap_deg360(best_offset);
        fusion_calib.confidence = max(0.35, min(fusion_calib.confidence, 0.65));
        cout << "\n[LEARN] restored best audio offset because current error degraded: "
             << best_offset << endl;
    }
}

void update_tracking_runtime_learning(bool det_fresh,
                                      float dx,
                                      float dy,
                                      int fw,
                                      int fh,
                                      double now) {
    if (!g_runtime_learning_enabled) return;

    bool save_profile = false;
    bool restore_profile = false;

    {
        lock_guard<mutex> lock(runtime_learning_mtx);

        if (!det_fresh) {
            if (runtime_learning.track_samples > 10) {
                runtime_learning.track_bad_samples++;
            }
            restore_profile = runtime_profile.loaded &&
                              runtime_profile.best_center_error_ema <= 0.10 &&
                              runtime_learning.track_bad_samples >= 18;
            if (restore_profile) {
                runtime_learning.track_bad_samples = 0;
                runtime_learning.last_action = "restore_best_tracking";
            }
        } else {
            float nx = fabs(dx) / max(1.0f, fw * 0.5f);
            float ny = fabs(dy) / max(1.0f, fh * 0.5f);
            double center_err = sqrt((double)nx * nx + (double)ny * ny);
            if (runtime_learning.center_error_ema > 90000.0) {
                runtime_learning.center_error_ema = center_err;
            } else {
                runtime_learning.center_error_ema =
                    0.92 * runtime_learning.center_error_ema + 0.08 * center_err;
            }
            runtime_learning.track_samples++;

            int sign = 0;
            if (fabs(dx) > fw * 0.08f) sign = dx > 0 ? 1 : -1;
            if (sign != 0 && runtime_learning.last_dx_sign != 0 &&
                sign != runtime_learning.last_dx_sign) {
                runtime_learning.sign_flips++;
            }
            if (sign != 0) runtime_learning.last_dx_sign = sign;

            if (center_err <= 0.12) {
                runtime_learning.stable_track_samples++;
                runtime_learning.track_bad_samples = max(0, runtime_learning.track_bad_samples - 1);
            } else if (center_err >= 0.34) {
                runtime_learning.track_bad_samples++;
            }

            if (now - runtime_learning.last_track_adjust_time >= 4.0) {
                if (runtime_learning.sign_flips >= 5 &&
                    runtime_learning.center_error_ema <= 0.24) {
                    g_ptz_max_speed.store(clamp_val(g_ptz_max_speed.load() - 4, 56, 100));
                    g_ptz_max_speed_high_zoom.store(clamp_val(g_ptz_max_speed_high_zoom.load() - 3, 36, 72));
                    g_center_dead_zone_x_ratio.store(clampf(g_center_dead_zone_x_ratio.load() + 0.004f, 0.035f, 0.070f));
                    g_center_dead_zone_y_ratio.store(clampf(g_center_dead_zone_y_ratio.load() + 0.004f, 0.035f, 0.070f));
                    runtime_learning.sign_flips = 0;
                    runtime_learning.last_track_adjust_time = now;
                    runtime_learning.last_action = "dampen_tracking";
                    cout << "\n[LEARN] dampen tracking to reduce overshoot: ptz_max="
                         << g_ptz_max_speed.load()
                         << " high_zoom=" << g_ptz_max_speed_high_zoom.load()
                         << " dead=" << g_center_dead_zone_x_ratio.load()
                         << "/" << g_center_dead_zone_y_ratio.load() << endl;
                } else if (runtime_learning.center_error_ema >= 0.28 &&
                           runtime_learning.sign_flips <= 1 &&
                           runtime_learning.track_bad_samples <= 3) {
                    g_ptz_max_speed.store(clamp_val(g_ptz_max_speed.load() + 4, 56, 100));
                    g_ptz_max_speed_high_zoom.store(clamp_val(g_ptz_max_speed_high_zoom.load() + 3, 36, 72));
                    runtime_learning.last_track_adjust_time = now;
                    runtime_learning.last_action = "speedup_tracking";
                    cout << "\n[LEARN] speed up tracking for persistent center error: ptz_max="
                         << g_ptz_max_speed.load()
                         << " high_zoom=" << g_ptz_max_speed_high_zoom.load() << endl;
                }
            }

            save_profile = runtime_learning.stable_track_samples >= 80 &&
                           runtime_learning.center_error_ema <= 0.10 &&
                           (!runtime_profile.loaded ||
                            runtime_profile.best_center_error_ema > 90000.0 ||
                            runtime_learning.center_error_ema + 0.01 < runtime_profile.best_center_error_ema);
            if (save_profile) {
                runtime_profile.loaded = true;
                runtime_profile.best_center_error_ema = runtime_learning.center_error_ema;
                runtime_profile.track_samples = runtime_learning.track_samples;
                runtime_profile.ptz_max_speed = g_ptz_max_speed.load();
                runtime_profile.ptz_max_speed_high_zoom = g_ptz_max_speed_high_zoom.load();
                runtime_profile.center_dead_zone_x_ratio = g_center_dead_zone_x_ratio.load();
                runtime_profile.center_dead_zone_y_ratio = g_center_dead_zone_y_ratio.load();
                runtime_learning.stable_track_samples = 0;
                runtime_learning.last_action = "saved_best_tracking";
                save_runtime_learning_profile_locked();
                cout << "\n[LEARN] saved best tracking profile: center_error_ema="
                     << runtime_profile.best_center_error_ema
                     << " ptz_max=" << runtime_profile.ptz_max_speed << endl;
            }
        }
    }

    if (restore_profile) {
        RuntimeLearningProfile p;
        {
            lock_guard<mutex> lock(runtime_learning_mtx);
            p = runtime_profile;
        }
        g_ptz_max_speed.store(p.ptz_max_speed);
        g_ptz_max_speed_high_zoom.store(p.ptz_max_speed_high_zoom);
        g_center_dead_zone_x_ratio.store((float)p.center_dead_zone_x_ratio);
        g_center_dead_zone_y_ratio.store((float)p.center_dead_zone_y_ratio);
        cout << "\n[LEARN] restored best tracking profile: center_error_ema="
             << p.best_center_error_ema << endl;
    }
}

string runtime_learning_json_suffix() {
    RuntimeLearningState s;
    RuntimeLearningProfile p;
    {
        lock_guard<mutex> lock(runtime_learning_mtx);
        s = runtime_learning;
        p = runtime_profile;
    }
    char buf[768];
    snprintf(buf, sizeof(buf),
             ",\"learning_enabled\":%d,"
             "\"learning_action\":\"%s\","
             "\"learning_audio_error_ema\":%.3f,"
             "\"learning_center_error_ema\":%.4f,"
             "\"learning_best_audio_error_ema\":%.3f,"
             "\"learning_best_center_error_ema\":%.4f,"
             "\"learning_profile_loaded\":%d",
             g_runtime_learning_enabled ? 1 : 0,
             s.last_action.c_str(),
             s.audio_error_ema,
             s.center_error_ema,
             p.best_audio_error_ema,
             p.best_center_error_ema,
             p.loaded ? 1 : 0);
    return string(buf);
}

void publish_fusion_telemetry(const FusionTelemetry& telemetry) {
    lock_guard<mutex> lock(fusion_mtx);
    fusion_telemetry = telemetry;
}

string fusion_json_suffix() {
    string learning_suffix = runtime_learning_json_suffix();
    FusionTelemetry t;
    {
        lock_guard<mutex> lock(fusion_mtx);
        t = fusion_telemetry;
        t.mic_to_camera_offset_deg = fusion_calib.offset_deg;
        t.offset_confidence = fusion_calib.confidence;
        t.offset_confidence_label = fusion_confidence_label(fusion_calib.confidence);
        t.calibration_samples = fusion_calib.total_samples;
    }
    char buf[8192];
    snprintf(buf, sizeof(buf),
             ",\"detection_enabled\":%d,"
             "\"fusion_mode\":\"%s\","
             "\"fusion_state\":\"%s\","
             "\"control_owner\":\"%s\","
             "\"control_source\":\"%s\","
             "\"manual_owner\":\"%s\","
             "\"search_state\":\"%s\","
             "\"audio_guidance_state\":\"%s\","
             "\"audio_reject_reason\":\"%s\","
             "\"audio_calibration_state\":\"%s\","
             "\"audio_guided\":%d,"
             "\"camera_azimuth_deg\":%.3f,"
             "\"audio_pan_err_deg\":%.3f,"
             "\"audio_visual_angle_error_deg\":%.3f,"
             "\"visual_azimuth_deg\":%.3f,"
             "\"audio_world_azimuth_deg\":%.3f,"
             "\"mic_to_camera_offset_deg\":%.3f,"
             "\"offset_confidence\":%.3f,"
             "\"offset_confidence_label\":\"%s\","
             "\"calibration_samples\":%d,"
             "\"doa_stability\":%.3f,"
             "\"pan_speed\":%d,"
             "\"tilt_speed\":%d,"
             "\"zoom_speed\":%d,"
             "\"auto_track_enabled\":%d,"
             "\"auto_zoom_enabled\":%d,"
             "\"auto_focus_enabled\":%d,"
             "\"manual_override_active\":%d,"
             "\"manual_hold_remaining_ms\":%lld,"
             "\"emergency_stop_active\":%d,"
             "\"target_age_ms\":%lld,"
             "\"target_state\":\"%s\","
             "\"ptz_block_reason\":\"%s\","
             "\"auto_block_reason\":\"%s\","
             "\"stale_command_dropped\":%lld,"
             "\"high_zoom_limited\":%d,"
             "\"max_speed_cap\":%d,"
             "\"actual_pan_speed\":%d,"
             "\"actual_tilt_speed\":%d,"
             "\"zoom_ratio\":%.3f,"
             "\"target_max_side\":%.4f,"
             "\"target_max_side_ema\":%.4f,"
             "\"zoom_state\":\"%s\","
             "\"zoom_action\":\"%s\","
             "\"zoom_reason\":\"%s\","
             "\"target_near_edge\":%d,"
             "\"target_stable\":%d,"
             "\"target_lost\":%d,"
             "\"lost_duration_sec\":%.3f,"
             "\"search_zoom_level\":%.3f,"
             "\"focus_triggered\":%d,"
             "\"focus_reason\":\"%s\","
             "\"focus_state\":\"%s\","
             "\"last_focus_request_ms\":%lld,"
             "\"focus_cooldown_remaining_ms\":%lld,"
             "\"vertical_sweep_delta_deg\":%.3f%s",
             g_detection_enabled.load() ? 1 : 0,
             t.mode.c_str(),
             t.fusion_state.c_str(),
             t.control_owner.c_str(),
             t.control_source.c_str(),
             t.manual_owner.c_str(),
             t.search_state.c_str(),
             t.audio_guidance_state.c_str(),
             t.audio_reject_reason.c_str(),
             t.audio_calibration_state.c_str(),
             t.audio_guided ? 1 : 0,
             t.camera_azimuth_deg,
             t.audio_pan_err_deg,
             t.audio_visual_angle_error_deg,
             t.visual_azimuth_deg,
             t.audio_world_azimuth_deg,
             t.mic_to_camera_offset_deg,
             t.offset_confidence,
             t.offset_confidence_label.c_str(),
             t.calibration_samples,
             t.doa_stability,
             t.pan_speed,
             t.tilt_speed,
             t.zoom_speed,
             t.auto_track_enabled ? 1 : 0,
             t.auto_zoom_enabled ? 1 : 0,
             t.auto_focus_enabled ? 1 : 0,
             t.manual_override_active ? 1 : 0,
             t.manual_hold_remaining_ms,
             t.emergency_stop_active ? 1 : 0,
             t.target_age_ms,
             t.target_state.c_str(),
             t.ptz_block_reason.c_str(),
             t.ptz_block_reason.c_str(),
             t.stale_command_dropped,
             t.high_zoom_limited ? 1 : 0,
             t.max_speed_cap,
             t.actual_pan_speed,
             t.actual_tilt_speed,
             t.zoom_ratio,
             t.target_max_side,
             t.target_max_side_ema,
             t.zoom_state.c_str(),
             t.zoom_action.c_str(),
             t.zoom_reason.c_str(),
             t.target_near_edge ? 1 : 0,
             t.target_stable ? 1 : 0,
             t.target_lost ? 1 : 0,
             t.lost_duration_sec,
             t.search_zoom_level,
             t.focus_triggered ? 1 : 0,
             t.focus_reason.c_str(),
             t.focus_state.c_str(),
             t.last_focus_request_ms,
             t.focus_cooldown_remaining_ms,
             t.vertical_sweep_delta_deg,
             learning_suffix.c_str());
    return string(buf);
}

void print_runtime_tuning_params() {
    cout << "\n========== RUNTIME TUNING PARAMS ==========" << endl;
    cout << "PTZ auto min/max/highZoom = "
         << g_ptz_min_speed.load() << " / "
         << g_ptz_max_speed.load() << " / "
         << g_ptz_max_speed_high_zoom.load() << endl;
    cout << "PTZ curve = " << g_ptz_speed_curve.load()
         << ", period = " << g_ptz_control_period_sec.load() << " s" << endl;
    cout << "Dead zone ratio x/y = "
         << g_center_dead_zone_x_ratio.load() << " / "
         << g_center_dead_zone_y_ratio.load() << endl;
    cout << "Zoom IN min/max = "
         << g_zoom_min_speed.load() << " / "
         << g_zoom_max_speed.load() << endl;
    cout << "Zoom OUT min/max = "
         << g_zoom_out_min_speed.load() << " / "
         << g_zoom_out_max_speed.load() << endl;
    cout << "Lost-search zoom-out = " << g_zoom_search_out_speed.load()
         << ", zoom period = " << g_zoom_cmd_period_sec.load() << " s"
         << ", flip cooldown = " << g_zoom_flip_cooldown_sec.load() << " s" << endl;
    cout << "Zoom stable range: expand<" << g_zoom_target_min_ratio.load()
         << ", hold=" << g_zoom_target_ideal_low_ratio.load()
         << "~" << g_zoom_target_ideal_high_ratio.load()
         << ", shrink>" << g_zoom_target_max_ratio.load() << endl;
    cout << "Manual WSAD PTZ speed = " << g_manual_ptz_speed.load()
         << ", manual +/- zoom speed = " << g_manual_zoom_speed.load()
         << ", manual send period = " << MANUAL_PTZ_SEND_PERIOD_SEC << " s"
         << ", manual hold PTZ/zoom = " << MANUAL_PTZ_HOLD_MS << "/" << MANUAL_ZOOM_HOLD_MS << " ms"
         << ", stop retry = " << MANUAL_STOP_RETRY_COUNT << endl;
    cout << "\nHotkeys:" << endl;
    cout << "  J / K : auto PTZ max speed -/+5" << endl;
    cout << "  U / I : manual WSAD PTZ speed -/+5" << endl;
    cout << "  N / M : high-zoom PTZ max speed -/+5" << endl;
    cout << "  Z / X : auto zoom IN max speed -/+15" << endl;
    cout << "  , / . : auto zoom OUT max speed -/+25" << endl;
    cout << "  ; / ' : lost-search zoom-out speed -/+15" << endl;
    cout << "  [ / ] : PTZ control period faster/slower" << endl;
    cout << "  9 / 0 : center dead zone smaller/larger" << endl;
    cout << "  H     : print this table" << endl;
    cout << "==========================================\n" << endl;
}

int wrap_azimuth(int az) {
    int r = az % 3600;
    if (r < 0) r += 3600;
    return r;
}

float zoom_to_ratio(int absolute_zoom) {
    float zr = absolute_zoom / 10.0f;
    if (zr < 1.0f) zr = 1.0f;
    if (zr > 37.0f) zr = 37.0f;
    return zr;
}

pair<float, float> interp_fov(float zoom_ratio) {
    float zr = clampf(zoom_ratio, FOV_TABLE.front().zoom, FOV_TABLE.back().zoom);

    for (size_t i = 0; i + 1 < FOV_TABLE.size(); ++i) {
        const auto& a = FOV_TABLE[i];
        const auto& b = FOV_TABLE[i + 1];

        if (a.zoom <= zr && zr <= b.zoom) {
            float t = (log(zr) - log(a.zoom)) / max(1e-6f, log(b.zoom) - log(a.zoom));
            float hfov = a.hfov + t * (b.hfov - a.hfov);
            float vfov = a.vfov + t * (b.vfov - a.vfov);
            return {hfov, vfov};
        }
    }

    return {FOV_TABLE.back().hfov, FOV_TABLE.back().vfov};
}

string shell_capture(const string& cmd) {
    string result;
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return result;

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), fp)) {
        result += buffer;
    }

    pclose(fp);
    return result;
}

static string join_log_path(const string& filename);
static void append_csv_line(const string& filename, const string& header, const string& line);

static string camera_profile_to_string(CameraProfile profile) {
    switch (profile) {
    case CameraProfile::DAY_NORMAL: return "DAY_NORMAL";
    case CameraProfile::DAY_CLOUDY_UAV: return "DAY_CLOUDY_UAV";
    case CameraProfile::FAST_TRACKING: return "FAST_TRACKING";
    case CameraProfile::LOW_LIGHT_KEEP_RGB: return "LOW_LIGHT_KEEP_RGB";
    case CameraProfile::NIGHT_IR_TEST: return "NIGHT_IR_TEST";
    default: return "DAY_CLOUDY_UAV";
    }
}

static CameraProfile current_camera_profile() {
    int v = g_camera_profile.load();
    if (v < (int)CameraProfile::DAY_NORMAL || v > (int)CameraProfile::NIGHT_IR_TEST) {
        return DEFAULT_CAMERA_PROFILE;
    }
    return (CameraProfile)v;
}

static bool camera_profile_from_string(const string& raw, CameraProfile& out_profile) {
    string s = raw;
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)toupper(c); });
    if (s == "DAY_NORMAL" || s == "NORMAL" || s == "DAY") {
        out_profile = CameraProfile::DAY_NORMAL;
        return true;
    }
    if (s == "DAY_CLOUDY_UAV" || s == "CLOUDY" || s == "CLOUDY_UAV") {
        out_profile = CameraProfile::DAY_CLOUDY_UAV;
        return true;
    }
    if (s == "FAST_TRACKING" || s == "FAST") {
        out_profile = CameraProfile::FAST_TRACKING;
        return true;
    }
    if (s == "LOW_LIGHT_KEEP_RGB" || s == "LOW_LIGHT" || s == "KEEP_RGB") {
        out_profile = CameraProfile::LOW_LIGHT_KEEP_RGB;
        return true;
    }
    if (s == "NIGHT_IR_TEST" || s == "NIGHT" || s == "IR") {
        out_profile = CameraProfile::NIGHT_IR_TEST;
        return true;
    }
    return false;
}

static string camera_track_state_snapshot() {
    lock_guard<mutex> lock(g_camera_quality_mtx);
    return g_camera_track_state;
}

static string image_quality_warning_snapshot() {
    lock_guard<mutex> lock(g_camera_quality_mtx);
    return g_image_quality_warning;
}

static string last_profile_switch_reason_snapshot() {
    lock_guard<mutex> lock(g_camera_quality_mtx);
    return g_last_profile_switch_reason;
}

static void set_camera_quality_text(const string& track_state,
                                    const string& warning,
                                    const string& reason = "") {
    lock_guard<mutex> lock(g_camera_quality_mtx);
    if (!track_state.empty()) g_camera_track_state = track_state;
    if (!warning.empty()) g_image_quality_warning = warning;
    if (!reason.empty()) g_last_profile_switch_reason = reason;
}

static string csv_escape_simple(const string& s) {
    bool need_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            need_quote = true;
            break;
        }
    }
    if (!need_quote) return s;
    string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

static string timestamp_for_filename() {
    time_t tt = time(nullptr);
    struct tm tmv{};
    localtime_r(&tt, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return string(buf);
}

static string vision_profile_xml_dir() {
    string dir = join_log_path("logs");
    struct stat st{};
    if (stat(dir.c_str(), &st) != 0) {
        mkdir(dir.c_str(), 0755);
    }
    return dir;
}

static bool save_text_file(const string& path, const string& text) {
    ofstream ofs(path);
    if (!ofs.good()) return false;
    ofs << text;
    return true;
}

static int profile_daynight_preset(CameraProfile profile) {
    return profile == CameraProfile::NIGHT_IR_TEST ? PRESET_NIGHT_MODE : PRESET_DAY_MODE;
}

static string profile_daynight_cmd(CameraProfile profile) {
    return profile == CameraProfile::NIGHT_IR_TEST ? "NIGHT" : "DAY";
}

static void append_camera_profile_log(CameraProfile old_profile,
                                      CameraProfile new_profile,
                                      const string& reason,
                                      const string& manual_or_auto,
                                      const string& daynight_cmd,
                                      int preset_id,
                                      bool isapi_success,
                                      const string& ret_code,
                                      double send_ms) {
    ostringstream line;
    line << metric_now_ms_cpp() << ","
         << camera_profile_to_string(old_profile) << ","
         << camera_profile_to_string(new_profile) << ","
         << csv_escape_simple(reason) << ","
         << manual_or_auto << ","
         << fixed << setprecision(3)
         << g_camera_brightness_mean.load() << ","
         << g_camera_contrast_std.load() << ","
         << g_camera_saturation_mean.load() << ","
         << g_camera_blur_laplacian_var.load() << ","
         << (g_camera_is_grayscale.load() ? 1 : 0) << ","
         << (g_camera_ptz_moving.load() ? 1 : 0) << ","
         << csv_escape_simple(camera_track_state_snapshot()) << ","
         << g_camera_zoom_ratio.load() << ","
         << daynight_cmd << ","
         << preset_id << ","
         << (isapi_success ? 1 : 0) << ","
         << csv_escape_simple(ret_code.empty() ? "empty" : ret_code) << ","
         << send_ms;

    append_csv_line(
        "camera_log.csv",
        "timestamp_ms,profile_old,profile_new,switch_reason,manual_or_auto,brightness_mean,contrast_std,saturation_mean,blur_laplacian_var,is_grayscale_like,ptz_moving,track_state,zoom_ratio,daynight_cmd,preset_id,isapi_success,ret_code,send_ms",
        line.str()
    );
}

static string trim_copy(string s) {
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    size_t start = 0;
    while (start < s.size() && isspace((unsigned char)s[start])) start++;
    return s.substr(start);
}

static string vision_local_log_dir() {
    const char* env = getenv("ANTI_UAV_VISION_LOCAL_LOG_DIR");
    if (env != nullptr && strlen(env) > 0) return string(env);
    return ".";
}

static string join_log_path(const string& filename) {
    string dir = vision_local_log_dir();
    if (dir.empty() || dir == ".") return filename;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + filename;
    return dir + "/" + filename;
}

static void append_csv_line(const string& filename, const string& header, const string& line) {
    static mutex csv_mtx;
    lock_guard<mutex> lock(csv_mtx);

    string path = join_log_path(filename);
    bool need_header = true;
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && st.st_size > 0) {
        need_header = false;
    }

    ofstream ofs(path, ios::app);
    if (!ofs.good()) return;
    if (need_header) ofs << header << "\n";
    ofs << line << "\n";
}

static string camera_mode_name_from_preset(int preset_id) {
    if (preset_id == PRESET_DAY_MODE) return "DAY";
    if (preset_id == PRESET_NIGHT_MODE) return "NIGHT";
    if (preset_id == PRESET_AUTO_DAYNIGHT) return "AUTO";
    return "UNKNOWN";
}

static void append_camera_log(const string& mode, int preset_id, const string& reason,
                              bool success, const string& ret_code, double send_ms) {
    CameraProfile profile = current_camera_profile();
    append_camera_profile_log(profile, profile, reason, "preset",
                              mode, preset_id, success, ret_code, send_ms);
}

static void append_ptz_log(const string& cmd_source, const string& cmd,
                           int pan, int tilt, int zoom, int speed,
                           bool manual_override, const string& stop_reason,
                           double send_ms, const string& ret_code) {
    ostringstream line;
    line << metric_now_ms_cpp() << "," << cmd_source << "," << cmd << ","
         << pan << "," << tilt << "," << zoom << "," << speed << ","
         << (manual_override ? 1 : 0) << "," << stop_reason << ","
         << fixed << setprecision(3) << send_ms << "," << ret_code;
    append_csv_line(
        "ptz_log.csv",
        "timestamp_ms,cmd_source,cmd,pan,tilt,zoom,speed,manual_override,stop_reason,send_ms,ret_code",
        line.str()
    );
}

bool call_special_preset(int preset_id, const string& reason) {
    string mode = camera_mode_name_from_preset(preset_id);
    string url = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" +
                 to_string(HIK_CHANNEL) + "/presets/" + to_string(preset_id) + "/goto";
    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 0.4 --max-time 0.9 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-X PUT "
        "'" + url + "'";

    double t0 = now_sec();
    string ret_code = trim_copy(shell_capture(cmd));
    double send_ms = (now_sec() - t0) * 1000.0;
    bool ok = ret_code.find("200") != string::npos || ret_code.find("201") != string::npos;

    if (ok) {
        g_camera_mode_preset_cmd.store(preset_id);
        if (preset_id == PRESET_DAY_MODE) {
            g_day_retry_holdoff_until_ms.store(0);
        } else {
            // 用户主动选择夜晚红外或自动日夜后，不让灰度检测立即把模式拉回白天。
            g_day_retry_holdoff_until_ms.store(now_ms() + 5LL * 60LL * 1000LL);
        }
    }
    cout << "[CAMERA_MODE] set " << mode << " by preset " << preset_id
         << ", ret=" << (ret_code.empty() ? "empty" : ret_code)
         << ", reason=" << reason << endl;
    if (!ok) {
        cerr << "[CAMERA_MODE] warning: special preset failed, YOLO main flow continues." << endl;
    }
    append_camera_log(mode, preset_id, reason, ok, ret_code.empty() ? "empty" : ret_code, send_ms);
    return ok;
}

bool goto_camera_preset(int preset_id, const string& reason) {
    if (preset_id <= 0 || preset_id > 300) return false;
    string url = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" +
                 to_string(HIK_CHANNEL) + "/presets/" + to_string(preset_id) + "/goto";
    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 0.5 --max-time 1.2 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-X PUT '" + url + "'";
    string code = trim_copy(shell_capture(cmd));
    bool ok = code.find("200") != string::npos || code.find("201") != string::npos;
    cout << "[PTZ_PRESET] goto preset=" << preset_id
         << " reason=" << reason << " ret=" << (code.empty() ? "empty" : code) << endl;
    return ok;
}

bool set_camera_day_mode(const string& reason = "qt_button") {
    return call_special_preset(PRESET_DAY_MODE, reason);
}

bool set_camera_night_mode(const string& reason = "qt_button") {
    return call_special_preset(PRESET_NIGHT_MODE, reason);
}

bool set_camera_auto_daynight(const string& reason = "qt_button") {
    return call_special_preset(PRESET_AUTO_DAYNIGHT, reason);
}

static bool replace_xml_tag_value(string& xml, const string& tag, const string& value) {
    string begin = "<" + tag;
    string end = "</" + tag + ">";
    size_t p1 = xml.find(begin);
    if (p1 == string::npos) return false;
    size_t v1 = xml.find(">", p1);
    if (v1 == string::npos) return false;
    v1 += 1;
    size_t v2 = xml.find(end, v1);
    if (v2 == string::npos) return false;
    xml.replace(v1, v2 - v1, value);
    return true;
}

static bool replace_xml_any_tag(string& xml, const vector<string>& tags,
                                const string& value, const string& label) {
    bool changed = false;
    for (const string& tag : tags) {
        if (replace_xml_tag_value(xml, tag, value)) {
            changed = true;
        }
    }
    if (!changed) {
        cerr << "[CAMERA_PROFILE] warning: XML field not found for " << label << endl;
    }
    return changed;
}

static bool replace_xml_tag_value_in_range(string& xml,
                                           size_t search_begin,
                                           size_t search_end,
                                           const string& tag,
                                           const string& value) {
    string begin = "<" + tag;
    string end = "</" + tag + ">";
    size_t p1 = xml.find(begin, search_begin);
    if (p1 == string::npos || p1 >= search_end) return false;
    size_t v1 = xml.find(">", p1);
    if (v1 == string::npos || v1 >= search_end) return false;
    v1 += 1;
    size_t v2 = xml.find(end, v1);
    if (v2 == string::npos || v2 > search_end) return false;
    xml.replace(v1, v2 - v1, value);
    return true;
}

static bool replace_xml_section_child(string& xml,
                                      const string& section_tag,
                                      const vector<string>& child_tags,
                                      const string& value,
                                      const string& label,
                                      bool warn_if_missing = true) {
    string section_begin = "<" + section_tag;
    string section_end = "</" + section_tag + ">";
    size_t s1 = xml.find(section_begin);
    if (s1 == string::npos) {
        if (warn_if_missing) {
            cerr << "[CAMERA_PROFILE] warning: XML section not found for " << label << endl;
        }
        return false;
    }
    size_t s2 = xml.find(section_end, s1);
    if (s2 == string::npos) {
        if (warn_if_missing) {
            cerr << "[CAMERA_PROFILE] warning: XML section end not found for " << label << endl;
        }
        return false;
    }
    s2 += section_end.size();

    for (const string& tag : child_tags) {
        if (replace_xml_tag_value_in_range(xml, s1, s2, tag, value)) {
            return true;
        }
    }

    if (warn_if_missing) {
        cerr << "[CAMERA_PROFILE] warning: XML field not found for " << label << endl;
    }
    return false;
}

static void apply_xml_value_if_present(string& xml, CameraProfile profile) {
    const bool night = profile == CameraProfile::NIGHT_IR_TEST;
    const string daynight = night ? "night" : "day";
    const string ircut = night ? "night" : "day";

    string shutter = "1/250";
    string shutter_limit = "1/250";
    string gain_limit = "50";
    string noise_reduce = "25";
    string sharpness = "50";
    string brightness = "50";
    string contrast = "50";
    string saturation = "50";
    string dehaze_mode = "close";
    string dehaze_level = "0";

    if (profile == CameraProfile::DAY_CLOUDY_UAV) {
        shutter = "1/250";
        shutter_limit = "1/120";
        gain_limit = "45";
        noise_reduce = "20";
        sharpness = "65";
        brightness = "54";
        contrast = "60";
        saturation = "82";
        dehaze_mode = "close";
        dehaze_level = "0";
    } else if (profile == CameraProfile::FAST_TRACKING) {
        shutter = "1/500";
        shutter_limit = "1/250";
        gain_limit = "45";
        noise_reduce = "15";
        sharpness = "62";
        contrast = "54";
        saturation = "72";
    } else if (profile == CameraProfile::LOW_LIGHT_KEEP_RGB) {
        shutter = "1/100";
        shutter_limit = "1/100";
        gain_limit = "82";
        noise_reduce = "18";
        sharpness = "65";
        brightness = "56";
        contrast = "62";
        saturation = "95";
    } else if (profile == CameraProfile::NIGHT_IR_TEST) {
        shutter = "1/50";
        shutter_limit = "1/25";
        gain_limit = "80";
        noise_reduce = "50";
        sharpness = "50";
        contrast = "50";
        saturation = "0";
        dehaze_mode = "auto";
        dehaze_level = "50";
    }

    replace_xml_section_child(xml, "IrcutFilter", {"IrcutFilterType", "ircutFilterType", "IRCutFilterType"},
                              ircut, "IrcutFilterType");
    replace_xml_section_child(xml, "DSS", {"enabled"}, "false", "DSS slow shutter");
    // 这台 DF 球机的 Image/channels/1 接口会拒绝 ExposureType=shutterPriority
    // 和 IrLight mode=close，日夜/红外仍由特殊预置点 39/40/46 控制。
    replace_xml_section_child(xml, "WDR", {"mode"}, "close", "WDR mode");
    replace_xml_section_child(xml, "WDR", {"WDRLevel"}, "0", "WDR level", false);
    replace_xml_section_child(xml, "BLC", {"enabled"}, "false", "BLC");
    replace_xml_section_child(xml, "HLC", {"enabled"}, "false", "HLC");
    replace_xml_section_child(xml, "HLC", {"HLCLevel"}, "0", "HLC level", false);
    replace_xml_section_child(xml, "NoiseReduce", {"FrameNoiseReduceLevel"}, noise_reduce,
                              "frame noise reduction");
    replace_xml_section_child(xml, "NoiseReduce", {"InterFrameNoiseReduceLevel"}, noise_reduce,
                              "inter-frame noise reduction");
    replace_xml_section_child(xml, "WhiteBalance", {"WhiteBalanceStyle"}, "auto", "white balance");
    replace_xml_section_child(xml, "FocusConfiguration", {"focusStyle"}, "AUTO", "focus style");
    replace_xml_section_child(xml, "Sharpness", {"SharpnessLevel"}, sharpness, "sharpness");
    replace_xml_section_child(xml, "Gain", {"GainLimit", "gainLimit", "maxGain", "MaxGain"},
                              gain_limit, "gain limit");
    replace_xml_section_child(xml, "Shutter", {"ShutterLevel", "shutterLevel", "shutterSpeed", "ShutterSpeed"},
                              shutter, "shutter");
    replace_xml_section_child(xml, "Shutter", {"maxShutterLevelLimit", "maxShutter", "MaxShutter", "shutterLimit", "ShutterLimit"},
                              shutter_limit, "shutter limit");
    replace_xml_section_child(xml, "Color", {"brightnessLevel", "BrightnessLevel", "brightness", "Brightness"},
                              brightness, "brightness");
    replace_xml_section_child(xml, "Color", {"contrastLevel", "ContrastLevel", "contrast", "Contrast"},
                              contrast, "contrast");
    replace_xml_section_child(xml, "Color", {"saturationLevel", "SaturationLevel", "saturation", "Saturation"},
                              saturation, "saturation");
    replace_xml_section_child(xml, "Dehaze", {"DehazeMode", "defog", "Defog"}, dehaze_mode,
                              "dehaze mode");
    replace_xml_section_child(xml, "Dehaze", {"DehazeLevel", "defogLevel", "DefogLevel"}, dehaze_level,
                              "dehaze level", false);
}

bool get_camera_image_config_xml(string& xml) {
    string url = "http://" + HIK_IP + "/ISAPI/Image/channels/" + to_string(HIK_CHANNEL);
    string cmd =
        "curl -s --digest "
        "--connect-timeout 0.8 --max-time 1.8 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "'" + url + "'";
    xml = shell_capture(cmd);
    bool ok = !xml.empty() && xml.find("<") != string::npos;
    if (!ok) {
        cerr << "[CAMERA_PROFILE] warning: GET image config failed or empty." << endl;
    }
    return ok;
}

bool put_camera_image_config_xml(const string& xml) {
    string tmp = "/tmp/hik_image_profile_" + to_string(getpid()) + "_" + to_string(now_ms()) + ".xml";
    {
        ofstream ofs(tmp);
        ofs << xml;
    }

    string url = "http://" + HIK_IP + "/ISAPI/Image/channels/" + to_string(HIK_CHANNEL);
    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 1.2 --max-time 5.0 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-H 'Content-Type: application/xml' "
        "-X PUT --data-binary @" + tmp + " "
        "'" + url + "'";
    string code = trim_copy(shell_capture(cmd));
    unlink(tmp.c_str());
    bool ok = code.find("200") != string::npos || code.find("201") != string::npos;
    if (!ok) {
        cerr << "[CAMERA_PROFILE] warning: PUT image config failed, ret="
             << (code.empty() ? "empty" : code) << endl;
    }
    return ok;
}

bool enable_camera_autofocus() {
    string xml;
    if (!get_camera_image_config_xml(xml)) return false;
    if (!replace_xml_section_child(
            xml, "FocusConfiguration", {"focusStyle"}, "AUTO", "focus style")) {
        return false;
    }
    bool ok = put_camera_image_config_xml(xml);
    cout << "[CAMERA_FOCUS] autofocus " << (ok ? "enabled" : "failed") << endl;
    return ok;
}

bool apply_camera_profile(CameraProfile profile, const string& reason) {
    CameraProfile old_profile = current_camera_profile();
    const string profile_name = camera_profile_to_string(profile);
    const string manual_or_auto = reason.find("auto") != string::npos ? "auto" : "manual";
    const int preset_id = profile_daynight_preset(profile);
    const string daynight_cmd = profile_daynight_cmd(profile);

    cout << "[CAMERA_PROFILE] apply " << profile_name
         << ", reason=" << reason << endl;

    bool preset_ok = profile == CameraProfile::NIGHT_IR_TEST
                         ? set_camera_night_mode(reason)
                         : set_camera_day_mode(reason);

    string before_xml;
    string ret_code = preset_ok ? "preset_ok" : "preset_fail";
    bool isapi_ok = false;
    double t0 = now_sec();
    if (get_camera_image_config_xml(before_xml)) {
        string stamp = timestamp_for_filename();
        string dir = vision_profile_xml_dir();
        string before_path = dir + "/camera_image_config_before_" + stamp + ".xml";
        string after_path = dir + "/camera_image_config_after_" + stamp + ".xml";
        save_text_file(before_path, before_xml);

        string after_xml = before_xml;
        apply_xml_value_if_present(after_xml, profile);
        save_text_file(after_path, after_xml);

        isapi_ok = put_camera_image_config_xml(after_xml);
        ret_code = isapi_ok ? "isapi_ok" : "isapi_fail";
        if (!isapi_ok) {
            set_camera_quality_text("", "ISAPI_CAMERA_CONFIG_FAILED", reason);
        }
    } else {
        set_camera_quality_text("", "ISAPI_CAMERA_CONFIG_FAILED", reason);
        ret_code = "get_isapi_fail";
    }
    double send_ms = (now_sec() - t0) * 1000.0;

    g_camera_profile.store((int)profile);
    g_last_profile_switch_ms.store(now_ms());
    g_profile_candidate.store((int)profile);
    g_profile_candidate_count.store(0);
    set_camera_quality_text("", "", reason);

    cout << "[CAMERA_PROFILE] applied " << profile_name
         << ", preset_ok=" << (preset_ok ? 1 : 0)
         << ", isapi_ok=" << (isapi_ok ? 1 : 0)
         << ", reason=" << reason << endl;

    append_camera_profile_log(old_profile, profile, reason, manual_or_auto,
                              daynight_cmd, preset_id, isapi_ok,
                              ret_code, send_ms);
    return preset_ok || isapi_ok;
}

int xml_tag_int(const string& xml, const string& tag, int default_value) {
    string begin = "<" + tag + ">";
    string end = "</" + tag + ">";

    size_t p1 = xml.find(begin);
    if (p1 == string::npos) return default_value;
    p1 += begin.size();

    size_t p2 = xml.find(end, p1);
    if (p2 == string::npos) return default_value;

    try {
        return stoi(xml.substr(p1, p2 - p1));
    } catch (...) {
        return default_value;
    }
}

PTZState get_ptz_status() {
    PTZState st;

    string cmd =
        "curl -s --digest "
        "--connect-timeout 0.4 --max-time 0.8 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "'" + PTZ_STATUS_URL + "'";

    string xml = shell_capture(cmd);
    if (xml.empty()) return st;

    st.azimuth = xml_tag_int(xml, "azimuth", 0);
    st.elevation = xml_tag_int(xml, "elevation", 0);
    st.absolute_zoom = xml_tag_int(xml, "absoluteZoom", 10);
    st.ok = true;
    return st;
}

static void reset_live_fusion_calibration(const string& reason,
                                          bool reset_runtime_learning) {
    {
        lock_guard<mutex> lock(fusion_mtx);
        fusion_calib = FusionCalibration();
        fusion_telemetry.audio_guided = false;
        fusion_telemetry.audio_pan_err_deg = 99999.0;
        fusion_telemetry.audio_visual_angle_error_deg = 99999.0;
        fusion_telemetry.audio_world_azimuth_deg = 99999.0;
        fusion_telemetry.mic_to_camera_offset_deg = 0.0;
        fusion_telemetry.offset_confidence = 0.0;
        fusion_telemetry.offset_confidence_label = "low";
        fusion_telemetry.calibration_samples = 0;
    }
    if (reset_runtime_learning) {
        lock_guard<mutex> lock(runtime_learning_mtx);
        runtime_learning = RuntimeLearningState();
    }
    g_audio_calibration_reset_requested.store(true);
    save_fusion_calibration();
    cout << "[FUSION] live calibration reset: " << reason
         << " runtime_learning=" << (reset_runtime_learning ? "ON" : "OFF") << endl;
}

static bool mark_current_audio_calibration_target(string& message) {
    SharedAudioCue audio;
    SharedDetection det;
    {
        lock_guard<mutex> lock(latest_audio_mtx);
        audio = latest_audio;
    }
    {
        lock_guard<mutex> lock(latest_det_mtx);
        det = latest_det;
    }

    const double now = now_sec();
    if (!audio.detected || !valid_deg360(audio.smooth_doa_deg) ||
        (now - audio.ts) > AUDIO_CUE_MAX_AGE_SEC) {
        message = "audio_not_ready";
        return false;
    }
    if (audio.stability < AUDIO_GUIDE_MIN_STABILITY ||
        audio.score < AUDIO_GUIDE_MIN_SCORE_PERCENT ||
        audio.snr_db < AUDIO_GUIDE_MIN_SNR_DB) {
        ostringstream ss;
        ss << "audio_not_stable stability=" << fixed << setprecision(2)
           << audio.stability << " score=" << audio.score
           << " snr=" << audio.snr_db;
        message = ss.str();
        return false;
    }
    if (!det.valid || !det.confirmed || (now - det.ts) > PTZ_TARGET_MAX_AGE_SEC) {
        message = "visual_target_not_ready";
        return false;
    }

    PTZState st = get_ptz_status();
    if (!st.ok) {
        message = "ptz_status_failed";
        return false;
    }

    const double visual_world = visual_target_azimuth_deg(det, st);
    const double offset = wrap_deg360(visual_world - audio.smooth_doa_deg);
    const double audio_world = wrap_deg360(audio.smooth_doa_deg + offset);
    const double err = fabs(circular_error_deg(visual_world, audio_world));

    {
        lock_guard<mutex> lock(fusion_mtx);
        fusion_calib.offset_deg = offset;
        fusion_calib.confidence = max(fusion_calib.confidence,
                                      (double)AUDIO_OFFSET_HIGH_CONF);
        fusion_calib.total_samples =
            max(fusion_calib.total_samples, AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES);
        fusion_calib.stable_samples =
            max(fusion_calib.stable_samples, AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES);
        fusion_calib.last_error_deg = err;
        fusion_calib.last_visual_azimuth_deg = visual_world;
        fusion_calib.last_audio_world_deg = audio_world;

        fusion_telemetry.visual_azimuth_deg = visual_world;
        fusion_telemetry.audio_world_azimuth_deg = audio_world;
        fusion_telemetry.audio_visual_angle_error_deg = err;
        fusion_telemetry.mic_to_camera_offset_deg = offset;
        fusion_telemetry.offset_confidence = fusion_calib.confidence;
        fusion_telemetry.offset_confidence_label =
            fusion_confidence_label(fusion_calib.confidence);
        fusion_telemetry.calibration_samples = fusion_calib.stable_samples;
    }
    save_fusion_calibration();

    ostringstream ss;
    ss << "calibration_marked offset=" << fixed << setprecision(2) << offset
       << " visual=" << visual_world
       << " audio_doa=" << audio.smooth_doa_deg
       << " err=" << err;
    message = ss.str();
    cout << "[FUSION] " << message << endl;
    return true;
}

bool send_ptz_absolute(int azimuth, int elevation, int absolute_zoom) {
    azimuth = wrap_azimuth(azimuth);
    elevation = max(-900, min(900, elevation));
    absolute_zoom = max(10, min(370, absolute_zoom));

    string xml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<PTZData version=\"2.0\" xmlns=\"http://www.hikvision.com/ver20/XMLSchema\">\n"
        "  <AbsoluteHigh>\n"
        "    <azimuth>" + to_string(azimuth) + "</azimuth>\n"
        "    <elevation>" + to_string(elevation) + "</elevation>\n"
        "    <absoluteZoom>" + to_string(absolute_zoom) + "</absoluteZoom>\n"
        "  </AbsoluteHigh>\n"
        "</PTZData>\n";

    string tmp = "/tmp/hik_ptz_abs_" + to_string(getpid()) + ".xml";
    {
        ofstream ofs(tmp);
        ofs << xml;
    }

    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 0.4 --max-time 0.8 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-H 'Content-Type: application/xml' "
        "-X PUT --data-binary @" + tmp + " "
        "'" + PTZ_ABSOLUTE_URL + "'";

    string code = shell_capture(cmd);
    unlink(tmp.c_str());

    return code.find("200") != string::npos || code.find("201") != string::npos;
}


bool send_ptz_continuous(int pan_user, int tilt_user, int zoom_user = 0) {
    // pan_user / tilt_user 是“用户方向”：目标在右边 pan_user 为正，目标在下方 tilt_user 为正。
    // 通过 CONT_*_DIR 转成海康设备方向。
    int dev_pan = CONT_PAN_DIR * pan_user;
    int dev_tilt = CONT_TILT_DIR * tilt_user;
    int dev_zoom = CONT_ZOOM_DIR * zoom_user;

    string xml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<PTZData version=\"2.0\" xmlns=\"http://www.hikvision.com/ver20/XMLSchema\">\n"
        "  <pan>" + to_string(dev_pan) + "</pan>\n"
        "  <tilt>" + to_string(dev_tilt) + "</tilt>\n"
        "  <zoom>" + to_string(dev_zoom) + "</zoom>\n"
        "</PTZData>\n";

    string tmp = "/tmp/hik_ptz_cont_" + to_string(getpid()) + ".xml";
    {
        ofstream ofs(tmp);
        ofs << xml;
    }

    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 0.25 --max-time 0.55 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-H 'Content-Type: application/xml' "
        "-X PUT --data-binary @" + tmp + " "
        "'" + PTZ_CONTINUOUS_URL + "'";

    string code = shell_capture(cmd);
    unlink(tmp.c_str());

    return code.find("200") != string::npos || code.find("201") != string::npos;
}

static bool put_hik_xml_request(const string& url, const string& xml, string& ret_code) {
    string tmp = "/tmp/hik_safe_" + to_string(getpid()) + "_" + to_string(now_ms()) + ".xml";
    {
        ofstream ofs(tmp);
        ofs << xml;
    }

    string cmd =
        "curl -s -o /dev/null -w '%{http_code}' --digest "
        "--connect-timeout 0.5 --max-time 1.2 "
        "-u '" + HIK_USER + ":" + HIK_PASS + "' "
        "-H 'Content-Type: application/xml' "
        "-X PUT --data-binary @" + tmp + " "
        "'" + url + "'";

    ret_code = trim_copy(shell_capture(cmd));
    unlink(tmp.c_str());
    return ret_code.find("200") != string::npos || ret_code.find("201") != string::npos;
}

static bool configure_hik_park_action(int preset_id, int park_time_sec, string& ret_code) {
    park_time_sec = max(5, min(720, park_time_sec));
    string url = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/" +
                 to_string(HIK_CHANNEL) + "/parkAction";
    string xml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<ParkAction version=\"2.0\" xmlns=\"http://www.hikvision.com/ver20/XMLSchema\">\n"
        "  <enabled>true</enabled>\n"
        "  <Parktime>" + to_string(park_time_sec) + "</Parktime>\n"
        "  <Action>\n"
        "    <ActionType>preset</ActionType>\n"
        "    <ActionNum>" + to_string(preset_id) + "</ActionNum>\n"
        "  </Action>\n"
        "</ParkAction>\n";
    return put_hik_xml_request(url, xml, ret_code);
}

static void append_camera_safety_log(const string& reason,
                                     bool stop_ok,
                                     bool park_cfg_ok,
                                     bool abs_ok,
                                     bool preset_ok,
                                     const string& ret_code,
                                     double send_ms) {
    ostringstream line;
    line << metric_now_ms_cpp() << ","
         << csv_escape_simple(reason) << ","
         << (stop_ok ? 1 : 0) << ","
         << (park_cfg_ok ? 1 : 0) << ","
         << (abs_ok ? 1 : 0) << ","
         << (preset_ok ? 1 : 0) << ","
         << csv_escape_simple(ret_code.empty() ? "empty" : ret_code) << ","
         << fixed << setprecision(3) << send_ms;
    append_csv_line(
        "camera_safety_log.csv",
        "timestamp_ms,reason,stop_ok,park_cfg_ok,absolute_ok,preset34_ok,ret_code,send_ms",
        line.str()
    );
}

static bool safe_shutdown_hikvision_camera(const string& reason) {
    if (g_camera_safe_shutdown_done.exchange(true)) {
        return true;
    }

    double t0 = now_sec();
    cout << "[CAMERA_SAFE] begin safe shutdown, reason=" << reason << endl;

    bool stop_ok = false;
    for (int i = 0; i < 3; ++i) {
        stop_ok = send_ptz_continuous(0, 0, 0) || stop_ok;
        this_thread::sleep_for(chrono::milliseconds(80));
    }

    const bool allow_reposition =
        env_flag_or_default_early("ANTI_UAV_CAMERA_REPOSITION_ON_EXIT", false);
    if (!allow_reposition) {
        const double send_ms = (now_sec() - t0) * 1000.0;
        append_camera_safety_log(
            reason,
            stop_ok,
            false,
            false,
            false,
            "reposition_skipped",
            send_ms
        );
        cout << "[CAMERA_SAFE] reposition skipped; zero-speed stop only. "
             << "Set ANTI_UAV_CAMERA_REPOSITION_ON_EXIT=1 to enable presets/home."
             << endl;
        return stop_ok;
    }

    // 特殊预置点 96 是“停止扫描”，可防止巡航/扫描动作在停机前继续运行。
    bool stop_scan_ok = call_special_preset(96, "safe_shutdown_stop_scan");

    const char* park_env = getenv("ANTI_UAV_ENABLE_PARK_ACTION");
    bool enable_park_action = park_env != nullptr && string(park_env) == "1";
    string park_ret = enable_park_action ? "" : "park_skipped";
    bool park_cfg_ok = false;
    if (enable_park_action) {
        park_cfg_ok = configure_hik_park_action(34, 5, park_ret);
        if (park_cfg_ok) {
            cout << "[CAMERA_SAFE] parkAction configured to preset 34, ret=" << park_ret << endl;
        } else {
            cerr << "[CAMERA_SAFE] warning: parkAction configure failed, ret="
                 << (park_ret.empty() ? "empty" : park_ret) << endl;
        }
    } else {
        cout << "[CAMERA_SAFE] parkAction config skipped. Set ANTI_UAV_ENABLE_PARK_ACTION=1 to test it." << endl;
    }

    PTZState st = get_ptz_status();
    int safe_azimuth = st.ok ? st.azimuth : 0;
    bool abs_ok = send_ptz_absolute(safe_azimuth, 0, 10);
    this_thread::sleep_for(chrono::milliseconds(600));

    // 特殊预置点 34 是“回到零点”。若当前机械状态允许，它会把球机收回安全零位。
    bool preset_ok = call_special_preset(34, "safe_shutdown_home_zero");
    this_thread::sleep_for(chrono::milliseconds(1200));

    bool final_stop_ok = send_ptz_continuous(0, 0, 0);
    string ret_summary = string("stop_scan=") + (stop_scan_ok ? "ok" : "fail") +
                         ";park=" + (park_ret.empty() ? "empty" : park_ret) +
                         ";final_stop=" + (final_stop_ok ? "ok" : "fail");
    double send_ms = (now_sec() - t0) * 1000.0;
    append_camera_safety_log(reason, stop_ok, park_cfg_ok, abs_ok, preset_ok, ret_summary, send_ms);

    bool ok = stop_ok || park_cfg_ok || abs_ok || preset_ok || final_stop_ok;
    cout << "[CAMERA_SAFE] done ok=" << (ok ? 1 : 0)
         << " stop=" << (stop_ok ? 1 : 0)
         << " park=" << (park_cfg_ok ? 1 : 0)
         << " abs=" << (abs_ok ? 1 : 0)
         << " preset34=" << (preset_ok ? 1 : 0)
         << " final_stop=" << (final_stop_ok ? 1 : 0)
         << " send_ms=" << fixed << setprecision(1) << send_ms << endl;
    return ok;
}

static CameraProfile choose_auto_camera_profile(double brightness,
                                                double saturation,
                                                double blur_var,
                                                bool is_gray,
                                                bool ptz_moving,
                                                const string& track_state) {
    if (ptz_moving || track_state == "TRACK" || track_state == "Tracked") {
        return CameraProfile::FAST_TRACKING;
    }
    if (is_gray && !ALLOW_AUTO_NIGHT_IR) {
        return CameraProfile::LOW_LIGHT_KEEP_RGB;
    }
    if (brightness < 42.0 && saturation < 18.0) {
        return CameraProfile::LOW_LIGHT_KEEP_RGB;
    }
    if (brightness < 95.0 || saturation < 34.0 || blur_var < 85.0) {
        return CameraProfile::DAY_CLOUDY_UAV;
    }
    return CameraProfile::DAY_NORMAL;
}

static void maybe_auto_camera_profile(double brightness,
                                      double saturation,
                                      double blur_var,
                                      bool is_gray,
                                      bool ptz_moving,
                                      const string& track_state) {
    if (!g_auto_camera_profile_enabled.load()) return;

    long long ms = now_ms();
    if (ms - g_last_profile_eval_ms.load() < CAMERA_PROFILE_MIN_EVAL_MS) return;
    g_last_profile_eval_ms.store(ms);

    if (ms < g_manual_profile_override_until_ms.load()) return;
    if (ms - g_last_profile_switch_ms.load() < CAMERA_PROFILE_MIN_HOLD_MS) return;

    CameraProfile desired = choose_auto_camera_profile(brightness, saturation, blur_var,
                                                       is_gray, ptz_moving, track_state);
    if (desired == CameraProfile::NIGHT_IR_TEST && !ALLOW_AUTO_NIGHT_IR) {
        desired = CameraProfile::LOW_LIGHT_KEEP_RGB;
    }
    if (desired == current_camera_profile()) {
        g_profile_candidate.store((int)desired);
        g_profile_candidate_count.store(0);
        return;
    }

    int prev = g_profile_candidate.load();
    if (prev != (int)desired) {
        g_profile_candidate.store((int)desired);
        g_profile_candidate_count.store(1);
        return;
    }

    int count = g_profile_candidate_count.fetch_add(1) + 1;
    if (count < CAMERA_PROFILE_CONFIRM_STATS) return;

    g_profile_candidate_count.store(0);
    thread([desired]() {
        apply_camera_profile(desired, "auto_quality_switch");
    }).detach();
}

static void update_camera_image_mode_stats(const Mat& bgr) {
    if (bgr.empty() || bgr.channels() != 3) return;

    static atomic<long long> last_quality_ms(0);
    long long nowm = now_ms();
    if (nowm - last_quality_ms.load() < 1000) return;
    last_quality_ms.store(nowm);

    Mat small;
    resize(bgr, small, Size(160, 120), 0, 0, INTER_AREA);

    vector<Mat> ch;
    split(small, ch);
    Mat diff_bg, diff_br, diff_gr;
    absdiff(ch[0], ch[1], diff_bg);
    absdiff(ch[0], ch[2], diff_br);
    absdiff(ch[1], ch[2], diff_gr);
    double color_diff = mean(diff_bg)[0] + mean(diff_br)[0] + mean(diff_gr)[0];

    Mat gray;
    cvtColor(small, gray, COLOR_BGR2GRAY);
    Scalar gray_mean, gray_std;
    meanStdDev(gray, gray_mean, gray_std);

    Mat hsv;
    cvtColor(small, hsv, COLOR_BGR2HSV);
    vector<Mat> hsv_ch;
    split(hsv, hsv_ch);
    double saturation_mean = mean(hsv_ch[1])[0];

    Mat lap;
    Laplacian(gray, lap, CV_64F);
    Scalar lap_mean, lap_std;
    meanStdDev(lap, lap_mean, lap_std);
    double blur_var = lap_std[0] * lap_std[0];

    bool is_gray = color_diff < 9.0 || saturation_mean < 12.0;
    bool ptz_moving = abs(manual_pan_user.load()) > 0 ||
                      abs(manual_tilt_user.load()) > 0 ||
                      abs(manual_zoom_user.load()) > 0;
    string track_state = camera_track_state_snapshot();
    float zoom_ratio = zoom_to_ratio(g_latest_ptz_absolute_zoom.load());

    g_camera_is_grayscale.store(is_gray);
    g_camera_brightness_mean.store(gray_mean[0]);
    g_camera_contrast_std.store(gray_std[0]);
    g_camera_saturation_mean.store(saturation_mean);
    g_camera_blur_laplacian_var.store(blur_var);
    g_camera_zoom_ratio.store(zoom_ratio);
    g_camera_ptz_moving.store(ptz_moving);

    string warning = "OK";
    if (is_gray && current_camera_profile() != CameraProfile::NIGHT_IR_TEST) {
        warning = "GRAYSCALE_LIKE_WARNING";
        cerr << "[CAMERA_PROFILE] GRAYSCALE_LIKE_WARNING brightness="
             << gray_mean[0] << " sat=" << saturation_mean << endl;
    } else if (blur_var < 70.0) {
        warning = "IMAGE_BLUR_WARNING";
        cerr << "[CAMERA_PROFILE] IMAGE_BLUR_WARNING lap_var=" << blur_var
             << ", prefer faster shutter over stronger DNR." << endl;
    } else if (saturation_mean < 22.0) {
        warning = "LOW_SATURATION_WARNING";
    }
    set_camera_quality_text(track_state, warning);

    maybe_auto_camera_profile(gray_mean[0], saturation_mean, blur_var,
                              is_gray, ptz_moving, track_state);

    int gray_frames = is_gray ? (g_camera_grayscale_frames.fetch_add(1) + 1) : 0;
    if (!is_gray) {
        g_camera_grayscale_frames.store(0);
        return;
    }

    long long ms = now_ms();
    if (FORCE_DAY_MODE_ON_START &&
        DISABLE_AUTO_IR_FOR_TEST &&
        g_camera_mode_preset_cmd.load() == PRESET_DAY_MODE &&
        gray_frames >= 30 &&
        ms >= g_day_retry_holdoff_until_ms.load() &&
        ms - g_last_force_day_retry_ms.load() > 60000) {
        g_last_force_day_retry_ms.store(ms);
        thread([]() {
            apply_camera_profile(CameraProfile::LOW_LIGHT_KEEP_RGB, "grayscale_retry");
        }).detach();
    }
}

void update_shared_detection(bool valid, bool confirmed, int frame_id, int fw, int fh, const DetBox* d) {
    lock_guard<mutex> lock(latest_det_mtx);

    latest_det.valid = valid;
    latest_det.confirmed = confirmed;
    latest_det.frame_id = frame_id;
    latest_det.fw = fw;
    latest_det.fh = fh;
    latest_det.ts = now_sec();

    if (valid && d != nullptr) {
        latest_det.cx = d->cx;
        latest_det.cy = d->cy;
        latest_det.w = d->w;
        latest_det.h = d->h;
        latest_det.conf = d->conf;
    } else {
        latest_det.cx = latest_det.cy = latest_det.w = latest_det.h = latest_det.conf = 0.0f;
    }
}

void audio_fusion_receiver_thread() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        cerr << "[FUSION] audio UDP socket create failed" << endl;
        return;
    }

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(g_audio_fusion_port);

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        cerr << "[FUSION] bind failed on audio fusion port " << g_audio_fusion_port << endl;
        close(sock);
        return;
    }

    cout << "[FUSION] listening audio cue on UDP 0.0.0.0:" << g_audio_fusion_port << endl;

    char buffer[2048];
    while (is_running.load()) {
        ssize_t n = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
        if (n <= 0) continue;
        buffer[n] = '\0';
        string json(buffer);

        string command = json_get_string(json, "command", "");
        if (command == "set_detection_enabled" || json_has_key(json, "detection_enabled")) {
            bool enabled = json_get_bool(json, "detection_enabled", g_detection_enabled.load());
            g_detection_enabled.store(enabled);
            update_shared_detection(false, false, -1, 0, 0, nullptr);
            cout << "[FUSION] detection_enabled=" << (enabled ? "ON" : "OFF")
                 << " from UDP control" << endl;
            continue;
        }

        SharedAudioCue cue;
        cue.detected = json_get_bool(json, "audio_detected", json_get_bool(json, "detected", false));
        cue.raw_doa_deg = json_get_double(json, "raw_doa_deg", json_get_double(json, "doa_deg", -1.0));
        cue.smooth_doa_deg = json_get_double(json, "doa_deg", json_get_double(json, "angle", -1.0));
        cue.score = json_get_double(json, "yamnet_score", json_get_double(json, "confidence", 0.0));
        cue.score_ema = json_get_double(json, "score_ema", cue.score);
        cue.stability = json_get_double(json, "doa_stability",
                                        json_get_double(json, "doa_confidence", 0.0));
        cue.rms_dbfs = json_get_double(json, "rms_dbfs", -120.0);
        cue.snr_db = json_get_double(json, "snr_db", -99.0);
        cue.ts_mono_ms = (long long)json_get_double(json, "ts_mono_ms", 0.0);
        cue.ts = now_sec();
        cue.state = json_get_string(json, "audio_state", cue.detected ? "CONFIRMED" : "IDLE");

        if (cue.detected && cue.smooth_doa_deg >= 0.0 && cue.smooth_doa_deg < 360.0) {
            lock_guard<mutex> lock(latest_audio_mtx);
            latest_audio = cue;
        } else {
            lock_guard<mutex> lock(latest_audio_mtx);
            latest_audio.detected = false;
            latest_audio.state = cue.state;
            latest_audio.ts = now_sec();
            latest_audio.score = cue.score;
            latest_audio.score_ema = cue.score_ema;
            latest_audio.stability = cue.stability;
            latest_audio.snr_db = cue.snr_db;
        }
    }

    close(sock);
    cout << "[FUSION] audio receiver stopped" << endl;
}

static void apply_qt_manual_move(int pan_dir, int tilt_dir, int zoom_dir,
                                  int ptz_speed, int zoom_speed) {
    long long ms = now_ms();
    const float zoom_ratio = zoom_to_ratio(g_latest_ptz_absolute_zoom.load());
    if (zoom_ratio >= 10.0f) {
        ptz_speed = min(ptz_speed, 62);
    } else if (zoom_ratio >= 5.0f) {
        ptz_speed = min(ptz_speed, 78);
    }
    int pan = clamp_val(pan_dir, -1, 1) * ptz_speed;
    int tilt = clamp_val(tilt_dir, -1, 1) * ptz_speed;
    int zoom = clamp_val(zoom_dir, -1, 1) * zoom_speed;

    if (pan != 0 || tilt != 0) {
        manual_pan_user.store(pan);
        manual_tilt_user.store(tilt);
        manual_ptz_until_ms.store(ms + QT_MANUAL_TIMEOUT_MS);
    } else {
        manual_pan_user.store(0);
        manual_tilt_user.store(0);
        manual_ptz_until_ms.store(0);
    }

    if (zoom != 0) {
        manual_zoom_user.store(zoom);
        manual_zoom_until_ms.store(ms + QT_MANUAL_TIMEOUT_MS);
    } else {
        manual_zoom_user.store(0);
        manual_zoom_until_ms.store(0);
    }

    manual_auto_resume_block_until_ms.store(0);
    manual_force_send.store(true);
    append_ptz_log("qt_udp", "move", pan_dir, tilt_dir, zoom_dir,
                   (zoom != 0 && pan == 0 && tilt == 0) ? zoom_speed : ptz_speed,
                   true, "none", 0.0, "queued");
}

static void apply_qt_manual_stop(const string& stop_reason) {
    manual_pan_user.store(0);
    manual_tilt_user.store(0);
    manual_ptz_until_ms.store(0);
    manual_zoom_user.store(0);
    manual_zoom_until_ms.store(0);
    manual_auto_resume_block_until_ms.store(now_ms() + (long long)round(MANUAL_AUTO_RESUME_DELAY_SEC * 1000.0f));
    manual_force_send.store(true);

    double t0 = now_sec();
    bool ok = send_ptz_continuous(0, 0, 0);
    double send_ms = (now_sec() - t0) * 1000.0;
    append_ptz_log("qt_udp", "stop", 0, 0, 0, 0, true,
                   stop_reason, send_ms, ok ? "200" : "fail");
}

static string apply_qt_control_params(const string& json) {
    int mode = clamp_val((int)round(json_get_double(json, "detection_mode", g_detection_policy_mode.load())), 0, 4);
    g_detection_policy_mode.store(mode);

    g_search_conf_thresh.store(clampf((float)json_get_double(json, "search_conf", g_search_conf_thresh.load()), 0.20f, 0.90f));
    g_track_conf_thresh.store(clampf((float)json_get_double(json, "track_conf", g_track_conf_thresh.load()), 0.20f, 0.90f));
    g_alarm_conf_thresh.store(clampf((float)json_get_double(json, "alarm_conf", g_alarm_conf_thresh.load()), 0.20f, 0.95f));
    g_search_min_box_ratio.store(clampf((float)json_get_double(json, "search_min_box", g_search_min_box_ratio.load()), 0.01f, 0.30f));
    g_track_min_box_ratio.store(clampf((float)json_get_double(json, "track_min_box", g_track_min_box_ratio.load()), 0.02f, 0.35f));
    g_alarm_min_box_ratio.store(clampf((float)json_get_double(json, "alarm_min_box", g_alarm_min_box_ratio.load()), 0.03f, 0.40f));
    int confirm_frames = clamp_val((int)round(json_get_double(json, "confirm_frames", g_search_confirm_frames.load())), 1, 8);
    g_search_confirm_frames.store(confirm_frames);
    g_track_confirm_frames.store(clamp_val((int)round(json_get_double(json, "track_confirm_frames", g_track_confirm_frames.load())), 1, 8));
    g_alarm_confirm_frames.store(clamp_val((int)round(json_get_double(json, "alarm_confirm_frames", g_alarm_confirm_frames.load())), 1, 8));

    float target_min = clampf((float)json_get_double(json, "target_box_min", g_zoom_target_min_ratio.load()), 0.04f, 0.20f);
    float ideal_low = clampf((float)json_get_double(json, "target_box_ideal_low", g_zoom_target_ideal_low_ratio.load()), target_min, 0.28f);
    float ideal_high = clampf((float)json_get_double(json, "target_box_ideal_high", g_zoom_target_ideal_high_ratio.load()), ideal_low, 0.35f);
    float target_max = clampf((float)json_get_double(json, "target_box_max", g_zoom_target_max_ratio.load()), ideal_high, 0.45f);
    g_zoom_target_min_ratio.store(target_min);
    g_zoom_target_ideal_low_ratio.store(ideal_low);
    g_zoom_target_ideal_high_ratio.store(ideal_high);
    g_zoom_target_max_ratio.store(target_max);
    float min_zoom = clampf((float)json_get_double(json, "min_zoom", g_zoom_min_ratio.load()), 1.0f, 20.0f);
    float max_zoom = clampf((float)json_get_double(json, "max_zoom", g_zoom_max_ratio.load()), min_zoom, 37.0f);
    float lost_search_zoom = clampf((float)json_get_double(json, "lost_search_zoom", g_lost_search_zoom_ratio.load()), min_zoom, max_zoom);
    g_zoom_min_ratio.store(min_zoom);
    g_zoom_max_ratio.store(max_zoom);
    g_lost_search_zoom_ratio.store(lost_search_zoom);

    if (json_has_key(json, "auto_track_enabled")) {
        ptz_control_enabled.store(json_get_bool(json, "auto_track_enabled", ptz_control_enabled.load()));
    }
    if (json_has_key(json, "auto_zoom_enabled")) {
        auto_zoom_enabled.store(json_get_bool(json, "auto_zoom_enabled", auto_zoom_enabled.load()));
    }
    if (json_has_key(json, "auto_focus_enabled")) {
        g_autofocus_enabled.store(json_get_bool(json, "auto_focus_enabled", g_autofocus_enabled.load()));
    }

    g_tracking_mode.store(clamp_val((int)round(json_get_double(json, "tracking_mode", g_tracking_mode.load())), 0, 3));
    g_tracking_pan_gain.store(clampf((float)json_get_double(json, "pan_gain", g_tracking_pan_gain.load()), 0.30f, 2.20f));
    g_tracking_tilt_gain.store(clampf((float)json_get_double(json, "tilt_gain", g_tracking_tilt_gain.load()), 0.30f, 2.20f));
    g_tracking_feedforward_gain.store(clampf((float)json_get_double(json, "feedforward_gain", g_tracking_feedforward_gain.load()), 0.0f, 1.20f));
    g_ptz_max_speed.store(clamp_val((int)round(json_get_double(json, "max_speed", g_ptz_max_speed.load())), 12, 110));
    g_ptz_max_speed_high_zoom.store(clamp_val((int)round(json_get_double(json, "high_zoom_max_speed", g_ptz_max_speed_high_zoom.load())), 8, 75));
    g_center_dead_zone_x_ratio.store(clampf((float)json_get_double(json, "center_dead_zone", g_center_dead_zone_x_ratio.load()), 0.02f, 0.16f));
    g_center_dead_zone_y_ratio.store(g_center_dead_zone_x_ratio.load());
    g_zoom_step_in_speed.store(clamp_val((int)round(json_get_double(json, "zoom_step_in", g_zoom_step_in_speed.load())), 5, 120));
    g_zoom_step_out_speed.store(clamp_val((int)round(json_get_double(json, "zoom_step_out", g_zoom_step_out_speed.load())), 5, 180));
    g_zoom_cmd_period_sec.store(clampf((float)json_get_double(json, "zoom_cooldown_ms", g_zoom_cmd_period_sec.load() * 1000.0f) / 1000.0f, 0.08f, 3.0f));
    g_zoom_search_out_speed.store(clamp_val((int)round(json_get_double(json, "lost_zoom_out_step", g_zoom_search_out_speed.load())), 10, 220));
    g_autofocus_cooldown_sec.store(clampf((float)json_get_double(json, "focus_cooldown_ms", g_autofocus_cooldown_sec.load() * 1000.0f) / 1000.0f, 1.0f, 60.0f));

    g_control_params_version.fetch_add(1);
    save_control_params_snapshot("qt_set_control_params");
    const float ack_conf = clampf(g_search_conf_thresh.load(), 0.20f, 0.90f);
    const float ack_min_box = clampf(g_search_min_box_ratio.load(), 0.01f, 0.30f);
    ostringstream msg;
    msg << "mode=" << mode
        << ";conf=" << fixed << setprecision(2) << ack_conf
        << ";min_box=" << fixed << setprecision(2) << ack_min_box
        << ";target=" << fixed << setprecision(2) << g_zoom_target_min_ratio.load()
        << "-" << g_zoom_target_max_ratio.load()
        << ";ptz_max=" << g_ptz_max_speed.load()
        << ";zoom_step=" << g_zoom_step_in_speed.load() << "/" << g_zoom_step_out_speed.load();
    return msg.str();
}

static bool load_control_params_file(const string& path) {
    if (path.empty()) return false;
    ifstream ifs(path);
    if (!ifs.good()) {
        cout << "[CTRL_PARAMS] no saved control params: " << path << endl;
        return false;
    }
    string text((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    string msg = apply_qt_control_params(text);
    save_control_params_snapshot("startup_loaded_control_params");
    cout << "[CTRL_PARAMS] loaded " << path << " " << msg << endl;
    return true;
}

static string autofocus_reason_name(int reason) {
    switch (reason) {
    case 1: return "USER_REQUEST";
    case 2: return "AFTER_ZOOM_REQUEST";
    case 3: return "LOW_BLUR_REQUEST";
    case 4: return "CONFIDENCE_DROP_REQUEST";
    default: return "READY";
    }
}

static bool generate_autofocus_state_request(int reason, long long now_mono, string* message = nullptr) {
    long long cooldown_until = g_autofocus_cooldown_until_ms.load();
    if (now_mono < cooldown_until) {
        if (message) {
            *message = "focus_cooldown:" + to_string(cooldown_until - now_mono) + "ms";
        }
        return false;
    }

    g_last_autofocus_reason.store(reason);
    g_last_autofocus_request_ms.store(now_mono);
    g_autofocus_cooldown_until_ms.store(
        now_mono + (long long)round(g_autofocus_cooldown_sec.load() * 1000.0f));
    const string reason_name = autofocus_reason_name(reason);
    if (message) {
        *message = "focus_request_generated:" + reason_name;
    }
    cout << "\n[AUTOFOCUS] generated state request reason=" << reason_name
         << "; real camera focus command is not sent in local/no-hardware flow" << endl;
    return true;
}

void qt_control_receiver_thread() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        cerr << "[QT_CTRL] UDP socket create failed" << endl;
        return;
    }

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(g_qt_control_port);

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        cerr << "[QT_CTRL] bind failed on UDP " << g_qt_control_port << endl;
        close(sock);
        return;
    }

    cout << "[QT_CTRL] listening UDP " << g_qt_control_port << endl;

    char buffer[2048];
    while (is_running.load()) {
        sockaddr_in sender_addr{};
        socklen_t sender_len = sizeof(sender_addr);
        ssize_t n = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                             reinterpret_cast<sockaddr*>(&sender_addr), &sender_len);
        if (n <= 0) continue;
        buffer[n] = '\0';
        string json(buffer);
        string type = json_get_string(json, "type", "");
        auto send_ack = [&](const string& command, bool ok, const string& message) {
            ostringstream ack;
            ack << "{\"type\":\"qt_ptz_ack\",\"command\":\""
                << json_escape_string(command) << "\",\"ok\":" << (ok ? 1 : 0)
                << ",\"message\":\"" << json_escape_string(message) << "\""
                << ",\"auto_tracking\":" << (ptz_control_enabled.load() ? 1 : 0)
                << ",\"pan\":" << g_latest_ptz_azimuth_unit.load()
                << ",\"tilt\":" << g_latest_ptz_elevation_unit.load()
                << ",\"zoom\":" << g_latest_ptz_absolute_zoom.load()
                << ",\"ts_ms\":" << metric_now_ms_cpp() << "}";
            const string payload = ack.str();
            sendto(sock, payload.data(), payload.size(), 0,
                   reinterpret_cast<const sockaddr*>(&sender_addr), sender_len);
        };
        const string source = json_get_string(json, "source", "");
        const string requested_cmd = json_get_string(json, "cmd", type);
        const bool safety_command =
            requested_cmd == "stop" || requested_cmd == "emergency_stop";
        const long long control_now_ms = now_ms();
        if (source == "elf2") {
            g_local_ui_lease_until_ms.store(control_now_ms + 2000);
        } else if (source == "windows" && !safety_command &&
                   control_now_ms < g_local_ui_lease_until_ms.load()) {
            send_ack(requested_cmd, false, "local_ui_has_priority");
            continue;
        }

        if (type == "qt_camera_profile") {
            string profile_text = json_get_string(json, "profile", "");
            string profile_upper = profile_text;
            transform(profile_upper.begin(), profile_upper.end(), profile_upper.begin(),
                      [](unsigned char c) { return (char)toupper(c); });
            if (profile_upper == "AUTO_PROFILE") {
                g_auto_camera_profile_enabled.store(true);
                g_manual_profile_override_until_ms.store(0);
                set_camera_quality_text("", "", "qt_auto_profile_enabled");
                cout << "[QT_CTRL] camera auto profile enabled" << endl;
            } else {
                CameraProfile profile = DEFAULT_CAMERA_PROFILE;
                if (camera_profile_from_string(profile_text, profile)) {
                    g_manual_profile_override_until_ms.store(now_ms() + CAMERA_PROFILE_MANUAL_HOLD_MS);
                    thread([profile]() {
                        apply_camera_profile(profile, "qt_profile_button");
                    }).detach();
                    cout << "[QT_CTRL] qt_camera_profile " << camera_profile_to_string(profile) << endl;
                } else {
                    cerr << "[QT_CTRL] unknown camera profile: " << profile_text << endl;
                }
            }
            continue;
        }

        if (type == "qt_camera_mode") {
            string mode = json_get_string(json, "mode", "");
            if (mode == "day") {
                set_camera_day_mode("qt_button");
            } else if (mode == "night") {
                set_camera_night_mode("qt_button");
            } else if (mode == "auto") {
                set_camera_auto_daynight("qt_button");
            } else {
                cerr << "[QT_CTRL] unknown camera mode: " << mode << endl;
            }
            continue;
        }

        if (type != "qt_ptz_cmd") {
            cerr << "[QT_CTRL] unknown json type: " << type << endl;
            send_ack(type, false, "unknown_type");
            continue;
        }

        string cmd = json_get_string(json, "cmd", "");
        const long long command_seq =
            (long long)round(json_get_double(json, "seq", 0.0));
        const bool has_seq = json_has_key(json, "seq");
        const bool is_emergency = cmd == "emergency_stop";
        bool stale_command = false;
        string stale_reason;
        if (!is_emergency && has_seq && command_seq > 0) {
            atomic<long long>& last_seq = last_seq_for_control_source(source);
            long long previous_seq = last_seq.load();
            if (command_seq <= previous_seq) {
                stale_command = true;
                stale_reason = "stale_seq";
            } else {
                last_seq.store(command_seq);
            }
        }
        if (stale_command) {
            g_stale_command_dropped_count.fetch_add(1);
            g_last_stale_command_drop_ms.store(control_now_ms);
            cerr << "[QT_CTRL] dropped stale command source=" << source
                 << " cmd=" << cmd << " seq=" << command_seq
                 << " reason=" << stale_reason << endl;
            send_ack(cmd, false, stale_reason);
            continue;
        }

        if (cmd == "move") {
            manual_owner_code.store(manual_owner_code_from_source(source));
            int pan = clamp_val((int)round(json_get_double(json, "pan", 0.0)), -1, 1);
            int tilt = clamp_val((int)round(json_get_double(json, "tilt", 0.0)), -1, 1);
            int zoom = clamp_val((int)round(json_get_double(json, "zoom", 0.0)), -1, 1);
            int ptz_speed = clamp_val((int)round(json_get_double(json, "speed", QT_MANUAL_PTZ_SPEED)), 5, 100);
            int zoom_speed = clamp_val((int)round(json_get_double(json, "zoom_speed", QT_MANUAL_ZOOM_SPEED)), 5, QT_MANUAL_ZOOM_SPEED_MAX);

            if (pan == 0 && tilt == 0 && zoom == 0) {
                apply_qt_manual_stop("qt_zero_move_stop");
            } else {
                apply_qt_manual_move(pan, tilt, zoom, ptz_speed, zoom_speed);
                cout << "[QT_CTRL] qt_ptz_cmd move pan=" << pan
                     << " tilt=" << tilt << " zoom=" << zoom
                     << " speed=" << ptz_speed << " zoom_speed=" << zoom_speed << endl;
            }
            send_ack(cmd, true, "queued");
        } else if (cmd == "stop") {
            manual_owner_code.store(manual_owner_code_from_source(source));
            apply_qt_manual_stop("qt_release_stop");
            cout << "[QT_CTRL] qt_ptz_cmd stop" << endl;
            send_ack(cmd, true, "stopped");
        } else if (cmd == "emergency_stop") {
            manual_owner_code.store(manual_owner_code_from_source(source));
            ptz_control_enabled.store(false);
            auto_zoom_enabled.store(false);
            apply_qt_manual_stop("qt_emergency_stop");
            send_ack(cmd, true, "stopped_auto_disabled");
        } else if (cmd == "set_auto_track") {
            bool enabled = json_get_bool(json, "enabled", !ptz_control_enabled.load());
            ptz_control_enabled.store(enabled);
            if (!enabled) {
                auto_zoom_enabled.store(false);
                apply_qt_manual_stop("qt_auto_track_disabled");
            }
            send_ack(cmd, true, enabled ? "auto_tracking_on" : "auto_tracking_off");
        } else if (cmd == "set_control_params") {
            string message = apply_qt_control_params(json);
            send_ack(cmd, true, message);
        } else if (cmd == "set_audio_guidance") {
            bool enabled = json_get_bool(json, "enabled", !g_audio_guidance_enabled.load());
            g_audio_guidance_enabled.store(enabled);
            if (!enabled) {
                apply_qt_manual_stop("qt_audio_guidance_disabled");
            }
            cout << "[QT_CTRL] audio_guidance=" << (enabled ? "ON" : "OFF") << endl;
            send_ack(cmd, true, enabled ? "audio_guidance_on" : "audio_guidance_off");
        } else if (cmd == "reset_audio_calibration" || cmd == "reset_audio_learning") {
            const bool reset_learning = (cmd == "reset_audio_learning") ||
                                        json_get_bool(json, "runtime_learning", false);
            reset_live_fusion_calibration(cmd, reset_learning);
            send_ack(cmd, true, reset_learning ? "audio_learning_reset"
                                               : "audio_calibration_reset");
        } else if (cmd == "mark_audio_calibration_target") {
            string message;
            bool ok = mark_current_audio_calibration_target(message);
            send_ack(cmd, ok, message);
        } else if (cmd == "request_autofocus" || cmd == "focus_request") {
            string message;
            bool ok = generate_autofocus_state_request(1, now_ms(), &message);
            send_ack(cmd, ok, message);
        } else if (cmd == "autofocus") {
            // 该分支会触发真实球机配置，仅保留给明确授权的硬件调试；Qt 本地流程使用 request_autofocus。
            bool ok = enable_camera_autofocus();
            send_ack(cmd, ok, ok ? "autofocus_enabled" : "autofocus_failed");
        } else if (cmd == "save_home") {
            PTZState current = get_ptz_status();
            if (current.ok) {
                lock_guard<mutex> lock(g_ptz_home_mtx);
                g_saved_home_ptz = current;
                g_saved_home_ptz_valid = true;
            }
            send_ack(cmd, current.ok, current.ok ? "home_saved" : "ptz_status_failed");
        } else if (cmd == "return_initial" || cmd == "return_home") {
            PTZState target;
            bool valid = false;
            {
                lock_guard<mutex> lock(g_ptz_home_mtx);
                if (cmd == "return_home" && g_saved_home_ptz_valid) {
                    target = g_saved_home_ptz;
                    valid = true;
                } else if (g_session_initial_ptz_valid) {
                    target = g_session_initial_ptz;
                    valid = true;
                }
            }
            apply_qt_manual_stop("qt_return_position");
            bool ok = valid && send_ptz_absolute(
                target.azimuth, target.elevation, target.absolute_zoom);
            send_ack(cmd, ok, ok ? "position_restored" : "home_not_available");
        } else if (cmd == "goto_preset") {
            int preset = static_cast<int>(round(json_get_double(
                json, "preset", static_cast<double>(g_config_home_preset))));
            apply_qt_manual_stop("qt_goto_preset");
            bool ok = preset > 0 && goto_camera_preset(preset, "qt_touch_panel");
            send_ack(cmd, ok, ok ? "preset_called" : "invalid_or_failed_preset");
        } else if (cmd == "click_center") {
            const double nx = clamp_val(json_get_double(json, "x", 0.5), 0.0, 1.0);
            const double ny = clamp_val(json_get_double(json, "y", 0.5), 0.0, 1.0);
            const double dx = nx - 0.5;
            const double dy = ny - 0.5;
            const double magnitude = max(abs(dx), abs(dy));
            if (magnitude < 0.05) {
                apply_qt_manual_stop("qt_click_center_dead_zone");
                send_ack(cmd, true, "already_centered");
            } else {
                const int pan = abs(dx) < 0.05 ? 0 : (dx > 0.0 ? 1 : -1);
                const int tilt = abs(dy) < 0.05 ? 0 : (dy > 0.0 ? 1 : -1);
                const int speed = clamp_val(
                    static_cast<int>(round(10.0 + magnitude * 38.0)), 10, 30);
                apply_qt_manual_move(pan, tilt, 0, speed, QT_MANUAL_ZOOM_SPEED);
                send_ack(cmd, true, "bounded_center_pulse");
            }
        } else {
            cerr << "[QT_CTRL] unknown ptz cmd: " << cmd << endl;
            send_ack(cmd, false, "unknown_command");
        }
    }

    close(sock);
    cout << "[QT_CTRL] receiver stopped" << endl;
}

// ============================================================
// 4. Kalman 跟踪器
// ============================================================

struct KalmanTracker {
    KalmanFilter kf;
    bool initialized = false;

    void init(float sx, float sy) {
        kf.init(4, 2, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kf.measurementMatrix = (Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        setIdentity(kf.processNoiseCov, Scalar::all(0.04));
        setIdentity(kf.measurementNoiseCov, Scalar::all(0.12));
        setIdentity(kf.errorCovPost, Scalar::all(0.1));

        kf.statePost = (Mat_<float>(4, 1) << sx, sy, 0, 0);
        initialized = true;
    }

    Point2f predict() {
        Mat p = kf.predict();
        return Point2f(p.at<float>(0), p.at<float>(1));
    }

    Point2f update(float mx, float my) {
        Mat measurement = (Mat_<float>(2, 1) << mx, my);
        Mat c = kf.correct(measurement);
        return Point2f(c.at<float>(0), c.at<float>(1));
    }
};

// ============================================================
// 5. 图像预处理：真正的 letterbox，不形变
// ============================================================

void letterbox_to_640(const Mat& src_bgr, Mat& infer_rgb, Mat& ui_bgr,
                      float& ratio, float& dw, float& dh) {
    int src_w = src_bgr.cols;
    int src_h = src_bgr.rows;

    ratio = min((float)IMG_SIZE / (float)src_w, (float)IMG_SIZE / (float)src_h);

    int new_w = (int)round(src_w * ratio);
    int new_h = (int)round(src_h * ratio);

    dw = (IMG_SIZE - new_w) / 2.0f;
    dh = (IMG_SIZE - new_h) / 2.0f;

    Mat resized;
    resize(src_bgr, resized, Size(new_w, new_h), 0, 0, INTER_LINEAR);

    ui_bgr = Mat(IMG_SIZE, IMG_SIZE, CV_8UC3, Scalar(114, 114, 114));

    int left = (int)round(dw);
    int top  = (int)round(dh);

    resized.copyTo(ui_bgr(Rect(left, top, new_w, new_h)));
    cvtColor(ui_bgr, infer_rgb, COLOR_BGR2RGB);
}

float clamp_float(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

DetBox map_box_to_original(const ObjectBox& b, const FrameContext& ctx) {
    DetBox d;

    // YOLO 解码结果是 640x640 letterbox 坐标。
    d.ui_cx = b.x;
    d.ui_cy = b.y;
    d.ui_w = b.w;
    d.ui_h = b.h;
    d.conf = b.conf;

    float cx = (b.x - ctx.dw) / ctx.ratio;
    float cy = (b.y - ctx.dh) / ctx.ratio;
    float w  = b.w / ctx.ratio;
    float h  = b.h / ctx.ratio;

    // 裁剪到原始画面范围。
    cx = clamp_float(cx, 0.0f, (float)(ctx.src_w - 1));
    cy = clamp_float(cy, 0.0f, (float)(ctx.src_h - 1));
    w  = clamp_float(w, 1.0f, (float)ctx.src_w);
    h  = clamp_float(h, 1.0f, (float)ctx.src_h);

    d.cx = cx;
    d.cy = cy;
    d.w = w;
    d.h = h;

    return d;
}

Rect det_to_rect(const DetBox& d, int img_w, int img_h) {
    int x = (int)round(d.cx - d.w / 2.0f);
    int y = (int)round(d.cy - d.h / 2.0f);
    int w = (int)round(d.w);
    int h = (int)round(d.h);

    x = max(0, min(x, img_w - 1));
    y = max(0, min(y, img_h - 1));
    w = max(1, min(w, img_w - x));
    h = max(1, min(h, img_h - y));

    return Rect(x, y, w, h);
}


bool open_rtsp_capture(cv::VideoCapture& cap, const std::string& url) {
    cap.release();

    const char* ffmpeg_threads_env = getenv("ANTI_UAV_FFMPEG_THREADS");
    string ffmpeg_threads = (ffmpeg_threads_env && ffmpeg_threads_env[0]) ? ffmpeg_threads_env : "2";
    const char* rtsp_backend_env = getenv("ANTI_UAV_RTSP_BACKEND");
    string rtsp_backend = (rtsp_backend_env && rtsp_backend_env[0]) ? rtsp_backend_env : "gstreamer";
    const char* gst_latency_env = getenv("ANTI_UAV_GST_LATENCY_MS");
    string gst_latency = (gst_latency_env && gst_latency_env[0]) ? gst_latency_env : "50";

    auto try_gstreamer = [&]() -> bool {
        cout << "[VIDEO] try GStreamer Rockchip MPP TCP..." << endl;
        std::string gst =
            "rtspsrc location=" + url +
            " protocols=tcp latency=" + gst_latency + " drop-on-latency=true ! "
            "rtph264depay ! h264parse ! mppvideodec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true";
        if (cap.open(gst, cv::CAP_GSTREAMER)) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cout << "[VIDEO] opened by CAP_GSTREAMER_MPP" << endl;
            return true;
        }

        cap.release();
        cout << "[VIDEO] GStreamer MPP failed, try GStreamer software decoder..." << endl;
        gst =
            "rtspsrc location=" + url +
            " protocols=tcp latency=" + gst_latency + " drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 max-threads=" + ffmpeg_threads + " ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true";
        if (cap.open(gst, cv::CAP_GSTREAMER)) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cout << "[VIDEO] opened by CAP_GSTREAMER_AVDEC" << endl;
            return true;
        }
        cap.release();
        return false;
    };

    if (rtsp_backend != "ffmpeg" && try_gstreamer()) {
        return true;
    }

    cout << "[VIDEO] try OpenCV FFmpeg low-latency TCP..." << endl;

    // 实时控制优先：禁止 FFmpeg 在 RTSP 输入端缓存旧帧，否则画面会出现数秒级“回放式”延迟。
    setenv("OPENCV_FFMPEG_THREADS", ffmpeg_threads.c_str(), 1);
    setenv("OPENCV_FFMPEG_READ_ATTEMPTS", "512", 1);
    setenv("OPENCV_FFMPEG_DECODE_ATTEMPTS", "512", 1);
    string ffmpeg_open_options =
        "rtsp_transport;tcp|stimeout;3000000|fflags;nobuffer|flags;low_delay|"
        "analyzeduration;0|probesize;32768|max_delay;0|reorder_queue_size;0|"
        "buffer_size;102400|threads;" + ffmpeg_threads;
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", ffmpeg_open_options.c_str(), 1);
    cout << "[VIDEO] FFmpeg options: " << ffmpeg_open_options << endl;

    if (cap.open(url, cv::CAP_FFMPEG)) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cout << "[VIDEO] opened by CAP_FFMPEG" << endl;
        return true;
    }

    cap.release();

    cout << "[VIDEO] CAP_FFMPEG failed, try CAP_ANY..." << endl;

    if (cap.open(url, cv::CAP_ANY)) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cout << "[VIDEO] opened by CAP_ANY" << endl;
        return true;
    }

    cap.release();

    if (rtsp_backend == "ffmpeg" && try_gstreamer()) {
        return true;
    }

    cerr << "[VIDEO] all RTSP open methods failed" << endl;
    return false;
}

// ============================================================
// 6. RTSP 采集线程
// ============================================================

void capture_thread(string rtsp_url) {
    const char* ffmpeg_threads_env = getenv("ANTI_UAV_FFMPEG_THREADS");
    string ffmpeg_threads = (ffmpeg_threads_env && ffmpeg_threads_env[0]) ? ffmpeg_threads_env : "2";
    // OpenCV FFmpeg 低延迟参数。
    string ffmpeg_capture_options =
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
        "analyzeduration;100000|probesize;2048|max_delay;0|threads;" + ffmpeg_threads;
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", ffmpeg_capture_options.c_str(), 1);

    size_t shm_size = IMG_SIZE * IMG_SIZE * 3;
    int shm_fd = -1;
    void* shm_ptr = MAP_FAILED;

    if (ENABLE_SHM) {
        shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (shm_fd >= 0) {
            ftruncate(shm_fd, shm_size);
            shm_ptr = mmap(0, shm_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
        }
        if (shm_ptr == MAP_FAILED) {
            cerr << "⚠️ 共享内存创建失败，将只运行识别与 UDP 输出" << endl;
        }
    }

    int frame_id = 0;

    while (is_running.load()) {
        cout << "[VIDEO] opening: " << redact_url_credentials(rtsp_url) << endl;

        VideoCapture cap;

        if (!open_rtsp_capture(cap, rtsp_url)) {
            cerr << "❌ RTSP 打开失败，1 秒后重试。请检查 IP、账号密码、网络脚本。" << endl;
            this_thread::sleep_for(chrono::seconds(1));
            continue;
        }

        cout << "[VIDEO] opened" << endl;

        while (is_running.load()) {
            Mat frame;
            const long long grab_start_ms = now_ms();
            if (!cap.read(frame) || frame.empty()) {
                cerr << "⚠️ RTSP 读取失败，准备重连" << endl;
                break;
            }
            const long long grab_done_ms = now_ms();
            const long long decode_done_ms = now_ms();

            if (ROTATE_IMAGE_180) {
                cv::rotate(frame, frame, cv::ROTATE_180);
            }

            int this_frame_id = frame_id++;

            // 显示线程永远显示最新原始画面，不等 YOLO 推理。
            {
                lock_guard<mutex> lock(latest_camera_mtx);
                latest_camera_frame = frame.clone();
                latest_camera_id = this_frame_id;
            }

            // 实时性优先：INT 默认每帧推理；仅在显式配置 stride>1 时低频送入 YOLO。
            if (g_infer_frame_stride > 1 && this_frame_id % g_infer_frame_stride != 0) {
                continue;
            }

            FrameContext ctx;
            ctx.frame_id = this_frame_id;
            ctx.camera_estimated_mono_ms = grab_done_ms - 40;
            ctx.rtsp_grab_start_mono_ms = grab_start_ms;
            ctx.rtsp_grab_done_mono_ms = grab_done_ms;
            ctx.decode_done_mono_ms = decode_done_ms;
            ctx.src_w = frame.cols;
            ctx.src_h = frame.rows;
            ctx.stream_bgr = frame.clone();

            letterbox_to_640(frame, ctx.infer_rgb, ctx.ui_bgr, ctx.ratio, ctx.dw, ctx.dh);

            if (ENABLE_SHM && shm_ptr != MAP_FAILED && ctx.ui_bgr.isContinuous()) {
                memcpy(shm_ptr, ctx.ui_bgr.data, shm_size);
            }

            input_queue.push_latest(std::move(ctx));
        }

        cap.release();
        this_thread::sleep_for(chrono::milliseconds(300));
    }

    if (shm_ptr != MAP_FAILED) munmap(shm_ptr, shm_size);
    if (shm_fd >= 0) close(shm_fd);
}

// ============================================================
// 7. RKNN 推理线程
// ============================================================

void inference_thread(int thread_id, rknn_context ctx_rknn) {
    while (is_running.load()) {
        FrameContext ctx;
        if (!input_queue.pop(ctx)) continue;

        if (ctx.infer_rgb.empty() || !ctx.infer_rgb.isContinuous()) continue;

        if (!g_detection_enabled.load()) {
            ctx.decoded_boxes.clear();
            g_latest_yolo_infer_ms.store(0.0);
            ctx.infer_rgb.release();
            output_queue.push_latest(std::move(ctx));
            continue;
        }

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = IMG_SIZE * IMG_SIZE * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = ctx.infer_rgb.data;

        int ret = rknn_inputs_set(ctx_rknn, 1, inputs);
        if (ret < 0) {
            cerr << "[RKNN] inputs_set failed: " << ret << endl;
            continue;
        }

        ctx.infer_start_mono_ms = now_ms();
        auto __metric_infer_t0 = std::chrono::steady_clock::now();
        ret = rknn_run(ctx_rknn, NULL);
        if (ret < 0) {
            cerr << "[RKNN] run failed: " << ret << endl;
            continue;
        }

        // 支持两类模型：
        // 1 输出：旧 FP/单输出模型 [1,5,8400]
        // 6 输出：新版 INT8 raw-head，reg/cls 三尺度输出，后处理在 C++ 完成。
        const int output_count = g_rknn_output_count;
        vector<rknn_output> outputs(output_count);
        memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
        for (int i = 0; i < output_count; ++i) {
            outputs[i].want_float = 1;
        }

        ret = rknn_outputs_get(ctx_rknn, output_count, outputs.data(), NULL);
        if (ret < 0) {
            cerr << "[RKNN] outputs_get failed: " << ret << endl;
            continue;
        }

        if (output_count == 1) {
            ctx.decoded_boxes = decode_yolo11_single_output(outputs.data(), IMG_SIZE);
        } else if (output_count == 6) {
            ctx.decoded_boxes = decode_yolo11_raw_head_outputs(outputs.data(), output_count, IMG_SIZE);
        } else {
            static int warn_count = 0;
            if (warn_count++ < 20) {
                cerr << "[RKNN] unsupported output_count=" << output_count
                     << ", expected 1 or 6" << endl;
            }
            ctx.decoded_boxes.clear();
        }

        rknn_outputs_release(ctx_rknn, output_count, outputs.data());

        
        {
            auto __metric_infer_t1 = std::chrono::steady_clock::now();
            double __metric_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                __metric_infer_t1 - __metric_infer_t0
            ).count() / 1000.0;
            g_latest_yolo_infer_ms.store(__metric_ms);
        }
        ctx.infer_end_mono_ms = now_ms();
ctx.infer_rgb.release();
        output_queue.push_latest(std::move(ctx));
    }
}

// ============================================================
// 8. 后处理线程：取消环境学习，输出稳定目标
// ============================================================

void send_udp_json(int sock, const sockaddr_in& addr, const string& json) {
    std::string out = json;

    // METRICS INT8 PATCH: 不破坏原有字段，只在 JSON 开头注入论文实验字段
    if (!out.empty() && out[0] == '{') {
        ostringstream prefix;
        prefix << fixed << setprecision(3)
               << "\"type\":\"vision\","
               << "\"ts_ms\":" << metric_now_ms_cpp() << ","
               << "\"ts_mono_ms\":" << now_ms() << ","
               << "\"model_mode\":\"" << metric_model_mode_cpp() << "\","
               << "\"infer_ms\":" << g_latest_yolo_infer_ms.load() << ","
               << "\"fps\":" << g_latest_vision_fps.load() << ","
               << "\"camera_mode_cmd\":\"" << camera_mode_name_from_preset(g_camera_mode_preset_cmd.load()) << "\","
               << "\"camera_profile\":\"" << camera_profile_to_string(current_camera_profile()) << "\","
               << "\"is_grayscale\":" << (g_camera_is_grayscale.load() ? 1 : 0) << ","
               << "\"is_grayscale_like\":" << (g_camera_is_grayscale.load() ? 1 : 0) << ","
               << "\"brightness_mean\":" << g_camera_brightness_mean.load() << ","
               << "\"contrast_std\":" << g_camera_contrast_std.load() << ","
               << "\"saturation_mean\":" << g_camera_saturation_mean.load() << ","
               << "\"blur_laplacian_var\":" << g_camera_blur_laplacian_var.load() << ","
               << "\"image_quality_warning\":\"" << json_escape_string(image_quality_warning_snapshot()) << "\","
               << "\"last_profile_switch_reason\":\"" << json_escape_string(last_profile_switch_reason_snapshot()) << "\","
               << "\"auto_camera_profile_enabled\":" << (g_auto_camera_profile_enabled.load() ? 1 : 0) << ","
               << "\"ptz_moving\":" << (g_camera_ptz_moving.load() ? 1 : 0) << ","
               << "\"camera_azimuth_unit\":" << g_latest_ptz_azimuth_unit.load() << ","
               << "\"camera_elevation_unit\":" << g_latest_ptz_elevation_unit.load() << ","
               << "\"camera_absolute_zoom\":" << g_latest_ptz_absolute_zoom.load() << ","
               << "\"auto_tracking_enabled\":" << (ptz_control_enabled.load() ? 1 : 0) << ","
               << "\"auto_zoom_enabled\":" << (auto_zoom_enabled.load() ? 1 : 0) << ","
               << "\"quality_track_state\":\"" << json_escape_string(camera_track_state_snapshot()) << "\","
               << "\"quality_zoom_ratio\":" << g_camera_zoom_ratio.load() << ","
               << "\"quality_detected\":" << g_camera_detected.load() << ","
               << "\"lost_duration_ms\":" << (g_camera_lost_since_ms.load() > 0 ? (now_ms() - g_camera_lost_since_ms.load()) : 0) << ","
               << system_resource_json_fragment();

        // 如果原 JSON 已经有 type，就不重复插 type，但仍保留原始发送
        if (out.find("\"type\"") == std::string::npos) {
            out.insert(1, prefix.str());
        }
    }

    sendto(sock, out.c_str(), out.size(), 0, (const struct sockaddr*)&addr, sizeof(addr));
}

string make_json_no_target(int frame_id, int fw, int fh) {
    string suffix = fusion_json_suffix();
    char buf[8192];
    snprintf(buf, sizeof(buf),
             "{\"frame_id\":%d,\"detected\":0,\"tracking\":0,\"track_id\":-1,\"lost_frames\":0,\"state\":\"Lost\",\"fw\":%d,\"fh\":%d,"
             "\"bbox_x\":0,\"bbox_y\":0,\"bbox_w\":0,\"bbox_h\":0,"
             "\"cx\":0,\"cy\":0,\"dx\":0,\"dy\":0,\"conf\":0%s}",
             frame_id, fw, fh, suffix.c_str());
    return string(buf);
}

string make_json_target(int frame_id, int fw, int fh, const DetBox& d, bool tracking,
                        int track_id = -1, int lost_frames = 0,
                        const string& tracker_state = "Tracked") {
    int x = (int)round(d.cx - d.w / 2.0f);
    int y = (int)round(d.cy - d.h / 2.0f);
    int w = (int)round(d.w);
    int h = (int)round(d.h);

    float dx = d.cx - fw / 2.0f;
    float dy = d.cy - fh / 2.0f;

    string suffix = fusion_json_suffix();
    char buf[8192];
    snprintf(buf, sizeof(buf),
             "{\"frame_id\":%d,\"detected\":1,\"tracking\":%d,\"track_id\":%d,\"lost_frames\":%d,\"state\":\"%s\",\"fw\":%d,\"fh\":%d,"
             "\"bbox_x\":%d,\"bbox_y\":%d,\"bbox_w\":%d,\"bbox_h\":%d,"
             "\"cx\":%.1f,\"cy\":%.1f,\"dx\":%.1f,\"dy\":%.1f,\"conf\":%.3f,"
             "\"ui_cx\":%.1f,\"ui_cy\":%.1f,\"ui_w\":%.1f,\"ui_h\":%.1f%s}",
             frame_id,
             tracking ? 1 : 0,
             track_id,
             lost_frames,
             tracker_state.c_str(),
             fw,
             fh,
             x, y, w, h,
             d.cx, d.cy, dx, dy, d.conf,
             d.ui_cx, d.ui_cy, d.ui_w, d.ui_h,
             suffix.c_str());
    return string(buf);
}

string make_frame_stream_json(const FrameContext& ctx, const DetBox* target, bool tracking,
                               int track_id = -1, int lost_frames = 0,
                               const string& tracker_state = "Tracked",
                               long long stream_enqueue_mono_ms = -1) {
    string suffix = fusion_json_suffix();
    const int stream_w = g_frame_stream_letterbox640 ? IMG_SIZE :
        (ctx.src_w > 0 ? ctx.src_w : IMG_SIZE);
    const int stream_h = g_frame_stream_letterbox640 ? IMG_SIZE :
        (ctx.src_h > 0 ? ctx.src_h : IMG_SIZE);
    char buf[8192];

    if (target != nullptr) {
        const float stream_cx = g_frame_stream_letterbox640 ? target->ui_cx : target->cx;
        const float stream_cy = g_frame_stream_letterbox640 ? target->ui_cy : target->cy;
        const float stream_box_w = g_frame_stream_letterbox640 ? target->ui_w : target->w;
        const float stream_box_h = g_frame_stream_letterbox640 ? target->ui_h : target->h;
        const int x = (int)round(stream_cx - stream_box_w / 2.0f);
        const int y = (int)round(stream_cy - stream_box_h / 2.0f);
        const int w = (int)round(stream_box_w);
        const int h = (int)round(stream_box_h);
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"vision_frame\",\"video_source\":\"rk_stream\","
                 "\"frame_id\":%d,\"detected\":1,\"tracking\":%d,\"track_id\":%d,\"lost_frames\":%d,\"state\":\"%s\","
                 "\"fw\":%d,\"fh\":%d,\"coord_w\":%d,\"coord_h\":%d,"
                 "\"bbox_x\":%d,\"bbox_y\":%d,\"bbox_w\":%d,\"bbox_h\":%d,"
                 "\"cx\":%.1f,\"cy\":%.1f,\"dx\":%.1f,\"dy\":%.1f,\"conf\":%.3f,"
                 "\"ui_cx\":%.1f,\"ui_cy\":%.1f,\"ui_w\":%.1f,\"ui_h\":%.1f,"
                  "\"infer_ms\":%.3f,\"fps\":%.3f,\"ts_ms\":%lld,\"ts_mono_ms\":%lld,"
                  "\"camera_ts_estimated\":1,\"camera_estimated_mono_ms\":%lld,"
                  "\"rtsp_grab_start_mono_ms\":%lld,\"rtsp_grab_done_mono_ms\":%lld,"
                  "\"decode_done_mono_ms\":%lld,\"infer_start_mono_ms\":%lld,"
                  "\"infer_end_mono_ms\":%lld,\"postprocess_done_mono_ms\":%lld,"
                  "\"stream_enqueue_mono_ms\":%lld%s}",
                 ctx.frame_id,
                 tracking ? 1 : 0,
                 track_id,
                 lost_frames,
                 tracker_state.c_str(),
                 stream_w, stream_h, stream_w, stream_h,
                 x, y, w, h,
                  stream_cx,
                  stream_cy,
                  stream_cx - stream_w / 2.0f,
                  stream_cy - stream_h / 2.0f,
                 target->conf,
                 target->ui_cx, target->ui_cy, target->ui_w, target->ui_h,
                 g_latest_yolo_infer_ms.load(),
                  g_latest_vision_fps.load(),
                  metric_now_ms_cpp(),
                  now_ms(),
                  ctx.camera_estimated_mono_ms,
                  ctx.rtsp_grab_start_mono_ms,
                  ctx.rtsp_grab_done_mono_ms,
                  ctx.decode_done_mono_ms,
                  ctx.infer_start_mono_ms,
                  ctx.infer_end_mono_ms,
                  ctx.postprocess_done_mono_ms,
                  stream_enqueue_mono_ms,
                  suffix.c_str());
    } else {
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"vision_frame\",\"video_source\":\"rk_stream\","
                 "\"frame_id\":%d,\"detected\":0,\"tracking\":0,\"track_id\":-1,\"lost_frames\":0,\"state\":\"Lost\","
                 "\"fw\":%d,\"fh\":%d,\"coord_w\":%d,\"coord_h\":%d,"
                 "\"bbox_x\":0,\"bbox_y\":0,\"bbox_w\":0,\"bbox_h\":0,"
                 "\"cx\":0,\"cy\":0,\"dx\":0,\"dy\":0,\"conf\":0,"
                  "\"infer_ms\":%.3f,\"fps\":%.3f,\"ts_ms\":%lld,\"ts_mono_ms\":%lld,"
                  "\"camera_ts_estimated\":1,\"camera_estimated_mono_ms\":%lld,"
                  "\"rtsp_grab_start_mono_ms\":%lld,\"rtsp_grab_done_mono_ms\":%lld,"
                  "\"decode_done_mono_ms\":%lld,\"infer_start_mono_ms\":%lld,"
                  "\"infer_end_mono_ms\":%lld,\"postprocess_done_mono_ms\":%lld,"
                  "\"stream_enqueue_mono_ms\":%lld%s}",
                 ctx.frame_id,
                 stream_w, stream_h, stream_w, stream_h,
                 g_latest_yolo_infer_ms.load(),
                  g_latest_vision_fps.load(),
                  metric_now_ms_cpp(),
                  now_ms(),
                  ctx.camera_estimated_mono_ms,
                  ctx.rtsp_grab_start_mono_ms,
                  ctx.rtsp_grab_done_mono_ms,
                  ctx.decode_done_mono_ms,
                  ctx.infer_start_mono_ms,
                  ctx.infer_end_mono_ms,
                  ctx.postprocess_done_mono_ms,
                  stream_enqueue_mono_ms,
                  suffix.c_str());
    }
    return string(buf);
}

void publish_frame_stream(const FrameContext& ctx, const DetBox* target, bool tracking,
                          int track_id = -1, int lost_frames = 0,
                          const string& tracker_state = "Tracked") {
    Mat frame = g_frame_stream_letterbox640 ? ctx.ui_bgr : ctx.stream_bgr;
    if (g_frame_stream_port <= 0 || frame.empty()) return;

    if (g_frame_stream_max_width > 0 && frame.cols > g_frame_stream_max_width) {
        Mat scaled;
        const double scale = static_cast<double>(g_frame_stream_max_width) / frame.cols;
        resize(frame, scaled,
               Size(g_frame_stream_max_width,
                    max(1, static_cast<int>(round(frame.rows * scale)))),
               0, 0, INTER_AREA);
        frame = std::move(scaled);
    }

    const long long jpeg_start_ms = now_ms();
    vector<uchar> jpg;
    vector<int> params = {IMWRITE_JPEG_QUALITY, g_frame_stream_jpeg_quality};
    if (!imencode(".jpg", frame, jpg, params) || jpg.empty()) return;
    const long long jpeg_end_ms = now_ms();

    const long long enqueue_ms = now_ms();
    string meta = make_frame_stream_json(
        ctx, target, tracking, track_id, lost_frames, tracker_state, enqueue_ms);
    if (!meta.empty() && meta.front() == '{') {
        ostringstream timing;
        timing << "\"jpeg_start_mono_ms\":" << jpeg_start_ms << ","
               << "\"jpeg_end_mono_ms\":" << jpeg_end_ms << ",";
        timing << "\"jpeg_payload_size\":" << jpg.size() << ","
               << "\"payload_size\":" << jpg.size() << ",";
        meta.insert(1, timing.str());
    }
    {
        lock_guard<mutex> lock(latest_stream_mtx);
        latest_stream_meta = std::move(meta);
        latest_stream_jpg = std::move(jpg);
        latest_stream_seq++;
    }
    latest_stream_cv.notify_all();
}

bool send_all_frame_stream(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0 && is_running.load()) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, p, len, 0);
#endif
        if (n <= 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return len == 0;
}

void frame_stream_client_thread(int client_fd, string client_ip) {
    int nodelay = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF,
               &g_frame_stream_send_buffer, sizeof(g_frame_stream_send_buffer));
    timeval send_timeout{};
    send_timeout.tv_sec = g_frame_stream_send_timeout_ms / 1000;
    send_timeout.tv_usec = (g_frame_stream_send_timeout_ms % 1000) * 1000;
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    cout << "[FRAME_STREAM] client connected: " << client_ip << endl;

    uint64_t sent_seq = 0;
    uint64_t client_dropped_frames = 0;
    long long last_client_stat_ms = 0;
    while (is_running.load()) {
        string meta;
        vector<uchar> jpg;
        uint64_t seq = 0;
        {
            unique_lock<mutex> lock(latest_stream_mtx);
            latest_stream_cv.wait_for(lock, chrono::milliseconds(500), [&]() {
                return !is_running.load() || latest_stream_seq > sent_seq;
            });
            if (!is_running.load()) break;
            if (latest_stream_seq <= sent_seq || latest_stream_jpg.empty()) continue;
            if (sent_seq > 0 && latest_stream_seq > sent_seq + 1) {
                client_dropped_frames += latest_stream_seq - sent_seq - 1;
            }
            sent_seq = latest_stream_seq;
            seq = latest_stream_seq;
            meta = latest_stream_meta;
            jpg = latest_stream_jpg;
        }

        const long long tcp_send_start_ms = now_ms();
        if (!meta.empty() && meta.front() == '{') {
            ostringstream timing;
            timing << "\"tcp_send_start_mono_ms\":" << tcp_send_start_ms << ","
                   << "\"yolo_stream_send_ts_ms\":" << tcp_send_start_ms << ","
                   << "\"send_ts_ms\":" << tcp_send_start_ms << ","
                   << "\"frame_stream_seq\":" << seq << ","
                   << "\"client_dropped_frames\":" << client_dropped_frames << ",";
            meta.insert(1, timing.str());
        }

        string header = "AUVF " + to_string(meta.size()) + " " + to_string(jpg.size()) + "\n";
        if (!send_all_frame_stream(client_fd, header.data(), header.size()) ||
            !send_all_frame_stream(client_fd, meta.data(), meta.size()) ||
            !send_all_frame_stream(client_fd, jpg.data(), jpg.size())) {
            cerr << "[FRAME_STREAM] slow/disconnected client dropped: " << client_ip
                 << " dropped_frames=" << client_dropped_frames << endl;
            break;
        }
        const long long send_done_ms = now_ms();
        if (send_done_ms - last_client_stat_ms >= 1000) {
            last_client_stat_ms = send_done_ms;
            cout << "[FRAME_STREAM] client=" << client_ip
                 << " seq=" << seq
                 << " client_send_ms=" << (send_done_ms - tcp_send_start_ms)
                 << " client_dropped_frames=" << client_dropped_frames
                 << " jpg_bytes=" << jpg.size() << endl;
        }
    }
    close(client_fd);
    cout << "[FRAME_STREAM] client disconnected: " << client_ip << endl;
}

void frame_stream_thread() {
    if (g_frame_stream_port <= 0) {
        cout << "[FRAME_STREAM] disabled" << endl;
        return;
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        cerr << "[FRAME_STREAM] socket failed" << endl;
        return;
    }

    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(g_frame_stream_port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        cerr << "[FRAME_STREAM] bind failed on port " << g_frame_stream_port << endl;
        close(server_fd);
        return;
    }

    if (listen(server_fd, 4) < 0) {
        cerr << "[FRAME_STREAM] listen failed" << endl;
        close(server_fd);
        return;
    }

    cout << "[FRAME_STREAM] listening on 0.0.0.0:" << g_frame_stream_port << endl;

    while (is_running.load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        int ready = select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;
        thread(frame_stream_client_thread, client_fd,
               string(inet_ntoa(client_addr.sin_addr))).detach();
    }

    close(server_fd);
}

void draw_debug_on_ui(const FrameContext& ctx, const DetBox* target, bool tracking) {
    if (!ENABLE_IMSHOW && !ENABLE_SHM) return;

    lock_guard<mutex> lock(latest_ui_mtx);
    if (latest_ui_frame.empty()) return;

    Mat img = ctx.ui_bgr.clone();

    // 中心参考线
    drawMarker(img, Point(IMG_SIZE / 2, IMG_SIZE / 2), Scalar(0, 255, 255), MARKER_CROSS, 32, 2);

    if (target != nullptr) {
        int x = (int)round(target->ui_cx - target->ui_w / 2.0f);
        int y = (int)round(target->ui_cy - target->ui_h / 2.0f);
        int w = (int)round(target->ui_w);
        int h = (int)round(target->ui_h);
        rectangle(img, Rect(max(0, x), max(0, y), max(1, w), max(1, h)), Scalar(0, 255, 0), 2);
        circle(img, Point((int)target->ui_cx, (int)target->ui_cy), 4, Scalar(0, 0, 255), -1);

        char text[128];
        snprintf(text, sizeof(text), "%s conf=%.2f", tracking ? "TRACK" : "DETECT", target->conf);
        putText(img, text, Point(max(0, x), max(20, y - 6)), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 2);
    }

    latest_ui_frame = img;
}


enum class TrackState {
    SEARCHING = 0,
    CANDIDATE = 1,
    CONFIRMED = 2
};

const char* state_to_string(TrackState state) {
    switch (state) {
        case TrackState::SEARCHING: return "SEARCH";
        case TrackState::CANDIDATE: return "CANDIDATE";
        case TrackState::CONFIRMED: return "TRACK";
        default: return "UNKNOWN";
    }
}

struct DetectionPolicySnapshot {
    int mode = 1;
    float low_thresh = 0.35f;
    float high_thresh = 0.45f;
    float search_conf = 0.35f;
    float search_min_box = 0.04f;
    float track_conf = 0.45f;
    float track_min_box = 0.06f;
    float alarm_conf = 0.45f;
    float alarm_min_box = 0.08f;
    int confirm_frames = 3;
};

static DetectionPolicySnapshot current_detection_policy_snapshot() {
    DetectionPolicySnapshot p;
    p.mode = g_detection_policy_mode.load();
    p.search_conf = g_search_conf_thresh.load();
    p.search_min_box = g_search_min_box_ratio.load();
    p.track_conf = g_track_conf_thresh.load();
    p.track_min_box = g_track_min_box_ratio.load();
    p.alarm_conf = g_alarm_conf_thresh.load();
    p.alarm_min_box = g_alarm_min_box_ratio.load();
    p.confirm_frames = g_search_confirm_frames.load();

    if (p.mode == 0) { // 高召回搜索
        p.search_conf = min(p.search_conf, 0.35f);
        p.search_min_box = 0.04f;
        p.confirm_frames = 3;
    } else if (p.mode == 1) { // 均衡搜索
        p.search_conf = max(0.35f, min(p.search_conf, 0.45f));
        p.search_min_box = 0.04f;
        p.confirm_frames = 3;
    } else if (p.mode == 2) { // 低误判确认
        p.search_conf = p.track_conf;
        p.search_min_box = p.track_min_box;
        p.confirm_frames = g_track_confirm_frames.load();
    } else if (p.mode == 3) { // 保守报警
        p.search_conf = p.alarm_conf;
        p.search_min_box = p.alarm_min_box;
        p.confirm_frames = g_alarm_confirm_frames.load();
    }

    p.low_thresh = clampf(p.search_conf, 0.20f, 0.90f);
    p.high_thresh = clampf(max(p.search_conf, p.track_conf), 0.20f, 0.90f);
    p.search_min_box = clampf(p.search_min_box, 0.01f, 0.30f);
    p.track_min_box = clampf(p.track_min_box, 0.01f, 0.30f);
    p.alarm_min_box = clampf(p.alarm_min_box, 0.01f, 0.35f);
    p.confirm_frames = clamp_val(p.confirm_frames, 1, 8);
    return p;
}

bool is_reasonable_box(const DetBox& d, int fw, int fh) {
    if (d.conf < 0.0f) return false;
    if (d.w < 2.0f || d.h < 2.0f) return false;

    float frame_area = max(1.0f, (float)(fw * fh));
    float area = d.w * d.h;
    float area_ratio = area / frame_area;
    float max_side_ratio = max(d.w / max(1.0f, (float)fw),
                               d.h / max(1.0f, (float)fh));
    DetectionPolicySnapshot policy = current_detection_policy_snapshot();

    if (area_ratio < MIN_BOX_AREA_RATIO) return false;
    if (max_side_ratio < policy.search_min_box) return false;
    if (area_ratio > MAX_BOX_AREA_RATIO) return false;

    float aspect = d.w / max(1.0f, d.h);
    if (aspect < MIN_BOX_ASPECT || aspect > MAX_BOX_ASPECT) return false;

    return true;
}

bool size_is_consistent(const DetBox& prev, const DetBox& now) {
    float prev_area = max(1.0f, prev.w * prev.h);
    float now_area = max(1.0f, now.w * now.h);
    float ratio = now_area / prev_area;

    if (ratio > MAX_SIZE_JUMP_RATIO) return false;
    if (ratio < 1.0f / MAX_SIZE_JUMP_RATIO) return false;

    return true;
}

void postprocess_thread() {
    ByteTrackLiteConfig tracker_cfg = loadByteTrackLiteConfig(g_tracker_config_path);
    DetectionPolicySnapshot initial_policy = current_detection_policy_snapshot();
    tracker_cfg.low_thresh = initial_policy.low_thresh;
    tracker_cfg.high_thresh = initial_policy.high_thresh;
    tracker_cfg.confirm_frames = initial_policy.confirm_frames;
    ByteTrackLite tracker(tracker_cfg);

    auto start_time = chrono::high_resolution_clock::now();
    int highest_processed_id = -1;
    bool last_detection_enabled = g_detection_enabled.load();
    long long applied_control_params_version = g_control_params_version.load();

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(g_udp_port);
    servaddr.sin_addr.s_addr = inet_addr(g_udp_ip.c_str());

    cout << "✅ 后处理启动：ByteTrack-Lite + PTZ-aware GMC，不使用 ReID/深度特征" << endl;
    cout << "✅ tracker_config=" << g_tracker_config_path
         << " high=" << tracker_cfg.high_thresh
         << " low=" << tracker_cfg.low_thresh
         << " match=" << tracker_cfg.match_thresh
         << " buffer=" << tracker_cfg.track_buffer
         << " confirm=" << tracker_cfg.confirm_frames
         << " sign_pan=" << tracker_cfg.sign_pan
         << " sign_tilt=" << tracker_cfg.sign_tilt
         << " k_pan=" << tracker_cfg.k_pan
         << " k_tilt=" << tracker_cfg.k_tilt << endl;

    while (is_running.load()) {
        FrameContext ctx;
        if (!output_queue.pop(ctx)) continue;

        if (ctx.frame_id <= highest_processed_id) continue;
        highest_processed_id = ctx.frame_id;
        update_camera_image_mode_stats(ctx.ui_bgr);
        long long params_version = g_control_params_version.load();
        if (params_version != applied_control_params_version) {
            DetectionPolicySnapshot policy = current_detection_policy_snapshot();
            tracker_cfg.low_thresh = policy.low_thresh;
            tracker_cfg.high_thresh = policy.high_thresh;
            tracker_cfg.confirm_frames = policy.confirm_frames;
            tracker = ByteTrackLite(tracker_cfg);
            applied_control_params_version = params_version;
            cout << "\n[TUNE] applied detection policy mode=" << policy.mode
                 << " low=" << tracker_cfg.low_thresh
                 << " high=" << tracker_cfg.high_thresh
                 << " min_box=" << policy.search_min_box
                 << " confirm=" << tracker_cfg.confirm_frames << endl;
        }

        bool detection_enabled = g_detection_enabled.load();
        if (!detection_enabled) {
            if (last_detection_enabled) {
                tracker = ByteTrackLite(tracker_cfg);
            }
            last_detection_enabled = false;
            g_camera_detected.store(0);
            g_camera_lost_since_ms.store(now_ms());
            set_camera_quality_text("DISABLED", "");
            update_shared_detection(false, false, ctx.frame_id, ctx.src_w, ctx.src_h, nullptr);
            string json = make_json_no_target(ctx.frame_id, ctx.src_w, ctx.src_h);
            send_udp_json(sock, servaddr, json);
            publish_frame_stream(ctx, nullptr, false);
            draw_debug_on_ui(ctx, nullptr, false);

            auto now = chrono::high_resolution_clock::now();
            float elapsed = chrono::duration_cast<chrono::microseconds>(now - start_time).count() / 1000000.0f;
            float fps = (highest_processed_id + 1) / max(0.001f, elapsed);
            g_latest_vision_fps.store(fps);
            {
                lock_guard<mutex> lock(latest_ui_mtx);
                latest_status_text = "Detection disabled from Qt control";
            }
            continue;
        }
        if (!last_detection_enabled) {
            tracker = ByteTrackLite(tracker_cfg);
        }
        last_detection_enabled = true;

        // 1. 将 RKNN 解码框映射回海康原始画面，并做几何过滤。
        vector<Rect> nms_boxes;
        vector<float> nms_scores;
        vector<DetBox> dets;

        // NMS 前保留所有 >= low_thresh 的检测，ByteTrack 第二阶段会使用低分框续接轨迹。
        for (const auto& raw : ctx.decoded_boxes) {
            if (raw.conf < tracker_cfg.low_thresh) continue;

            DetBox d = map_box_to_original(raw, ctx);
            if (!is_reasonable_box(d, ctx.src_w, ctx.src_h)) continue;

            dets.push_back(d);
            nms_boxes.push_back(det_to_rect(d, ctx.src_w, ctx.src_h));
            nms_scores.push_back(d.conf);
        }

        vector<int> indices;
        if (!nms_boxes.empty()) {
            dnn::NMSBoxes(nms_boxes, nms_scores, tracker_cfg.low_thresh, IOU_THRESH, indices);
        }

        vector<ByteTrackDetection> tracker_dets;
        for (int idx : indices) {
            const DetBox& d = dets[idx];
            ByteTrackDetection td;
            td.x1 = d.cx - d.w * 0.5f;
            td.y1 = d.cy - d.h * 0.5f;
            td.x2 = d.cx + d.w * 0.5f;
            td.y2 = d.cy + d.h * 0.5f;
            td.score = d.conf;
            td.class_id = 0; // 当前无人机模型为单类别，统一映射为 drone class_id=0。
            tracker_dets.push_back(td);
        }

        ByteTrackPtzState ptz_snapshot;
        ptz_snapshot.valid = g_latest_ptz_state_ok.load();
        ptz_snapshot.frame_width = ctx.src_w;
        ptz_snapshot.frame_height = ctx.src_h;
        ptz_snapshot.pan_deg = hik_azimuth_to_deg(g_latest_ptz_azimuth_unit.load());
        ptz_snapshot.tilt_deg = g_latest_ptz_elevation_unit.load() / 10.0;
        ptz_snapshot.zoom_ratio = zoom_to_ratio(g_latest_ptz_absolute_zoom.load());
        auto current_fov = byteTrackInterpFov(tracker_cfg, ptz_snapshot.zoom_ratio);
        ptz_snapshot.hfov_deg = current_fov.first;
        ptz_snapshot.vfov_deg = current_fov.second;

        vector<ByteTrackOutput> tracks = tracker.update(tracker_dets, ptz_snapshot);

        // 2. 输出稳定目标。多目标时优先 lost_frames 更少、score 更高的 track。
        bool output_target = false;
        bool is_confirmed_output = false;
        DetBox target;
        int output_track_id = -1;
        int output_lost_frames = 0;
        string output_track_state = "Lost";

        if (!tracks.empty()) {
            const ByteTrackOutput& tr = tracks.front();
            target.cx = clamp_float(tr.center_x, 0.0f, (float)(ctx.src_w - 1));
            target.cy = clamp_float(tr.center_y, 0.0f, (float)(ctx.src_h - 1));
            target.w = clamp_float(tr.x2 - tr.x1, 1.0f, (float)ctx.src_w);
            target.h = clamp_float(tr.y2 - tr.y1, 1.0f, (float)ctx.src_h);
            target.conf = tr.score;
            target.ui_cx = target.cx * ctx.ratio + ctx.dw;
            target.ui_cy = target.cy * ctx.ratio + ctx.dh;
            target.ui_w = target.w * ctx.ratio;
            target.ui_h = target.h * ctx.ratio;
            output_track_id = tr.track_id;
            output_lost_frames = tr.lost_frames;
            output_track_state = tr.state;
            output_target = true;
            is_confirmed_output = tr.confirmed;
        }

        // 3. UDP 输出 + 共享给显示/云台控制线程。
        ctx.postprocess_done_mono_ms = now_ms();
        string json;
        if (output_target) {
            g_camera_detected.store(1);
            g_camera_lost_since_ms.store(0);
            set_camera_quality_text(output_track_state, "");
            update_shared_detection(true, is_confirmed_output, ctx.frame_id, ctx.src_w, ctx.src_h, &target);
            json = make_json_target(ctx.frame_id, ctx.src_w, ctx.src_h, target, is_confirmed_output,
                                    output_track_id, output_lost_frames, output_track_state);
            send_udp_json(sock, servaddr, json);
            publish_frame_stream(ctx, &target, is_confirmed_output,
                                 output_track_id, output_lost_frames, output_track_state);
            draw_debug_on_ui(ctx, &target, is_confirmed_output);
        } else {
            g_camera_detected.store(0);
            if (g_camera_lost_since_ms.load() == 0) {
                g_camera_lost_since_ms.store(now_ms());
            }
            set_camera_quality_text("SEARCH", "");
            update_shared_detection(false, false, ctx.frame_id, ctx.src_w, ctx.src_h, nullptr);
            json = make_json_no_target(ctx.frame_id, ctx.src_w, ctx.src_h);
            send_udp_json(sock, servaddr, json);
            publish_frame_stream(ctx, nullptr, false);
            draw_debug_on_ui(ctx, nullptr, false);
        }

        auto now = chrono::high_resolution_clock::now();
        float elapsed = chrono::duration_cast<chrono::microseconds>(now - start_time).count() / 1000000.0f;
        float fps = (highest_processed_id + 1) / max(0.001f, elapsed);
        g_latest_vision_fps.store(fps);

        char status[320];
        if (output_target) {
            snprintf(status, sizeof(status),
                     "FPS:%4.1f | BT-%s id:%d | conf:%.2f | cx:%4.0f cy:%4.0f | dx:%+5.0f dy:%+5.0f | lost:%d | frame:%dx%d",
                     fps,
                     output_track_state.c_str(),
                     output_track_id,
                     target.conf,
                     target.cx,
                     target.cy,
                     target.cx - ctx.src_w / 2.0f,
                     target.cy - ctx.src_h / 2.0f,
                     output_lost_frames,
                     ctx.src_w,
                     ctx.src_h);
        } else {
            snprintf(status, sizeof(status),
                     "FPS:%4.1f | ByteTrack | no stable output | dets:%zu tracks:%zu | frame:%dx%d",
                     fps,
                     tracker_dets.size(),
                     tracks.size(),
                     ctx.src_w,
                     ctx.src_h);
        }

        {
            lock_guard<mutex> lock(latest_ui_mtx);
            latest_status_text = status;
        }

        if (highest_processed_id % 10 == 0 || output_target) {
            printf("\r[YOLO] %s | PTZ:%s     ",
                   status,
                   ptz_control_enabled.load() ? "ON" : "OFF");
            fflush(stdout);
        }
    }

    close(sock);
}

// ============================================================
// 9. 调试显示线程
// ============================================================

void display_thread() {
    if (!ENABLE_IMSHOW) return;

    namedWindow("Hikvision RKNN YOLO", WINDOW_AUTOSIZE);

    while (is_running.load()) {
        Mat img;
        int cam_id = -1;

        {
            lock_guard<mutex> lock(latest_camera_mtx);
            if (!latest_camera_frame.empty()) {
                img = latest_camera_frame.clone();
                cam_id = latest_camera_id;
            }
        }

        string text;
        {
            lock_guard<mutex> lock(latest_ui_mtx);
            text = latest_status_text;
        }

        SharedDetection det;
        {
            lock_guard<mutex> lock(latest_det_mtx);
            det = latest_det;
        }

        if (!img.empty()) {
            int w = img.cols;
            int h = img.rows;
            int cx = w / 2;
            int cy = h / 2;

            int dzx = max((int)CENTER_DEAD_ZONE_X_MIN_PX, (int)round(w * g_center_dead_zone_x_ratio.load()));
            int dzy = max((int)CENTER_DEAD_ZONE_Y_MIN_PX, (int)round(h * g_center_dead_zone_y_ratio.load()));

            drawMarker(img, Point(cx, cy), Scalar(0, 255, 255), MARKER_CROSS, 32, 2);
            rectangle(img, Rect(cx - dzx, cy - dzy, 2 * dzx, 2 * dzy), Scalar(0, 255, 255), 1);

            double age = now_sec() - det.ts;
            if (det.valid && age < 1.5) {
                int x = (int)round(det.cx - det.w / 2.0f);
                int y = (int)round(det.cy - det.h / 2.0f);
                int bw = (int)round(det.w);
                int bh = (int)round(det.h);

                x = max(0, min(x, w - 1));
                y = max(0, min(y, h - 1));
                bw = max(1, min(bw, w - x));
                bh = max(1, min(bh, h - y));

                Scalar color = det.confirmed ? Scalar(0, 255, 0) : Scalar(0, 255, 255);
                rectangle(img, Rect(x, y, bw, bh), color, 2);
                circle(img, Point((int)det.cx, (int)det.cy), 4, Scalar(0, 0, 255), -1);

                char label[160];
                snprintf(label, sizeof(label),
                         "%s conf=%.2f dx=%.0f dy=%.0f",
                         det.confirmed ? "TRACK" : "CAND",
                         det.conf,
                         det.cx - w / 2.0f,
                         det.cy - h / 2.0f);
                putText(img, label, Point(x, max(20, y - 6)),
                        FONT_HERSHEY_SIMPLEX, 0.55, color, 2, LINE_AA);
            }

            char info1[360];
            snprintf(info1, sizeof(info1),
                     "Frame:%d | Detect:%s PTZ:%s AutoZoom:%s | WSAD/+/- manual | J/K PTZspeed | Z/X zoomIN | ,/. zoomOUT | H help | Q",
                     cam_id,
                     g_detection_enabled.load() ? "ON" : "OFF",
                     ptz_control_enabled.load() ? "ON" : "OFF",
                     auto_zoom_enabled.load() ? "ON" : "OFF");
            putText(img, info1, Point(12, 28), FONT_HERSHEY_SIMPLEX, 0.50,
                    ptz_control_enabled.load() ? Scalar(0, 255, 0) : Scalar(0, 255, 255),
                    2, LINE_AA);

            char infoTune[360];
            snprintf(infoTune, sizeof(infoTune),
                     "PTZmax:%d high:%d manual:%d | zoomIn:%d zoomOut:%d search:%d | dead:%.3f/%.3f period:%.2f",
                     g_ptz_max_speed.load(),
                     g_ptz_max_speed_high_zoom.load(),
                     g_manual_ptz_speed.load(),
                     g_zoom_max_speed.load(),
                     g_zoom_out_max_speed.load(),
                     g_zoom_search_out_speed.load(),
                     g_center_dead_zone_x_ratio.load(),
                     g_center_dead_zone_y_ratio.load(),
                     g_ptz_control_period_sec.load());
            putText(img, infoTune, Point(12, 58), FONT_HERSHEY_SIMPLEX, 0.45,
                    Scalar(255, 255, 0), 2, LINE_AA);

            putText(img, text, Point(12, 88), FONT_HERSHEY_SIMPLEX, 0.50,
                    Scalar(0, 255, 255), 2, LINE_AA);

            imshow("Hikvision RKNN YOLO", img);
        }

        int key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            is_running.store(false);
            break;
        } else if (key == 'c' || key == 'C') {
            bool new_state = !ptz_control_enabled.load();
            ptz_control_enabled.store(new_state);
            cout << "\n[KEY] PTZ auto tracking " << (new_state ? "ON" : "OFF") << endl;
        } else if (key == 'b' || key == 'B') {
            bool new_state = !auto_zoom_enabled.load();
            auto_zoom_enabled.store(new_state);
            cout << "\n[KEY] Auto zoom " << (new_state ? "ON" : "OFF") << endl;
        } else if (key == 'a' || key == 'A') {
            manual_owner_code.store(3);
            manual_pan_user.store(-g_manual_ptz_speed.load());
            manual_tilt_user.store(0);
            manual_ptz_until_ms.store(now_ms() + MANUAL_PTZ_HOLD_MS);
            manual_force_send.store(true);
        } else if (key == 'd' || key == 'D') {
            manual_owner_code.store(3);
            manual_pan_user.store(g_manual_ptz_speed.load());
            manual_tilt_user.store(0);
            manual_ptz_until_ms.store(now_ms() + MANUAL_PTZ_HOLD_MS);
            manual_force_send.store(true);
        } else if (key == 'w' || key == 'W') {
            manual_owner_code.store(3);
            manual_pan_user.store(0);
            manual_tilt_user.store(INVERT_DISPLAY_Y_FOR_CONTROL ? g_manual_ptz_speed.load() : -g_manual_ptz_speed.load());
            manual_ptz_until_ms.store(now_ms() + MANUAL_PTZ_HOLD_MS);
            manual_force_send.store(true);
        } else if (key == 's' || key == 'S') {
            manual_owner_code.store(3);
            manual_pan_user.store(0);
            manual_tilt_user.store(INVERT_DISPLAY_Y_FOR_CONTROL ? -g_manual_ptz_speed.load() : g_manual_ptz_speed.load());
            manual_ptz_until_ms.store(now_ms() + MANUAL_PTZ_HOLD_MS);
            manual_force_send.store(true);
        } else if (key == '+' || key == '=') {
            manual_owner_code.store(3);
            manual_zoom_user.store(g_manual_zoom_speed.load());
            manual_zoom_until_ms.store(now_ms() + MANUAL_ZOOM_HOLD_MS);
            manual_force_send.store(true);
            cout << "\n[KEY] manual ZOOM IN" << endl;
        } else if (key == '-' || key == '_') {
            manual_owner_code.store(3);
            manual_zoom_user.store(-g_manual_zoom_speed.load());
            manual_zoom_until_ms.store(now_ms() + MANUAL_ZOOM_HOLD_MS);
            manual_force_send.store(true);
            cout << "\n[KEY] manual ZOOM OUT" << endl;

        // ===== 在线调参：飞行过程中直接按键调整，不需要重新编译 =====
        } else if (key == 'j' || key == 'J') {
            g_ptz_max_speed.store(clamp_val(g_ptz_max_speed.load() - 5, 20, 110));
            cout << "\n[TUNE] auto PTZ max speed = " << g_ptz_max_speed.load() << endl;
        } else if (key == 'k' || key == 'K') {
            g_ptz_max_speed.store(clamp_val(g_ptz_max_speed.load() + 5, 20, 110));
            cout << "\n[TUNE] auto PTZ max speed = " << g_ptz_max_speed.load() << endl;
        } else if (key == 'u' || key == 'U') {
            g_manual_ptz_speed.store(clamp_val(g_manual_ptz_speed.load() - 5, 15, 110));
            cout << "\n[TUNE] manual WSAD PTZ speed = " << g_manual_ptz_speed.load() << endl;
        } else if (key == 'i' || key == 'I') {
            g_manual_ptz_speed.store(clamp_val(g_manual_ptz_speed.load() + 5, 15, 110));
            cout << "\n[TUNE] manual WSAD PTZ speed = " << g_manual_ptz_speed.load() << endl;
        } else if (key == 'n' || key == 'N') {
            g_ptz_max_speed_high_zoom.store(clamp_val(g_ptz_max_speed_high_zoom.load() - 5, 10, 90));
            cout << "\n[TUNE] high-zoom PTZ max speed = " << g_ptz_max_speed_high_zoom.load() << endl;
        } else if (key == 'm' || key == 'M') {
            g_ptz_max_speed_high_zoom.store(clamp_val(g_ptz_max_speed_high_zoom.load() + 5, 10, 90));
            cout << "\n[TUNE] high-zoom PTZ max speed = " << g_ptz_max_speed_high_zoom.load() << endl;
        } else if (key == 'z' || key == 'Z') {
            g_zoom_max_speed.store(clamp_val(g_zoom_max_speed.load() - 15, 20, 255));
            cout << "\n[TUNE] auto zoom IN max speed = " << g_zoom_max_speed.load() << endl;
        } else if (key == 'x' || key == 'X') {
            g_zoom_max_speed.store(clamp_val(g_zoom_max_speed.load() + 15, 20, 255));
            cout << "\n[TUNE] auto zoom IN max speed = " << g_zoom_max_speed.load() << endl;
        } else if (key == ',' || key == '<') {
            g_zoom_out_max_speed.store(clamp_val(g_zoom_out_max_speed.load() - 25, 30, 255));
            cout << "\n[TUNE] auto zoom OUT max speed = " << g_zoom_out_max_speed.load() << endl;
        } else if (key == '.' || key == '>') {
            g_zoom_out_max_speed.store(clamp_val(g_zoom_out_max_speed.load() + 25, 30, 255));
            cout << "\n[TUNE] auto zoom OUT max speed = " << g_zoom_out_max_speed.load() << endl;
        } else if (key == ';' || key == ':') {
            g_zoom_search_out_speed.store(clamp_val(g_zoom_search_out_speed.load() - 15, 5, 255));
            cout << "\n[TUNE] lost-search zoom-out speed = " << g_zoom_search_out_speed.load() << endl;
        } else if (key == '\'' || key == '"') {
            g_zoom_search_out_speed.store(clamp_val(g_zoom_search_out_speed.load() + 15, 5, 255));
            cout << "\n[TUNE] lost-search zoom-out speed = " << g_zoom_search_out_speed.load() << endl;
        } else if (key == '[' || key == '{') {
            g_ptz_control_period_sec.store(clampf(g_ptz_control_period_sec.load() - 0.02f, 0.06f, 0.30f));
            cout << "\n[TUNE] PTZ control period = " << g_ptz_control_period_sec.load() << " s" << endl;
        } else if (key == ']' || key == '}') {
            g_ptz_control_period_sec.store(clampf(g_ptz_control_period_sec.load() + 0.02f, 0.06f, 0.30f));
            cout << "\n[TUNE] PTZ control period = " << g_ptz_control_period_sec.load() << " s" << endl;
        } else if (key == '9' || key == '(') {
            g_center_dead_zone_x_ratio.store(clampf(g_center_dead_zone_x_ratio.load() - 0.005f, 0.025f, 0.15f));
            g_center_dead_zone_y_ratio.store(clampf(g_center_dead_zone_y_ratio.load() - 0.005f, 0.025f, 0.15f));
            cout << "\n[TUNE] center dead zone x/y = "
                 << g_center_dead_zone_x_ratio.load() << " / "
                 << g_center_dead_zone_y_ratio.load() << endl;
        } else if (key == '0' || key == ')') {
            g_center_dead_zone_x_ratio.store(clampf(g_center_dead_zone_x_ratio.load() + 0.005f, 0.025f, 0.15f));
            g_center_dead_zone_y_ratio.store(clampf(g_center_dead_zone_y_ratio.load() + 0.005f, 0.025f, 0.15f));
            cout << "\n[TUNE] center dead zone x/y = "
                 << g_center_dead_zone_x_ratio.load() << " / "
                 << g_center_dead_zone_y_ratio.load() << endl;
        } else if (key == 'h' || key == 'H') {
            print_runtime_tuning_params();

        } else if (key == ' ') {
            // 急停：空格明确关闭自动控制；WSAD/+/- 不会关闭自动模式。
            ptz_control_enabled.store(false);
            auto_zoom_enabled.store(false);
            manual_zoom_user.store(0);
            manual_zoom_until_ms.store(0);
            manual_pan_user.store(0);
            manual_tilt_user.store(0);
            manual_ptz_until_ms.store(0);
            manual_force_send.store(false);
            manual_auto_resume_block_until_ms.store(now_ms() + (long long)round(MANUAL_AUTO_RESUME_DELAY_SEC * 1000.0f));
            cout << "\n[KEY] emergency stop: PTZ auto OFF, Auto zoom OFF" << endl;
            for (int i = 0; i < MANUAL_STOP_RETRY_COUNT; ++i) {
                send_ptz_continuous(0, 0, 0);
                this_thread::sleep_for(chrono::milliseconds(MANUAL_STOP_RETRY_INTERVAL_MS));
            }
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    destroyAllWindows();
}

static int getenv_nonnegative_int_or_default(const char* name, int default_value);

void ptz_control_thread() {
    cout << "[PTZ] fast continuous control started. Manual uses non-blocking deadman STOP; auto resumes after "
         << MANUAL_AUTO_RESUME_DELAY_SEC << "s." << endl;

    double last_cmd_time = 0.0;
    int last_pan = 0;
    int last_tilt = 0;
    int last_zoom = 0;
    int print_count = 0;

    bool manual_was_active = false;
    bool manual_resume_block_was_active = false;
    int pending_manual_stop_retries = 0;
    double next_manual_stop_retry_time = 0.0;
    PTZState cached_ptz;
    {
        PTZState initial = get_ptz_status();
        if (initial.ok) {
            cached_ptz = initial;
            g_latest_ptz_state_ok.store(true);
            g_latest_ptz_azimuth_unit.store(initial.azimuth);
            g_latest_ptz_elevation_unit.store(initial.elevation);
            g_latest_ptz_absolute_zoom.store(initial.absolute_zoom);
            lock_guard<mutex> lock(g_ptz_home_mtx);
            g_session_initial_ptz = initial;
            g_session_initial_ptz_valid = true;
            g_saved_home_ptz = g_config_home_valid ? g_config_home_ptz : initial;
            g_saved_home_ptz_valid = true;
            cout << "[PTZ_HOME] session initial saved azimuth=" << initial.azimuth
                 << " elevation=" << initial.elevation
                 << " zoom=" << initial.absolute_zoom
                 << (g_config_home_valid ? " configured_home=ON" : " configured_home=OFF")
                 << endl;
        } else {
            cerr << "[PTZ_HOME] initial PTZ status unavailable" << endl;
        }
    }
    double last_ptz_status_time = -10.0;
    double last_calibration_save_time = -10.0;
    int low_conf_sweep_dir = 1;
    double next_low_conf_sweep_flip = 0.0;
    int audio_sweep_tilt_dir = 1;
    int audio_sweep_pan_dir = 1;
    double next_audio_sweep_tilt_flip = 0.0;
    double next_audio_sweep_pan_flip = 0.0;
    double last_audio_guidance_time = -10.0;
    double last_visual_lock_time = -10.0;
    double audio_candidate_start_time = -10.0;
    double audio_candidate_world_deg = 99999.0;
    double accepted_audio_world_deg = 99999.0;
    double accepted_audio_until = -10.0;
    double last_audio_direction_change_time = -10.0;
    double audio_arrival_start_time = -10.0;
    deque<double> audio_offset_bootstrap_history;
    int audio_offset_outlier_count = 0;
    double last_audio_offset_bootstrap_log_time = -10.0;
    double focus_confidence_baseline = -1.0;
    int focus_low_blur_frames = 0;
    int focus_conf_drop_frames = 0;

    // 手动模式采用 deadman 逻辑：
    // 只要按键事件持续刷新，就持续发送手动速度；
    // 一旦按键超时，非阻塞地连续补发 STOP。
    // 注意这里不再 sleep 阻塞控制线程，也不再暂停自动模式，避免“手动后卡一下”。
    auto schedule_manual_stop_retry = [&]() {
        pending_manual_stop_retries = MANUAL_STOP_RETRY_COUNT;
        next_manual_stop_retry_time = 0.0;
    };

    auto send_stop_if_needed = [&]() {
        if (last_pan != 0 || last_tilt != 0 || last_zoom != 0) {
            send_ptz_continuous(0, 0, 0);
            last_pan = 0;
            last_tilt = 0;
            last_zoom = 0;
        }
    };

    while (is_running.load()) {
        double now = now_sec();
        long long ms = now_ms();

        if (g_audio_calibration_reset_requested.exchange(false)) {
            audio_offset_bootstrap_history.clear();
            audio_offset_outlier_count = 0;
            audio_candidate_start_time = -10.0;
            audio_candidate_world_deg = 99999.0;
            accepted_audio_world_deg = 99999.0;
            accepted_audio_until = -10.0;
            audio_arrival_start_time = -10.0;
            last_audio_offset_bootstrap_log_time = -10.0;
            cout << "[FUSION] cleared live audio bootstrap window" << endl;
        }

        int manual_zoom = 0;
        if (ms <= manual_zoom_until_ms.load()) {
            manual_zoom = manual_zoom_user.load();
        } else {
            manual_zoom_user.store(0);
            manual_zoom_until_ms.store(0);
        }

        int manual_pan = 0;
        int manual_tilt = 0;
        bool manual_ptz_active = false;
        if (ms <= manual_ptz_until_ms.load()) {
            manual_pan = manual_pan_user.load();
            manual_tilt = manual_tilt_user.load();
            manual_ptz_active = (manual_pan != 0 || manual_tilt != 0);
        } else {
            manual_pan_user.store(0);
            manual_tilt_user.store(0);
            manual_ptz_until_ms.store(0);
        }

        bool manual_active_now = manual_ptz_active || (manual_zoom != 0);

        // 手动刚结束：只安排 STOP 重试，不阻塞、不暂停自动模式。
        // 这里额外屏蔽自动控制 300ms，先让 STOP 真正落到球机，避免松手后被自动命令覆盖。
        if (!manual_active_now && manual_was_active) {
            schedule_manual_stop_retry();
            manual_was_active = false;
            long long resume_until = ms + (long long)round(MANUAL_AUTO_RESUME_DELAY_SEC * 1000.0f);
            long long previous_until = manual_auto_resume_block_until_ms.load();
            manual_auto_resume_block_until_ms.store(max(previous_until, resume_until));
            if (previous_until < ms) {
                append_ptz_log("vision_deadman", "stop", 0, 0, 0, 0,
                               true, "manual_timeout_stop", 0.0, "scheduled");
            }
        }
        if (manual_active_now) {
            manual_was_active = true;
            pending_manual_stop_retries = 0;
        }

        bool manual_resume_blocked = !manual_active_now && ms < manual_auto_resume_block_until_ms.load();
        if (manual_resume_blocked) {
            manual_resume_block_was_active = true;
        } else if (manual_resume_block_was_active) {
            cout << "[MANUAL] manual override released, auto control resumed." << endl;
            manual_resume_block_was_active = false;
        }

        bool pan_tilt_on = ptz_control_enabled.load() && !manual_resume_blocked;
        bool zoom_on = auto_zoom_enabled.load() && !manual_resume_blocked;

        if (!pan_tilt_on && !zoom_on && manual_zoom == 0 && !manual_ptz_active) {
            if (manual_resume_blocked) {
                FusionTelemetry hold_telemetry;
                hold_telemetry.mode = "MANUAL_HOLD";
                hold_telemetry.fusion_state = "MANUAL_HOLD";
                hold_telemetry.control_owner = "manual";
                hold_telemetry.control_source = "manual_hold";
                hold_telemetry.manual_owner = manual_owner_name(manual_owner_code.load());
                hold_telemetry.auto_track_enabled = ptz_control_enabled.load();
                hold_telemetry.auto_zoom_enabled = auto_zoom_enabled.load();
                hold_telemetry.auto_focus_enabled = g_autofocus_enabled.load();
                hold_telemetry.manual_override_active = true;
                hold_telemetry.manual_hold_remaining_ms =
                    max(0LL, manual_auto_resume_block_until_ms.load() - ms);
                hold_telemetry.ptz_block_reason = "manual_override";
                hold_telemetry.audio_guidance_state = "blocked";
                hold_telemetry.audio_reject_reason = "manual_override";
                hold_telemetry.stale_command_dropped = g_stale_command_dropped_count.load();
                publish_fusion_telemetry(hold_telemetry);
            }
            send_stop_if_needed();
            this_thread::sleep_for(chrono::milliseconds(40));
            continue;
        }

        // 手动 WSAD/+/- 优先使用更短的补发周期；按键刚触发时强制立即发送一次。
        bool force_send_now = manual_force_send.exchange(false);
        float active_cmd_period = g_ptz_control_period_sec.load();
        if (manual_ptz_active) {
            active_cmd_period = MANUAL_PTZ_SEND_PERIOD_SEC;
        }
        if (manual_zoom != 0) {
            active_cmd_period = min(active_cmd_period, MANUAL_ZOOM_SEND_PERIOD_SEC);
        }

        if (!force_send_now && now - last_cmd_time < active_cmd_period) {
            this_thread::sleep_for(chrono::milliseconds(5));
            continue;
        }

        SharedDetection det;
        {
            lock_guard<mutex> lock(latest_det_mtx);
            det = latest_det;
        }

        SharedAudioCue audio;
        {
            lock_guard<mutex> lock(latest_audio_mtx);
            audio = latest_audio;
        }

        if ((now - last_ptz_status_time) >= 0.75) {
            PTZState st = get_ptz_status();
            if (st.ok) {
                cached_ptz = st;
                g_latest_ptz_state_ok.store(true);
                g_latest_ptz_azimuth_unit.store(st.azimuth);
                g_latest_ptz_elevation_unit.store(st.elevation);
                g_latest_ptz_absolute_zoom.store(st.absolute_zoom);
                last_ptz_status_time = now;
            } else if (last_ptz_status_time < 0.0) {
                last_ptz_status_time = now;
            }
        }

        bool det_fresh_for_ptz = det.valid && det.confirmed && (now - det.ts) <= PTZ_TARGET_MAX_AGE_SEC;
        bool det_fresh_for_zoom = det.valid && det.confirmed && (now - det.ts) <= ZOOM_TARGET_MAX_AGE_SEC;
        if (det_fresh_for_ptz) {
            last_visual_lock_time = now;
        }
        bool recently_visual_locked = (now - last_visual_lock_time) <= VISION_AUDIO_SUPPRESS_AFTER_TRACK_SEC;
        bool audio_packet_fresh = g_audio_guidance_enabled.load() &&
                                  audio.detected &&
                                  audio.smooth_doa_deg >= 0.0 &&
                                  audio.smooth_doa_deg < 360.0 &&
                                  audio.stability >= AUDIO_GUIDE_MIN_STABILITY &&
                                  audio.score >= AUDIO_GUIDE_MIN_SCORE_PERCENT &&
                                  audio.snr_db >= AUDIO_GUIDE_MIN_SNR_DB &&
                                  (now - audio.ts) <= AUDIO_CUE_MAX_AGE_SEC;
        bool audio_fresh = false;

        int pan_user = manual_pan;   // WSAD 手动云台优先级最高
        int tilt_user = manual_tilt;
        int zoom_user = manual_zoom;  // 手动 + / - 优先级最高

        int fw = max(1, det.fw);
        int fh = max(1, det.fh);

        float dx = det.cx - fw / 2.0f;
        float dy = det.cy - fh / 2.0f;

        float dead_x = max(CENTER_DEAD_ZONE_X_MIN_PX, fw * g_center_dead_zone_x_ratio.load());
        float dead_y = max(CENTER_DEAD_ZONE_Y_MIN_PX, fh * g_center_dead_zone_y_ratio.load());

        FusionTelemetry telemetry;
        telemetry.mode = det_fresh_for_ptz ? "VISION_TRACK" : (audio_packet_fresh ? "AUDIO_CANDIDATE" : "IDLE");
        telemetry.fusion_state = det_fresh_for_ptz ? "VISUAL_TRACKING" :
                                 (recently_visual_locked ? "LOST_HOLD" :
                                  (audio_packet_fresh ? "WAIT_AUDIO_GUIDE" : "IDLE"));
        telemetry.control_source = manual_ptz_active ? "manual_ptz" : (manual_zoom != 0 ? "manual_zoom" : "none");
        telemetry.search_state = "none";
        telemetry.doa_stability = audio.stability;
        telemetry.auto_track_enabled = ptz_control_enabled.load();
        telemetry.auto_zoom_enabled = auto_zoom_enabled.load();
        telemetry.auto_focus_enabled = g_autofocus_enabled.load();
        telemetry.manual_override_active = manual_ptz_active || manual_zoom != 0 || manual_resume_blocked;
        telemetry.manual_hold_remaining_ms = manual_resume_blocked
            ? max(0LL, manual_auto_resume_block_until_ms.load() - ms)
            : 0LL;
        telemetry.manual_owner = telemetry.manual_override_active
            ? manual_owner_name(manual_owner_code.load())
            : "none";
        telemetry.stale_command_dropped = g_stale_command_dropped_count.load();
        telemetry.emergency_stop_active =
            !ptz_control_enabled.load() && !auto_zoom_enabled.load() && !manual_ptz_active && manual_zoom == 0;
        telemetry.target_age_ms = det.valid ? (long long)round(max(0.0, now - det.ts) * 1000.0) : -1;
        telemetry.target_state = det_fresh_for_ptz ? "TRACK" :
                                 (det.valid && det.confirmed ? "STALE" :
                                  (det.valid ? "CANDIDATE" : "NO_TARGET"));
        telemetry.max_speed_cap = getenv_nonnegative_int_or_default("ANTI_UAV_PTZ_MAX_SPEED_CAP", 0);
        if (!ptz_control_enabled.load()) {
            telemetry.ptz_block_reason = telemetry.emergency_stop_active ? "emergency_stop" : "tracking_disabled";
        } else if (telemetry.manual_override_active) {
            telemetry.ptz_block_reason = "manual_override";
        } else if (!det.valid || !det.confirmed) {
            telemetry.ptz_block_reason = "no_target";
        } else if (!det_fresh_for_ptz) {
            telemetry.ptz_block_reason = "target_too_old";
        } else if (g_ptz_max_speed.load() <= 0) {
            telemetry.ptz_block_reason = "speed_cap_zero";
        } else if (fabs(dx) <= dead_x && fabs(dy) <= dead_y) {
            telemetry.ptz_block_reason = "inside_dead_zone";
        }
        if (cached_ptz.ok) {
            float zr = zoom_to_ratio(cached_ptz.absolute_zoom);
            auto fov = interp_fov(zr);
            telemetry.vertical_sweep_delta_deg = clampf(
                fov.second * AUDIO_VERTICAL_SWEEP_VFOV_FACTOR,
                AUDIO_VERTICAL_SWEEP_MIN_DELTA_DEG,
                AUDIO_VERTICAL_SWEEP_MAX_DELTA_DEG
            );
        }

        auto apply_audio_local_sweep = [&](int base_pan_user,
                                           const string& mode,
                                           int tilt_speed,
                                           int pan_dither_speed) {
            if (now >= next_audio_sweep_tilt_flip) {
                audio_sweep_tilt_dir *= -1;
                next_audio_sweep_tilt_flip = now + AUDIO_LOCAL_SWEEP_FLIP_SEC;
            }
            if (now >= next_audio_sweep_pan_flip) {
                audio_sweep_pan_dir *= -1;
                next_audio_sweep_pan_flip = now + AUDIO_LOCAL_PAN_DITHER_FLIP_SEC;
            }

            pan_user = base_pan_user;
            if (pan_dither_speed > 0) {
                pan_user += audio_sweep_pan_dir * pan_dither_speed;
            }
            tilt_user = audio_sweep_tilt_dir * tilt_speed;
            pan_user = clamp_val(pan_user, -g_ptz_max_speed_high_zoom.load(), g_ptz_max_speed_high_zoom.load());
            tilt_user = clamp_val(tilt_user, -g_ptz_max_speed_high_zoom.load(), g_ptz_max_speed_high_zoom.load());
            telemetry.mode = mode;
            telemetry.fusion_state = "WAIT_VISUAL_RECAPTURE";
            telemetry.control_source = "audio";
            telemetry.search_state = mode;
            telemetry.audio_guidance_state = mode;
            telemetry.audio_reject_reason = "none";
            telemetry.audio_guided = true;
        };

        double camera_azimuth_deg = cached_ptz.ok ? hik_azimuth_to_deg(cached_ptz.azimuth) : 99999.0;
        telemetry.camera_azimuth_deg = camera_azimuth_deg;
        double visual_world_azimuth_deg = 99999.0;
        double audio_world_azimuth_deg = 99999.0;
        double audio_pan_err_deg = 99999.0;
        double audio_visual_err_deg = 99999.0;
        double offset_confidence = 0.0;

        {
            lock_guard<mutex> lock(fusion_mtx);
            offset_confidence = fusion_calib.confidence;
            telemetry.mic_to_camera_offset_deg = fusion_calib.offset_deg;
            telemetry.offset_confidence = fusion_calib.confidence;
            telemetry.offset_confidence_label = fusion_confidence_label(fusion_calib.confidence);
            telemetry.calibration_samples = fusion_calib.total_samples;
        }
        telemetry.audio_calibration_state =
            fusion_calibration_state(offset_confidence, telemetry.calibration_samples);

        if (!g_audio_guidance_enabled.load()) {
            telemetry.audio_guidance_state = "disabled";
            telemetry.audio_reject_reason = "audio_guidance_disabled";
        } else if (telemetry.manual_override_active) {
            telemetry.audio_guidance_state = "blocked";
            telemetry.audio_reject_reason = "manual_override";
        } else if (telemetry.emergency_stop_active) {
            telemetry.audio_guidance_state = "blocked";
            telemetry.audio_reject_reason = "emergency_stop";
        } else if (!audio.detected) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "no_audio_detection";
        } else if (audio.smooth_doa_deg < 0.0 || audio.smooth_doa_deg >= 360.0) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "invalid_doa";
        } else if ((now - audio.ts) > AUDIO_CUE_MAX_AGE_SEC) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "audio_stale";
        } else if (audio.score < AUDIO_GUIDE_MIN_SCORE_PERCENT) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "low_audio_score";
        } else if (audio.snr_db < AUDIO_GUIDE_MIN_SNR_DB) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "low_snr";
        } else if (audio.stability < AUDIO_GUIDE_MIN_STABILITY) {
            telemetry.audio_guidance_state = "waiting_audio";
            telemetry.audio_reject_reason = "doa_unstable";
        } else if (!cached_ptz.ok) {
            telemetry.audio_guidance_state = "waiting_ptz";
            telemetry.audio_reject_reason = "ptz_status_unavailable";
        } else if (det_fresh_for_ptz) {
            telemetry.audio_guidance_state = "visual_tracking_suppressed";
            telemetry.audio_reject_reason = "visual_target_active";
        } else if (recently_visual_locked) {
            telemetry.audio_guidance_state = "lost_hold";
            telemetry.audio_reject_reason = "recent_visual_lock";
        } else if (telemetry.audio_calibration_state == "uncalibrated") {
            telemetry.audio_guidance_state = "waiting_calibration";
            telemetry.audio_reject_reason = "need_audio_visual_calibration";
        } else {
            telemetry.audio_guidance_state = "audio_candidate";
            telemetry.audio_reject_reason = "none";
        }

        if (audio_packet_fresh && cached_ptz.ok) {
            double packet_audio_world_deg = wrap_deg360(audio.smooth_doa_deg + telemetry.mic_to_camera_offset_deg);
            bool candidate_valid = audio_candidate_world_deg >= 0.0 && audio_candidate_world_deg < 360.0;
            double candidate_drift = candidate_valid
                                         ? fabs(circular_error_deg(packet_audio_world_deg, audio_candidate_world_deg))
                                         : 99999.0;
            if (!candidate_valid || candidate_drift > AUDIO_GUIDE_CONFIRM_DRIFT_DEG) {
                audio_candidate_world_deg = packet_audio_world_deg;
                audio_candidate_start_time = now;
                audio_arrival_start_time = -10.0;
            } else {
                double correction = circular_error_deg(packet_audio_world_deg, audio_candidate_world_deg);
                audio_candidate_world_deg = wrap_deg360(audio_candidate_world_deg + 0.35 * correction);
            }

            bool accepted_valid = accepted_audio_world_deg >= 0.0 &&
                                  accepted_audio_world_deg < 360.0 &&
                                  now <= accepted_audio_until;
            bool jump_allowed = !accepted_valid ||
                                fabs(circular_error_deg(audio_candidate_world_deg, accepted_audio_world_deg)) <= AUDIO_GUIDE_REJECT_JUMP_DEG ||
                                now > accepted_audio_until;
            bool candidate_confirmed = (now - audio_candidate_start_time) >= AUDIO_GUIDE_CONFIRM_SEC;
            if (candidate_confirmed && jump_allowed) {
                double accepted_delta = accepted_valid
                                            ? fabs(circular_error_deg(audio_candidate_world_deg, accepted_audio_world_deg))
                                            : 99999.0;
                accepted_audio_world_deg = audio_candidate_world_deg;
                accepted_audio_until = now + AUDIO_GUIDE_ACCEPT_HOLD_SEC;
                last_audio_guidance_time = now;
                if (!accepted_valid || accepted_delta > 2.0) {
                    last_audio_direction_change_time = now;
                }
            }
        }

        if (accepted_audio_world_deg >= 0.0 &&
            accepted_audio_world_deg < 360.0 &&
            now <= accepted_audio_until) {
            audio_fresh = true;
            audio_world_azimuth_deg = accepted_audio_world_deg;
            telemetry.audio_world_azimuth_deg = audio_world_azimuth_deg;
            if (cached_ptz.ok) {
                audio_pan_err_deg = circular_error_deg(audio_world_azimuth_deg, camera_azimuth_deg);
                telemetry.audio_pan_err_deg = audio_pan_err_deg;
            }
        }

        if (det_fresh_for_ptz && cached_ptz.ok) {
            visual_world_azimuth_deg = visual_target_azimuth_deg(det, cached_ptz);
            telemetry.visual_azimuth_deg = visual_world_azimuth_deg;
        }

        bool ptz_slow_for_calib = abs(last_pan) <= 55 && abs(last_tilt) <= 35;
        bool sync_for_calib = det_fresh_for_ptz && audio_fresh && cached_ptz.ok &&
                              fabs(det.ts - audio.ts) <= AUDIO_VISION_SYNC_SEC &&
                              ptz_slow_for_calib &&
                              audio.stability >= AUDIO_HIGH_CONF_MIN_STABILITY &&
                              audio.score >= AUDIO_GUIDE_MIN_SCORE_PERCENT &&
                              audio.snr_db >= AUDIO_GUIDE_MIN_SNR_DB;
        if (sync_for_calib) {
            double observed_offset = valid_deg360(audio.smooth_doa_deg)
                                         ? wrap_deg360(visual_world_azimuth_deg - audio.smooth_doa_deg)
                                         : 99999.0;
            double err = circular_error_deg(visual_world_azimuth_deg, audio_world_azimuth_deg);
            double abs_err = fabs(err);
            audio_visual_err_deg = abs_err;
            telemetry.audio_visual_angle_error_deg = abs_err;

            bool should_save = false;
            bool bootstrapped_offset = false;
            {
                lock_guard<mutex> lock(fusion_mtx);
                fusion_calib.total_samples++;
                if (valid_deg360(observed_offset)) {
                    audio_offset_bootstrap_history.push_back(observed_offset);
                    while ((int)audio_offset_bootstrap_history.size() > AUDIO_OFFSET_BOOTSTRAP_WINDOW) {
                        audio_offset_bootstrap_history.pop_front();
                    }
                }

                if (fusion_calib.confidence < AUDIO_OFFSET_BOOTSTRAP_CONF &&
                    (int)audio_offset_bootstrap_history.size() >= AUDIO_OFFSET_BOOTSTRAP_MIN_SAMPLES) {
                    vector<double> offset_samples(
                        audio_offset_bootstrap_history.begin(),
                        audio_offset_bootstrap_history.end()
                    );
                    double bootstrap_offset = circular_mean_deg(offset_samples);
                    double bootstrap_std = circular_std_deg(offset_samples);
                    if (valid_deg360(bootstrap_offset) &&
                        bootstrap_std <= AUDIO_OFFSET_BOOTSTRAP_MAX_STD_DEG) {
                        fusion_calib.offset_deg = bootstrap_offset;
                        fusion_calib.confidence = max(fusion_calib.confidence, (double)AUDIO_OFFSET_BOOTSTRAP_CONF);
                        fusion_calib.stable_samples += (int)offset_samples.size();
                        audio_offset_outlier_count = 0;
                        bootstrapped_offset = true;
                        if (now - last_audio_offset_bootstrap_log_time >= 1.0) {
                            cout << "\n[FUSION] bootstrapped live audio offset: "
                                 << bootstrap_offset << " deg, std=" << bootstrap_std
                                 << ", samples=" << offset_samples.size() << endl;
                            last_audio_offset_bootstrap_log_time = now;
                        }
                    }
                }

                if (valid_deg360(audio.smooth_doa_deg)) {
                    audio_world_azimuth_deg = wrap_deg360(audio.smooth_doa_deg + fusion_calib.offset_deg);
                    telemetry.audio_world_azimuth_deg = audio_world_azimuth_deg;
                    if (cached_ptz.ok) {
                        audio_pan_err_deg = circular_error_deg(audio_world_azimuth_deg, camera_azimuth_deg);
                        telemetry.audio_pan_err_deg = audio_pan_err_deg;
                    }
                    accepted_audio_world_deg = audio_world_azimuth_deg;
                    accepted_audio_until = max(accepted_audio_until, now + AUDIO_GUIDE_ACCEPT_HOLD_SEC);
                    err = circular_error_deg(visual_world_azimuth_deg, audio_world_azimuth_deg);
                    abs_err = fabs(err);
                    audio_visual_err_deg = abs_err;
                    telemetry.audio_visual_angle_error_deg = abs_err;
                }

                fusion_calib.last_error_deg = abs_err;
                fusion_calib.last_visual_azimuth_deg = visual_world_azimuth_deg;
                fusion_calib.last_audio_world_deg = audio_world_azimuth_deg;

                if (!bootstrapped_offset &&
                    fusion_calib.confidence < AUDIO_OFFSET_BOOTSTRAP_CONF &&
                    abs_err <= 25.0) {
                    double alpha = fusion_calib.confidence < 0.50 ? 0.08 : 0.015;
                    fusion_calib.offset_deg = wrap_deg360(fusion_calib.offset_deg + alpha * err);
                    audio_offset_outlier_count = 0;
                    if (abs_err <= 12.0) {
                        fusion_calib.stable_samples++;
                        fusion_calib.confidence = min(1.0, fusion_calib.confidence + 0.055);
                    } else if (abs_err <= 25.0) {
                        fusion_calib.confidence = min(0.85, fusion_calib.confidence + 0.015);
                    }
                } else if (!bootstrapped_offset && abs_err <= 25.0) {
                    audio_offset_outlier_count = 0;
                    if (abs_err <= 12.0) {
                        fusion_calib.stable_samples++;
                        fusion_calib.confidence = min(1.0, fusion_calib.confidence + 0.035);
                    } else {
                        fusion_calib.confidence = min(0.85, max(fusion_calib.confidence, (double)AUDIO_OFFSET_BOOTSTRAP_CONF));
                    }
                } else if (!bootstrapped_offset && abs_err > 25.0) {
                    audio_offset_outlier_count++;
                    if (fusion_calib.confidence < AUDIO_OFFSET_BOOTSTRAP_CONF) {
                        fusion_calib.confidence = max(0.0, fusion_calib.confidence - 0.005);
                    } else if (audio_offset_outlier_count >= AUDIO_OFFSET_OUTLIER_GRACE_SAMPLES) {
                        fusion_calib.confidence = max((double)AUDIO_OFFSET_BOOTSTRAP_CONF,
                                                      fusion_calib.confidence - 0.010);
                        audio_offset_outlier_count = AUDIO_OFFSET_OUTLIER_GRACE_SAMPLES / 2;
                    }
                }

                telemetry.mic_to_camera_offset_deg = fusion_calib.offset_deg;
                telemetry.offset_confidence = fusion_calib.confidence;
                telemetry.offset_confidence_label = fusion_confidence_label(fusion_calib.confidence);
                telemetry.calibration_samples = fusion_calib.total_samples;
                offset_confidence = fusion_calib.confidence;
                should_save = fusion_calib.confidence >= AUDIO_OFFSET_HIGH_CONF &&
                              (now - last_calibration_save_time) >= 5.0;
            }
            if (should_save) {
                save_fusion_calibration();
                last_calibration_save_time = now;
            }
            update_audio_runtime_learning(
                abs_err,
                telemetry.mic_to_camera_offset_deg,
                telemetry.offset_confidence
            );
        }

        // C 开启后：根据目标中心偏差做快速 continuous 粗闭环。
        // 如果正在 WSAD 手动控制，则暂停自动 pan/tilt；松开后自动恢复。
        if (!manual_ptz_active && pan_tilt_on && det_fresh_for_ptz) {
            if (fabs(dx) > dead_x) {
                float usable = max(1.0f, fw * 0.5f - dead_x);
                float e = clampf((fabs(dx) - dead_x) / usable, 0.0f, 1.0f);
                int speed = (int)round(g_ptz_min_speed.load() + pow(e, g_ptz_speed_curve.load()) * (g_ptz_max_speed.load() - g_ptz_min_speed.load()));
                speed = (int)round(speed * g_tracking_pan_gain.load());
                speed = clamp_val(speed, g_ptz_min_speed.load(), g_ptz_max_speed.load());
                pan_user = (dx > 0) ? speed : -speed;
            }

            float dy_for_control = INVERT_DISPLAY_Y_FOR_CONTROL ? -dy : dy;
            if (fabs(dy_for_control) > dead_y) {
                float usable = max(1.0f, fh * 0.5f - dead_y);
                float e = clampf((fabs(dy_for_control) - dead_y) / usable, 0.0f, 1.0f);
                int speed = (int)round(g_ptz_min_speed.load() + pow(e, g_ptz_speed_curve.load()) * (g_ptz_max_speed.load() - g_ptz_min_speed.load()));
                speed = (int)round(speed * g_tracking_tilt_gain.load());
                speed = clamp_val(speed, g_ptz_min_speed.load(), g_ptz_max_speed.load());
                tilt_user = (dy_for_control > 0) ? speed : -speed;
            }

            // 目标已经比较大，说明倍率较高或目标很近，自动降低 pan/tilt 速度，减少过冲。
            float box_side_ratio = max(det.w / (float)fw, det.h / (float)fh);
            if (box_side_ratio > 0.18f) {
                pan_user = max(-g_ptz_max_speed_high_zoom.load(), min(g_ptz_max_speed_high_zoom.load(), pan_user));
                tilt_user = max(-g_ptz_max_speed_high_zoom.load(), min(g_ptz_max_speed_high_zoom.load(), tilt_user));
                telemetry.high_zoom_limited = true;
            }
            telemetry.mode = "VISION_TRACK";
            telemetry.fusion_state = "VISUAL_TRACKING";
            telemetry.control_source = "vision";
            telemetry.search_state = "none";
        } else if (!manual_ptz_active && pan_tilt_on && audio_fresh && cached_ptz.ok && !recently_visual_locked) {
            const bool high_conf_audio_offset =
                offset_confidence >= AUDIO_OFFSET_HIGH_CONF &&
                telemetry.calibration_samples >= AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES &&
                audio.stability >= AUDIO_HIGH_CONF_MIN_STABILITY &&
                audio_offset_outlier_count < AUDIO_OFFSET_OUTLIER_GRACE_SAMPLES;
            const bool bootstrap_audio_offset =
                offset_confidence >= AUDIO_OFFSET_GUIDE_CONF &&
                telemetry.calibration_samples >= AUDIO_OFFSET_BOOTSTRAP_MIN_SAMPLES &&
                audio.stability >= AUDIO_GUIDE_MIN_STABILITY &&
                audio_offset_outlier_count < AUDIO_OFFSET_OUTLIER_GRACE_SAMPLES;
            const double dead_zone_deg = high_conf_audio_offset
                                             ? AUDIO_GUIDE_DEAD_ZONE_DEG
                                             : (bootstrap_audio_offset ? AUDIO_BOOTSTRAP_DEAD_ZONE_DEG
                                                                       : AUDIO_LOW_CONF_DEAD_ZONE_DEG);
            const int min_speed = high_conf_audio_offset
                                      ? AUDIO_HIGH_CONF_MIN_PAN_SPEED
                                      : (bootstrap_audio_offset ? AUDIO_BOOTSTRAP_MIN_PAN_SPEED
                                                                : AUDIO_LOW_CONF_MIN_PAN_SPEED);
            const int max_speed = high_conf_audio_offset
                                      ? AUDIO_HIGH_CONF_MAX_PAN_SPEED
                                      : (bootstrap_audio_offset ? AUDIO_BOOTSTRAP_MAX_PAN_SPEED
                                                                : AUDIO_LOW_CONF_MAX_PAN_SPEED);
            const string mode_suffix = high_conf_audio_offset ? "" : (bootstrap_audio_offset ? "_BOOTSTRAP" : "_LOWCONF");
            const bool initial_low_conf_small_turn =
                offset_confidence < AUDIO_OFFSET_BOOTSTRAP_CONF &&
                telemetry.calibration_samples < AUDIO_OFFSET_BOOTSTRAP_MIN_SAMPLES &&
                audio_pan_err_deg < 99998.0 &&
                fabs(audio_pan_err_deg) <= AUDIO_LOW_CONF_MAX_TURN_ERROR_DEG;
            const bool low_conf_turn_allowed =
                high_conf_audio_offset ||
                bootstrap_audio_offset;

            telemetry.audio_guided = true;
            if (!low_conf_turn_allowed) {
                pan_user = 0;
                tilt_user = 0;
                accepted_audio_until = now;
                last_audio_guidance_time = -10.0;
                telemetry.mode = "AUDIO_WAIT_CALIBRATION";
                telemetry.fusion_state = "WAIT_AUDIO_GUIDE";
                telemetry.control_source = "none";
                telemetry.search_state = initial_low_conf_small_turn
                                             ? "audio_small_turn_rejected_until_bootstrap"
                                             : "need_visual_audio_calibration";
                telemetry.audio_guidance_state = "waiting_calibration";
                telemetry.audio_reject_reason = telemetry.search_state;
                telemetry.audio_guided = false;
            } else if (audio_pan_err_deg < 99998.0 && fabs(audio_pan_err_deg) > dead_zone_deg) {
                double e = min(1.0, max(0.0, (fabs(audio_pan_err_deg) - dead_zone_deg) / 90.0));
                int speed = (int)round(min_speed + e * (max_speed - min_speed));
                pan_user = (audio_pan_err_deg < 0.0) ? speed : -speed;
                tilt_user = 0;
                audio_arrival_start_time = -10.0;
                telemetry.mode = string("AUDIO_TURN") + mode_suffix;
                telemetry.fusion_state = "AUDIO_GUIDING";
                telemetry.control_source = "audio";
                telemetry.search_state = "horizontal_turn";
                telemetry.audio_guidance_state = "guiding_turn";
                telemetry.audio_reject_reason = "none";
            } else {
                if (audio_arrival_start_time < 0.0 || audio_arrival_start_time < last_audio_direction_change_time - 0.05) {
                    audio_arrival_start_time = now;
                    next_audio_sweep_tilt_flip = now + AUDIO_GUIDE_SETTLE_SEC;
                    next_audio_sweep_pan_flip = now + AUDIO_GUIDE_SETTLE_SEC;
                }

                if ((now - audio_arrival_start_time) < AUDIO_GUIDE_SETTLE_SEC) {
                    pan_user = 0;
                    tilt_user = 0;
                    telemetry.mode = string("AUDIO_SETTLE") + mode_suffix;
                    telemetry.fusion_state = "WAIT_VISUAL_RECAPTURE";
                    telemetry.control_source = "audio";
                    telemetry.search_state = "settle_before_vertical_sweep";
                    telemetry.audio_guidance_state = "settle_before_search";
                    telemetry.audio_reject_reason = "none";
                } else {
                    double local_search_elapsed = now - audio_arrival_start_time - AUDIO_GUIDE_SETTLE_SEC;
                    double local_search_limit = high_conf_audio_offset
                                                    ? AUDIO_LOCAL_SEARCH_MAX_SEC
                                                    : AUDIO_LOW_CONF_LOCAL_SEARCH_MAX_SEC;
                    if (local_search_elapsed <= local_search_limit) {
                        apply_audio_local_sweep(
                            0,
                            string("AUDIO_LOCAL_SEARCH") + mode_suffix,
                            high_conf_audio_offset ? AUDIO_LOCAL_SWEEP_TILT_SPEED : AUDIO_LOW_CONF_LOCAL_SWEEP_TILT_SPEED,
                            high_conf_audio_offset ? AUDIO_LOCAL_SWEEP_PAN_SPEED : 0
                        );
                    } else {
                        pan_user = 0;
                        tilt_user = 0;
                        accepted_audio_until = now;
                        last_audio_guidance_time = -10.0;
                        telemetry.mode = string("AUDIO_SEARCH_HOLD") + mode_suffix;
                        telemetry.fusion_state = "WAIT_AUDIO_GUIDE";
                        telemetry.control_source = "none";
                        telemetry.search_state = "audio_wait_next_stable_cue";
                        telemetry.audio_guidance_state = "cooldown_wait_next_cue";
                        telemetry.audio_reject_reason = "local_search_timeout";
                        telemetry.audio_guided = false;
                    }
                }
            }
        } else if (!manual_ptz_active && pan_tilt_on && !det_fresh_for_ptz &&
                   !recently_visual_locked &&
                   offset_confidence >= AUDIO_OFFSET_HIGH_CONF &&
                   telemetry.calibration_samples >= AUDIO_OFFSET_HIGH_CONF_MIN_SAMPLES &&
                   audio.stability >= AUDIO_HIGH_CONF_MIN_STABILITY &&
                   (now - last_audio_guidance_time) <= AUDIO_RECOVERY_SWEEP_SEC) {
            apply_audio_local_sweep(
                0,
                "AUDIO_RECOVERY_SCAN",
                AUDIO_LOCAL_SWEEP_TILT_SPEED / 2,
                AUDIO_LOCAL_SWEEP_PAN_SPEED
            );
        }

        // B 开启后：自动变焦，让目标保持大概固定大小。
        // 用 EMA + 回差 + 冷却，减少来回拉扯；手动 +/- 永远优先。
        static float zoom_side_ema = 0.0f;
        static int last_auto_zoom_dir = 0;
        static double last_auto_zoom_cmd_time = 0.0;
        static double last_auto_zoom_flip_time = 0.0;
        static int held_auto_zoom_user = 0;
        static double held_auto_zoom_until = 0.0;
        static double last_confirmed_seen_time = 0.0;
        static double fast_zoom_out_enter_time = -10.0;
        static double last_zoom_motion_time = -10.0;
        static double last_focus_after_zoom_request_time = -10.0;
        static bool focus_after_zoom_pending = false;

        float box_side_ratio = 0.0f;
        float zoom_error = 0.0f;
        const double lost_duration = last_confirmed_seen_time > 0.0
                                         ? max(0.0, now - last_confirmed_seen_time)
                                         : 0.0;

        if (det_fresh_for_zoom) {
            last_confirmed_seen_time = now;
            fast_zoom_out_enter_time = -10.0;
            box_side_ratio = max(det.w / (float)fw, det.h / (float)fh);
            if (zoom_side_ema <= 0.001f) zoom_side_ema = box_side_ratio;
            else zoom_side_ema = (1.0f - ZOOM_EMA_ALPHA) * zoom_side_ema + ZOOM_EMA_ALPHA * box_side_ratio;
        } else {
            box_side_ratio = zoom_side_ema;
        }
        telemetry.target_max_side = box_side_ratio;
        telemetry.target_max_side_ema = zoom_side_ema;
        telemetry.target_lost = !det_fresh_for_zoom;
        telemetry.lost_duration_sec = lost_duration;
        telemetry.search_zoom_level = g_lost_search_zoom_ratio.load();

        if (manual_zoom == 0 && zoom_on) {
            bool allow_zoom_cmd = (now - last_auto_zoom_cmd_time) >= g_zoom_cmd_period_sec.load();
            int desired_zoom_dir = 0;

            if (det_fresh_for_zoom) {
                // 安全策略：不要追求把无人机放得很大，只保持在中等可识别范围。
                // 1) 10%~25% 是稳定识别区间，12%~22% 内完全保持，不追固定目标比例。
                // 2) 目标离中心太远时禁止继续放大，先让 PTZ 把目标拉回中心。
                float abs_dx_ratio_for_zoom = fabs(det.cx - fw * 0.5f) / max(1.0f, (float)fw);
                float abs_dy_ratio_for_zoom = fabs(det.cy - fh * 0.5f) / max(1.0f, (float)fh);
                bool near_edge_for_zoom = max(abs_dx_ratio_for_zoom, abs_dy_ratio_for_zoom) > ZOOM_IN_NEAR_EDGE_BLOCK_RATIO;
                telemetry.target_near_edge = near_edge_for_zoom;

                const float target_min_ratio = g_zoom_target_min_ratio.load();
                const float ideal_low_ratio = g_zoom_target_ideal_low_ratio.load();
                const float ideal_high_ratio = g_zoom_target_ideal_high_ratio.load();
                const float target_max_ratio = g_zoom_target_max_ratio.load();
                telemetry.target_stable =
                    zoom_side_ema >= target_min_ratio && zoom_side_ema <= target_max_ratio;
                zoom_error = 0.0f;

                // 稳定区间变焦逻辑：不要追一个精确目标大小，而是维持在一段可识别范围内。
                // 小于 10% 才分步放大；大于 25% 才分步缩小；10%~25% 内完全不变焦。
                if (near_edge_for_zoom) {
                    // 目标偏离中心时禁止放大；如果目标已经比较大，则缩小扩大视野。
                    if (zoom_side_ema > ideal_high_ratio) desired_zoom_dir = -1;
                    else desired_zoom_dir = 0;
                } else if (zoom_side_ema < target_min_ratio) {
                    desired_zoom_dir = 1;
                } else if (zoom_side_ema > target_max_ratio) {
                    desired_zoom_dir = -1;
                } else {
                    desired_zoom_dir = 0;
                }

                if (desired_zoom_dir == 0) {
                    // 进入回差区或安全上限区，立即停止自动变焦保持，避免继续放大导致丢识别。
                    held_auto_zoom_user = 0;
                    held_auto_zoom_until = 0.0;
                    telemetry.zoom_state = telemetry.target_stable ? "TRACK_STABLE" : "ZOOM_HOLD";
                    telemetry.zoom_action = "hold";
                    telemetry.zoom_reason = near_edge_for_zoom ? "target_near_edge" : "inside_10_25_range";
                } else if (allow_zoom_cmd) {
                    bool flipping = (last_auto_zoom_dir != 0 && desired_zoom_dir != last_auto_zoom_dir);
                    bool flip_allowed = !flipping || ((now - last_auto_zoom_flip_time) >= g_zoom_flip_cooldown_sec.load());

                    if (flip_allowed) {
                        float e = 0.0f;
                        int zspeed;
                        if (desired_zoom_dir < 0) {
                            e = clampf((zoom_side_ema - target_max_ratio) / max(0.001f, 0.45f - target_max_ratio), 0.0f, 1.0f);
                            zspeed = (int)round(g_zoom_out_min_speed.load() + e * (g_zoom_step_out_speed.load() - g_zoom_out_min_speed.load()));
                        } else {
                            e = clampf((target_min_ratio - zoom_side_ema) / max(0.001f, target_min_ratio), 0.0f, 1.0f);
                            zspeed = (int)round(g_zoom_min_speed.load() + e * (g_zoom_step_in_speed.load() - g_zoom_min_speed.load()));
                        }
                        zoom_user = desired_zoom_dir * zspeed;
                        held_auto_zoom_user = zoom_user;
                        held_auto_zoom_until = now + AUTO_ZOOM_HOLD_SEC;
                        telemetry.zoom_state = "ZOOMING";
                        telemetry.zoom_action = desired_zoom_dir > 0 ? "zoom_in" : "zoom_out";
                        telemetry.zoom_reason = desired_zoom_dir > 0 ? "max_side_below_10_percent" :
                                                 (near_edge_for_zoom ? "near_edge_shrink" : "max_side_above_25_percent");

                        if (flipping) last_auto_zoom_flip_time = now;
                        last_auto_zoom_dir = desired_zoom_dir;
                        last_auto_zoom_cmd_time = now;
                    } else {
                        telemetry.zoom_state = "ZOOM_HOLD";
                        telemetry.zoom_action = "hold";
                        telemetry.zoom_reason = "direction_flip_cooldown";
                    }
                } else if (now < held_auto_zoom_until) {
                    // 两次自动变焦决策之间继续保持上一条 zoom 命令。
                    // 但如果上一条是放大且当前框已接近上限，立刻停止，避免继续放大导致丢识别。
                    if ((held_auto_zoom_user > 0 && zoom_side_ema >= target_min_ratio) ||
                        (held_auto_zoom_user < 0 && zoom_side_ema <= target_max_ratio)) {
                        zoom_user = 0;
                        held_auto_zoom_user = 0;
                        held_auto_zoom_until = 0.0;
                        telemetry.zoom_state = "ZOOM_HOLD";
                        telemetry.zoom_action = "hold";
                        telemetry.zoom_reason = "entered_10_25_range";
                    } else {
                        zoom_user = held_auto_zoom_user;
                        telemetry.zoom_state = "ZOOMING";
                        telemetry.zoom_action = zoom_user > 0 ? "zoom_in_hold" : "zoom_out_hold";
                        telemetry.zoom_reason = "command_hold";
                    }
                } else {
                    telemetry.zoom_state = "ZOOM_HOLD";
                    telemetry.zoom_action = "hold";
                    telemetry.zoom_reason = "zoom_cooldown";
                }
            } else {
                // 无稳定视觉目标时主动缩小视野：丢目标、声学搜索、恢复扫描都应尽快扩视角。
                // 如果已经是 1x，负 zoom 指令通常不会继续缩小，不影响。
                bool lost_after_confirmed = last_confirmed_seen_time > 0.0 &&
                                            (now - last_confirmed_seen_time) >= LOST_ZOOM_OUT_AFTER_SEC;
                bool audio_searching = audio_fresh ||
                                       (now - last_audio_guidance_time) <= AUDIO_RECOVERY_SWEEP_SEC ||
                                       telemetry.audio_guided;
                bool need_search_zoom_out = lost_after_confirmed || audio_searching;
                bool reached_search_zoom = cached_ptz.ok &&
                                           zoom_to_ratio(cached_ptz.absolute_zoom) <= g_lost_search_zoom_ratio.load();
                if (need_search_zoom_out && fast_zoom_out_enter_time < 0.0) {
                    fast_zoom_out_enter_time = now;
                }
                bool fast_zoom_out_timeout = fast_zoom_out_enter_time > 0.0 &&
                                             (now - fast_zoom_out_enter_time) >= FAST_ZOOM_OUT_MAX_DURATION_SEC;
                if (need_search_zoom_out && !reached_search_zoom && !fast_zoom_out_timeout && allow_zoom_cmd) {
                    zoom_user = -g_zoom_search_out_speed.load();
                    held_auto_zoom_user = zoom_user;
                    held_auto_zoom_until = now + max(AUTO_ZOOM_HOLD_SEC, g_zoom_cmd_period_sec.load());
                    last_auto_zoom_dir = -1;
                    last_auto_zoom_cmd_time = now;
                    telemetry.zoom_state = "FAST_ZOOM_OUT_SEARCH";
                    telemetry.zoom_action = "zoom_out_fast";
                    telemetry.zoom_reason = lost_after_confirmed ? "target_lost" : "audio_search_expand_view";
                } else if (need_search_zoom_out && !reached_search_zoom && !fast_zoom_out_timeout && now < held_auto_zoom_until) {
                    zoom_user = held_auto_zoom_user;
                    telemetry.zoom_state = "FAST_ZOOM_OUT_SEARCH";
                    telemetry.zoom_action = "zoom_out_fast_hold";
                    telemetry.zoom_reason = "command_hold";
                } else if (reached_search_zoom || fast_zoom_out_timeout) {
                    zoom_user = 0;
                    held_auto_zoom_user = 0;
                    held_auto_zoom_until = 0.0;
                    telemetry.zoom_state = "WAIT_VISUAL_OR_AUDIO";
                    telemetry.zoom_action = "hold";
                    telemetry.zoom_reason = reached_search_zoom ? "reached_search_zoom" : "fast_zoom_out_timeout";
                } else {
                    telemetry.zoom_state = recently_visual_locked ? "LOST_HOLD" : "WAIT_VISUAL_OR_AUDIO";
                    telemetry.zoom_action = "hold";
                    telemetry.zoom_reason = recently_visual_locked ? "lost_hold" : "no_stable_target";
                }
            }
        }
        if (zoom_user != 0) {
            last_zoom_motion_time = now;
            focus_after_zoom_pending = true;
        }

        if (!det_fresh_for_ptz) {
            const bool manual_control_state =
                telemetry.control_source == "manual_ptz" ||
                telemetry.control_source == "manual_zoom";
            const bool lost_after_confirmed_for_state =
                last_confirmed_seen_time > 0.0 &&
                (now - last_confirmed_seen_time) >= LOST_ZOOM_OUT_AFTER_SEC;
            const bool reached_search_zoom_for_state =
                cached_ptz.ok &&
                zoom_to_ratio(cached_ptz.absolute_zoom) <= g_lost_search_zoom_ratio.load();

            if (!manual_control_state && telemetry.audio_guided && telemetry.control_source == "audio") {
                if (telemetry.mode.find("AUDIO_TURN") != string::npos) {
                    telemetry.fusion_state = "AUDIO_GUIDING";
                } else {
                    telemetry.fusion_state = "WAIT_VISUAL_RECAPTURE";
                }
            } else if (!manual_control_state &&
                       !lost_after_confirmed_for_state &&
                       (recently_visual_locked ||
                        (last_confirmed_seen_time > 0.0 &&
                         (now - last_confirmed_seen_time) < LOST_ZOOM_OUT_AFTER_SEC))) {
                telemetry.fusion_state = "LOST_HOLD";
                if (telemetry.control_source == "none") {
                    telemetry.control_source = "visual_hold";
                }
                telemetry.search_state = "lost_hold";
            } else if (!manual_control_state &&
                       lost_after_confirmed_for_state &&
                       !reached_search_zoom_for_state &&
                       (zoom_user < 0 || (held_auto_zoom_user < 0 && now < held_auto_zoom_until))) {
                telemetry.fusion_state = "FAST_ZOOM_OUT_SEARCH";
                if (telemetry.control_source == "none") {
                    telemetry.control_source = "lost_zoom";
                }
                telemetry.search_state = "fast_zoom_out_to_search";
            } else if (!manual_control_state &&
                       (audio_packet_fresh || audio_fresh ||
                        (lost_after_confirmed_for_state && reached_search_zoom_for_state))) {
                telemetry.fusion_state = reached_search_zoom_for_state ? "WAIT_VISUAL_OR_AUDIO" : "WAIT_AUDIO_GUIDE";
                if (telemetry.search_state == "none") {
                    telemetry.search_state = audio_packet_fresh || audio_fresh
                                                 ? "audio_candidate"
                                                 : "search_zoom_wait_audio";
                }
            }
        }

        if (telemetry.fusion_state == "LOST_HOLD" && telemetry.control_source == "none") {
            telemetry.control_source = "visual_hold";
            telemetry.search_state = "lost_hold";
        }

        auto smooth_auto_axis = [](int desired, int previous, int max_step) {
            if (desired == 0) return 0;
            if (previous != 0 && ((desired > 0) != (previous > 0))) {
                return 0;
            }
            int delta = desired - previous;
            if (delta > max_step) return previous + max_step;
            if (delta < -max_step) return previous - max_step;
            return desired;
        };
        if (!manual_ptz_active) {
            int max_axis_step = telemetry.audio_guided ? AUDIO_PTZ_MAX_SPEED_STEP : AUTO_PTZ_MAX_SPEED_STEP;
            pan_user = smooth_auto_axis(pan_user, last_pan, max_axis_step);
            tilt_user = smooth_auto_axis(tilt_user, last_tilt, max_axis_step);
        }

        telemetry.pan_speed = pan_user;
        telemetry.tilt_speed = tilt_user;
        telemetry.actual_pan_speed = pan_user;
        telemetry.actual_tilt_speed = tilt_user;
        if (telemetry.control_source == "audio" && !det_fresh_for_ptz) {
            telemetry.ptz_block_reason = "audio_guidance_active";
        } else if ((pan_user != 0 || tilt_user != 0) &&
                   (telemetry.control_source == "vision" || telemetry.control_source == "audio")) {
            telemetry.ptz_block_reason = "none";
        }
        if (cached_ptz.ok) {
            double zr = zoom_to_ratio(cached_ptz.absolute_zoom);
            if ((zoom_user > 0 && zr >= g_zoom_max_ratio.load()) ||
                (zoom_user < 0 && zr <= g_zoom_min_ratio.load())) {
                zoom_user = 0;
                held_auto_zoom_user = 0;
                held_auto_zoom_until = 0.0;
            }
        }
        long long now_mono = now_ms();
        if (det_fresh_for_ptz) {
            if (focus_confidence_baseline < 0.0) {
                focus_confidence_baseline = det.conf;
            }
            double conf_drop = focus_confidence_baseline - det.conf;
            if (det.conf >= focus_confidence_baseline) {
                focus_confidence_baseline = 0.90 * focus_confidence_baseline + 0.10 * det.conf;
            } else {
                focus_confidence_baseline = 0.995 * focus_confidence_baseline + 0.005 * det.conf;
            }
            focus_conf_drop_frames = conf_drop >= 0.18 ? focus_conf_drop_frames + 1 : 0;

            double blur_var = g_camera_blur_laplacian_var.load();
            focus_low_blur_frames = (blur_var > 0.0 && blur_var < 70.0) ? focus_low_blur_frames + 1 : 0;
        } else {
            focus_low_blur_frames = 0;
            focus_conf_drop_frames = 0;
        }

        long long focus_cooldown_remaining =
            max(0LL, g_autofocus_cooldown_until_ms.load() - now_mono);
        int focus_reason = 0;
        const bool zoom_recently_active =
            (zoom_user != 0) ||
            (last_zoom_motion_time > 0.0 &&
             (now - last_zoom_motion_time) < AUTO_FOCUS_AFTER_ZOOM_SETTLE_SEC);
        const bool zoom_settled_after_motion =
            last_zoom_motion_time > 0.0 &&
            (now - last_zoom_motion_time) >= AUTO_FOCUS_AFTER_ZOOM_SETTLE_SEC;
        if (manual_zoom == 0 &&
            g_autofocus_enabled.load() &&
            focus_cooldown_remaining == 0 &&
            det_fresh_for_ptz &&
            !zoom_recently_active) {
            if (focus_after_zoom_pending &&
                zoom_settled_after_motion &&
                (now - last_focus_after_zoom_request_time) >= AUTO_FOCUS_AFTER_ZOOM_MIN_INTERVAL_SEC) {
                focus_reason = 2;
            } else if (telemetry.target_stable && focus_low_blur_frames >= AUTO_FOCUS_BLUR_STABLE_FRAMES) {
                focus_reason = 3;
            } else if (telemetry.target_stable && focus_conf_drop_frames >= AUTO_FOCUS_CONF_DROP_STABLE_FRAMES) {
                focus_reason = 4;
            }
        }
        if (focus_reason != 0) {
            string focus_msg;
            if (generate_autofocus_state_request(focus_reason, now_mono, &focus_msg)) {
                focus_cooldown_remaining =
                    max(0LL, g_autofocus_cooldown_until_ms.load() - now_mono);
                telemetry.control_source = "focus_request";
                telemetry.focus_triggered = true;
                telemetry.focus_reason = autofocus_reason_name(focus_reason);
                if (focus_reason == 2) {
                    last_focus_after_zoom_request_time = now;
                    focus_after_zoom_pending = false;
                }
                if (focus_reason == 3) focus_low_blur_frames = 0;
                if (focus_reason == 4) focus_conf_drop_frames = 0;
            }
        } else if (zoom_recently_active) {
            telemetry.focus_reason = "zoom_settle";
        } else if (!det_fresh_for_ptz) {
            telemetry.focus_reason = "target_lost";
        }
        telemetry.zoom_speed = zoom_user;
        telemetry.zoom_ratio = cached_ptz.ok ? zoom_to_ratio(cached_ptz.absolute_zoom) : -1.0;
        telemetry.last_focus_request_ms = g_last_autofocus_request_ms.load();
        telemetry.focus_cooldown_remaining_ms =
            max(0LL, g_autofocus_cooldown_until_ms.load() - now_mono);
        if (telemetry.focus_cooldown_remaining_ms > 0) {
            telemetry.focus_state = autofocus_reason_name(g_last_autofocus_reason.load()) + "_COOLDOWN";
        } else {
            telemetry.focus_state = g_autofocus_enabled.load() ? "READY" : "DISABLED";
        }
        if (telemetry.emergency_stop_active) {
            telemetry.control_owner = "emergency_stop";
        } else if (telemetry.manual_override_active) {
            telemetry.control_owner = "manual";
        } else if (telemetry.control_source == "audio" || telemetry.audio_guided) {
            telemetry.control_owner = "audio_guiding";
        } else if (telemetry.zoom_state == "FAST_ZOOM_OUT_SEARCH" ||
                   telemetry.zoom_state == "ZOOMING") {
            telemetry.control_owner = "visual_zoom";
        } else if (det_fresh_for_ptz ||
                   telemetry.control_source == "vision" ||
                   telemetry.control_source == "focus_request") {
            telemetry.control_owner = "visual_tracking";
        } else {
            telemetry.control_owner = "idle";
        }
        update_tracking_runtime_learning(det_fresh_for_ptz, dx, dy, fw, fh, now);
        publish_fusion_telemetry(telemetry);

        bool ok = true;

        // 非阻塞 deadman STOP：只在当前没有任何手动/自动命令时补发。
        // 如果自动模式已经算出了新的非零 pan/tilt/zoom，直接让自动命令覆盖，不额外暂停。
        if (pan_user == 0 && tilt_user == 0 && zoom_user == 0 && pending_manual_stop_retries > 0) {
            if (now >= next_manual_stop_retry_time) {
                ok = send_ptz_continuous(0, 0, 0);
                pending_manual_stop_retries--;
                next_manual_stop_retry_time = now + MANUAL_STOP_RETRY_INTERVAL_MS / 1000.0;
                last_pan = 0;
                last_tilt = 0;
                last_zoom = 0;
                last_cmd_time = now_sec();
            }
        } else if (pan_user == 0 && tilt_user == 0 && zoom_user == 0) {
            send_stop_if_needed();
        } else {
            pending_manual_stop_retries = 0;
            ok = send_ptz_continuous(pan_user, tilt_user, zoom_user);
            last_pan = pan_user;
            last_tilt = tilt_user;
            last_zoom = zoom_user;
            last_cmd_time = now_sec();
        }

        print_count++;
        if (print_count % 5 == 1 || !ok) {
            cout << "\n[PTZ-CONT] " << (ok ? "OK" : "FAIL")
                 << " C=" << (pan_tilt_on ? "ON" : "OFF")
                 << " B=" << (zoom_on ? "ON" : "OFF")
                 << " mode=" << telemetry.mode
                 << " dx=" << dx << " dy=" << dy
                 << " dead=(" << dead_x << "," << dead_y << ")"
                 << " manual=" << (manual_ptz_active ? "PTZ" : (manual_zoom != 0 ? "ZOOM" : "NO"))
                 << " speed=(" << pan_user << "," << tilt_user << "," << zoom_user << ")"
                 << " boxSide=" << box_side_ratio
                 << " boxEma=" << zoom_side_ema
                 << " zoomState=" << telemetry.zoom_state
                 << " zoomAction=" << telemetry.zoom_action
                 << " zoomReason=" << telemetry.zoom_reason
                 << " zErr=" << zoom_error
                 << " conf=" << det.conf
                 << " focus=" << telemetry.focus_state
                 << " focusReason=" << telemetry.focus_reason
                 << " audioErr=" << telemetry.audio_pan_err_deg
                 << " off=" << telemetry.mic_to_camera_offset_deg
                 << " offConf=" << telemetry.offset_confidence
                 << endl;
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    send_ptz_continuous(0, 0, 0);
}

// ============================================================
// 10. 主程序
// ============================================================



void signal_handler(int) {
    is_running.store(false);
}

static string getenv_or_default(const char* name, const string& default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    return string(value);
}

static int getenv_int_or_default(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    int parsed = atoi(value);
    return parsed > 0 ? parsed : default_value;
}

static int getenv_nonnegative_int_or_default(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    int parsed = atoi(value);
    return parsed >= 0 ? parsed : default_value;
}

static bool env_is_set(const char* name) {
    const char* value = getenv(name);
    return value != nullptr && strlen(value) > 0;
}

static string rtsp_host_from_url(const string& url) {
    size_t start = url.find("://");
    start = (start == string::npos) ? 0 : start + 3;
    size_t at = url.find('@', start);
    if (at != string::npos) start = at + 1;
    size_t end = url.find_first_of(":/?", start);
    if (end == string::npos) end = url.size();
    if (end <= start) return "";
    return url.substr(start, end - start);
}

static void print_usage(const char* bin) {
    cout << "Usage: " << bin << " [rtsp_url] [options]\n"
         << "Options:\n"
         << "  --rtsp URL        海康 RTSP 地址\n"
         << "  --model PATH      YOLO RKNN 模型路径\n"
         << "  --udp-ip IP       本机 Qt 视觉 UDP 地址，默认 127.0.0.1\n"
         << "  --udp-port PORT   本机 Qt 视觉 UDP 端口，默认 5005\n"
         << "  --frame-stream-port PORT  RK 同帧视频流 TCP 端口，默认 5010；0 表示禁用\n"
         << "  --audio-fusion-port PORT  RK 本机音频融合 UDP 端口，默认 5007\n"
         << "  --qt-control-port PORT    Qt 手动云台/相机模式控制 UDP 端口，默认 5011\n"
         << "  --fusion-calibration PATH 在线校准 offset 配置文件\n"
         << "  --learning-profile PATH  运行最佳参数学习文件\n"
         << "  --tracker-config PATH    ByteTrack-Lite + PTZ-aware GMC 参数文件\n"
         << "  --control-params PATH    Qt 动态控制参数 JSON 文件\n"
         << "  --infer-stride N   YOLO 推理步长，默认 1 表示每帧推理\n"
         << "  --no-audio-guidance       禁用听觉引导视觉，仅保留日志\n"
         << "  --no-runtime-learning     禁用运行数据最佳状态学习/回退\n"
         << "  --load-audio-offset-prior 允许加载历史声学-视觉 offset；默认关闭，实飞建议先现场校准\n"
         << "  --model-info-only         只加载模型并打印 RKNN 输出信息，不启动 RTSP/云台\n"
         << "  --model-self-test         用一帧黑图跑一次 RKNN 推理和 YOLO 解码后退出\n"
         << "  -h, --help        显示帮助\n";
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    string rtsp_url = getenv_or_default("ANTI_UAV_RTSP_URL", DEFAULT_RTSP_URL);
    g_model_path = getenv_or_default("ANTI_UAV_YOLO_MODEL", DEFAULT_MODEL_PATH);
    g_udp_ip = getenv_or_default("ANTI_UAV_QT_HOST", DEFAULT_UDP_IP);
    g_udp_port = getenv_int_or_default("ANTI_UAV_VISION_PORT", DEFAULT_UDP_PORT);
    g_frame_stream_port = getenv_nonnegative_int_or_default("ANTI_UAV_FRAME_STREAM_PORT", DEFAULT_FRAME_STREAM_PORT);
    g_frame_stream_max_width = max(640, min(1920,
        getenv_int_or_default("ANTI_UAV_FRAME_STREAM_MAX_WIDTH", 1024)));
    g_frame_stream_jpeg_quality = max(40, min(90,
        getenv_int_or_default("ANTI_UAV_FRAME_STREAM_JPEG_QUALITY", 62)));
    g_frame_stream_send_buffer = max(65536, min(1048576,
        getenv_int_or_default("ANTI_UAV_FRAME_STREAM_SEND_BUFFER", 262144)));
    g_frame_stream_send_timeout_ms = max(60, min(1000,
        getenv_int_or_default("ANTI_UAV_FRAME_STREAM_SEND_TIMEOUT_MS", 180)));
    g_frame_stream_letterbox640 =
        getenv_or_default("ANTI_UAV_FRAME_STREAM_VIEW", "letterbox640") != "full";
    g_audio_fusion_port = getenv_int_or_default("ANTI_UAV_FUSION_PORT", DEFAULT_AUDIO_FUSION_PORT);
    g_qt_control_port = getenv_int_or_default("ANTI_UAV_QT_CONTROL_PORT", DEFAULT_QT_CONTROL_PORT);
    g_fusion_calibration_path = getenv_or_default("ANTI_UAV_FUSION_CALIBRATION", g_fusion_calibration_path);
    g_learning_profile_path = getenv_or_default("ANTI_UAV_LEARNING_PROFILE", g_learning_profile_path);
    g_tracker_config_path = getenv_or_default("ANTI_UAV_TRACKER_CONFIG", g_tracker_config_path);
    g_control_params_path = getenv_or_default("ANTI_UAV_CONTROL_PARAMS", g_control_params_path);
    g_control_params_session_snapshot_path =
        getenv_or_default("ANTI_UAV_CONTROL_PARAMS_SESSION_SNAPSHOT", "");
    if (g_control_params_session_snapshot_path.empty()) {
        string data_root = getenv_or_default("ANTI_UAV_DATA_ROOT", "");
        string session_id = getenv_or_default("ANTI_UAV_SESSION_ID", "");
        if (!data_root.empty() && !session_id.empty()) {
            g_control_params_session_snapshot_path =
                data_root + "/sessions/" + session_id + "/config/control_params_snapshot.json";
        }
    }
    g_infer_frame_stride = getenv_int_or_default("ANTI_UAV_YOLO_INFER_STRIDE", 1);
    g_runtime_learning_enabled = getenv_nonnegative_int_or_default("ANTI_UAV_RUNTIME_LEARNING", 1) != 0;
    g_load_audio_offset_prior = getenv_nonnegative_int_or_default("ANTI_UAV_LOAD_AUDIO_OFFSET_PRIOR", 0) != 0;
    if (env_is_set("PTZ_HOME_PAN") &&
        env_is_set("PTZ_HOME_TILT") &&
        env_is_set("PTZ_HOME_ZOOM")) {
        g_config_home_ptz.ok = true;
        g_config_home_ptz.azimuth = wrap_azimuth(atoi(getenv("PTZ_HOME_PAN")));
        g_config_home_ptz.elevation = max(-900, min(900, atoi(getenv("PTZ_HOME_TILT"))));
        g_config_home_ptz.absolute_zoom = max(10, min(370, atoi(getenv("PTZ_HOME_ZOOM"))));
        g_config_home_valid = true;
    }
    g_config_home_preset = getenv_nonnegative_int_or_default("PTZ_HOME_PRESET", 0);
    bool model_info_only = false;
    bool model_self_test = false;

    bool rtsp_positional_set = false;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need_value = [&](const string& name) -> const char* {
            if (i + 1 >= argc) {
                cerr << "[ERROR] 参数缺少值: " << name << endl;
                print_usage(argv[0]);
                exit(2);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--rtsp") {
            rtsp_url = need_value(arg);
            rtsp_positional_set = true;
        } else if (arg == "--model") {
            g_model_path = need_value(arg);
        } else if (arg == "--udp-ip" || arg == "--udp-host") {
            g_udp_ip = need_value(arg);
        } else if (arg == "--udp-port") {
            g_udp_port = atoi(need_value(arg));
            if (g_udp_port <= 0) {
                cerr << "[ERROR] UDP 端口非法: " << g_udp_port << endl;
                return 2;
            }
        } else if (arg == "--frame-stream-port") {
            g_frame_stream_port = atoi(need_value(arg));
            if (g_frame_stream_port < 0) {
                cerr << "[ERROR] RK 同帧视频流端口非法: " << g_frame_stream_port << endl;
                return 2;
            }
        } else if (arg == "--audio-fusion-port") {
            g_audio_fusion_port = atoi(need_value(arg));
            if (g_audio_fusion_port <= 0) {
                cerr << "[ERROR] 音频融合 UDP 端口非法: " << g_audio_fusion_port << endl;
                return 2;
            }
        } else if (arg == "--qt-control-port") {
            g_qt_control_port = atoi(need_value(arg));
            if (g_qt_control_port <= 0) {
                cerr << "[ERROR] Qt 控制 UDP 端口非法: " << g_qt_control_port << endl;
                return 2;
            }
        } else if (arg == "--fusion-calibration") {
            g_fusion_calibration_path = need_value(arg);
        } else if (arg == "--learning-profile") {
            g_learning_profile_path = need_value(arg);
        } else if (arg == "--tracker-config") {
            g_tracker_config_path = need_value(arg);
        } else if (arg == "--control-params") {
            g_control_params_path = need_value(arg);
        } else if (arg == "--infer-stride") {
            g_infer_frame_stride = atoi(need_value(arg));
            if (g_infer_frame_stride <= 0 || g_infer_frame_stride > 10) {
                cerr << "[ERROR] YOLO 推理步长非法: " << g_infer_frame_stride << endl;
                return 2;
            }
        } else if (arg == "--no-audio-guidance") {
            g_audio_guidance_enabled.store(false);
        } else if (arg == "--no-runtime-learning") {
            g_runtime_learning_enabled = false;
        } else if (arg == "--load-audio-offset-prior") {
            g_load_audio_offset_prior = true;
        } else if (arg == "--model-info-only") {
            model_info_only = true;
        } else if (arg == "--model-self-test") {
            model_self_test = true;
        } else if (!arg.empty() && arg[0] != '-' && !rtsp_positional_set) {
            rtsp_url = arg;
            rtsp_positional_set = true;
        } else {
            cerr << "[ERROR] 未知参数: " << arg << endl;
            print_usage(argv[0]);
            return 2;
        }
    }

    if (rtsp_url.empty() && !model_info_only && !model_self_test) {
        cerr << "[ERROR] 未设置 RTSP 输入源。请设置 ANTI_UAV_RTSP_URL，或使用 --rtsp 传入完整地址。" << endl;
        print_usage(argv[0]);
        return 2;
    }

    if (!env_is_set("ANTI_UAV_HIK_IP")) {
        string rtsp_host = rtsp_host_from_url(rtsp_url);
        if (!rtsp_host.empty()) {
            HIK_IP = rtsp_host;
        }
    }
    refresh_hik_urls();

    cout << "\n🚀 启动海康球机 RKNN YOLO 无人机识别程序" << endl;
    cout << "✅ 已取消环境适应学习 / 死物黑名单" << endl;
    cout << "✅ 已接入 ByteTrack-Lite + PTZ-aware GMC，低置信检测仅用于二次关联续接轨迹" << endl;
    cout << "✅ YOLO 推理与显示线程解耦，支持 FP 单输出和 INT8 raw-head 输出" << endl;
    cout << "✅ 自动云台跟踪：" << (ptz_control_enabled.load() ? "ON" : "OFF")
         << "，自动变焦：" << (auto_zoom_enabled.load() ? "ON" : "OFF") << endl;
    cout << "✅ C/B 开关自动模式；WSAD/+/- 手动优先，松开后自动恢复" << endl;
    cout << "✅ 模型：" << g_model_path << endl;
    cout << "✅ 输入源：" << redact_url_credentials(rtsp_url) << endl;
    cout << "✅ 海康控制 IP：" << HIK_IP
         << " env_override=" << (env_is_set("ANTI_UAV_HIK_IP") ? "ON" : "OFF") << endl;
    cout << "✅ 输出 UDP：" << g_udp_ip << ":" << g_udp_port << endl;
    cout << "✅ RK 同帧视频流 TCP："
         << (g_frame_stream_port > 0 ? ("0.0.0.0:" + to_string(g_frame_stream_port)) : string("disabled"))
         << endl;
    cout << "✅ RK 同帧低延迟参数：max_width=" << g_frame_stream_max_width
         << " jpeg_quality=" << g_frame_stream_jpeg_quality
         << " send_buffer=" << g_frame_stream_send_buffer
         << " send_timeout_ms=" << g_frame_stream_send_timeout_ms
         << " view=" << (g_frame_stream_letterbox640 ? "letterbox640" : "full")
         << endl;
    cout << "✅ YOLO 推理步长：每 " << g_infer_frame_stride << " 帧推理一次" << endl;
    cout << "✅ 音频融合 UDP：127.0.0.1:" << g_audio_fusion_port
         << " guidance=" << (g_audio_guidance_enabled.load() ? "ON" : "OFF") << endl;
    cout << "✅ Qt 手动控制 UDP：0.0.0.0:" << g_qt_control_port << endl;
    cout << "✅ 摄像头日夜模式：启动强制白天彩色="
         << (FORCE_DAY_MODE_ON_START ? "ON" : "OFF")
         << " disable_auto_ir_for_test=" << (DISABLE_AUTO_IR_FOR_TEST ? "ON" : "OFF")
         << " presets(day/night/auto)=" << PRESET_DAY_MODE << "/"
         << PRESET_NIGHT_MODE << "/" << PRESET_AUTO_DAYNIGHT << endl;
    cout << "✅ 摄像头图像 Profile：默认 "
         << camera_profile_to_string(DEFAULT_CAMERA_PROFILE)
         << " auto_profile=" << (g_auto_camera_profile_enabled.load() ? "ON" : "OFF")
         << " allow_auto_night_ir=" << (ALLOW_AUTO_NIGHT_IR ? "ON" : "OFF") << endl;
    cout << "✅ 融合校准文件：" << g_fusion_calibration_path << endl;
    cout << "✅ 运行最佳学习：" << (g_runtime_learning_enabled ? "ON" : "OFF")
         << " profile=" << g_learning_profile_path << endl;
    cout << "✅ ByteTrack-Lite 配置：" << g_tracker_config_path << endl;
    cout << "✅ 控制参数文件：" << g_control_params_path
         << " session_snapshot=" << (g_control_params_session_snapshot_path.empty()
                                      ? "OFF" : g_control_params_session_snapshot_path)
         << endl;
    cout << "✅ NPU：Core 0 单线程 YOLO 推理，Core 2 保留给音频 YAMNet" << endl;
    cout << "按 C 开/关跟踪，B 开/关自动变焦，WSAD 手动云台，+/- 手动变焦，空格急停，q 或 Ctrl+C 退出。\n" << endl;

    load_fusion_calibration();
    load_runtime_learning_profile();
    load_control_params_file(g_control_params_path);
    cout << "✅ 当前检测参数："
         << " search_conf=" << fixed << setprecision(3) << g_search_conf_thresh.load()
         << " track_conf=" << fixed << setprecision(3) << g_track_conf_thresh.load()
         << " alarm_conf=" << fixed << setprecision(3) << g_alarm_conf_thresh.load()
         << " search_min_box=" << fixed << setprecision(3) << g_search_min_box_ratio.load()
         << " track_min_box=" << fixed << setprecision(3) << g_track_min_box_ratio.load()
         << " alarm_min_box=" << fixed << setprecision(3) << g_alarm_min_box_ratio.load()
         << " control_params=" << g_control_params_path
         << endl;
    const int configured_ptz_speed_cap =
        getenv_nonnegative_int_or_default("ANTI_UAV_PTZ_MAX_SPEED_CAP", 0);
    if (configured_ptz_speed_cap > 0) {
        const int ptz_speed_cap =
            clamp_val(configured_ptz_speed_cap, g_ptz_min_speed.load(), 110);
        g_ptz_max_speed.store(min(g_ptz_max_speed.load(), ptz_speed_cap));
        g_ptz_max_speed_high_zoom.store(
            min(g_ptz_max_speed_high_zoom.load(), ptz_speed_cap));
        cout << "[SAFETY] automatic PTZ speed capped at "
             << ptz_speed_cap << endl;
    }
    auto env_text = [](const char* name, const string& code_default) -> string {
        const char* value = getenv(name);
        if (value == nullptr || strlen(value) == 0) {
            return string("<unset; code_default=") + code_default + ">";
        }
        return string(value);
    };
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_AUTO_PTZ env="
         << env_text("ANTI_UAV_AUTO_PTZ", "1")
         << " final=" << (ptz_control_enabled.load() ? 1 : 0)
         << " source=env_then_control_params(" << g_control_params_path << ")" << endl;
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_AUTO_ZOOM env="
         << env_text("ANTI_UAV_AUTO_ZOOM", "1")
         << " final=" << (auto_zoom_enabled.load() ? 1 : 0)
         << " source=env_then_control_params(" << g_control_params_path << ")" << endl;
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_AUTO_FOCUS env="
         << env_text("ANTI_UAV_AUTO_FOCUS", "1")
         << " final=" << (g_autofocus_enabled.load() ? 1 : 0)
         << " source=code_default_then_control_params(" << g_control_params_path << ")" << endl;
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_PTZ_MAX_SPEED_CAP env="
         << env_text("ANTI_UAV_PTZ_MAX_SPEED_CAP", "0")
         << " final=" << configured_ptz_speed_cap
         << " effective_max_speed=" << g_ptz_max_speed.load()
         << " effective_high_zoom_max_speed=" << g_ptz_max_speed_high_zoom.load() << endl;
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_AUDIO_GUIDANCE_ENABLED env="
         << env_text("ANTI_UAV_AUDIO_GUIDANCE_ENABLED", "1")
         << " final=" << (g_audio_guidance_enabled.load() ? 1 : 0)
         << " source=code_default_or_cli" << endl;
    if (env_is_set("ANTI_UAV_AUDIO_GUIDANCE_ENABLED")) {
        string audio_env = getenv_or_default("ANTI_UAV_AUDIO_GUIDANCE_ENABLED", "");
        if ((audio_env == "0" || audio_env == "false" || audio_env == "OFF" || audio_env == "off") &&
            g_audio_guidance_enabled.load()) {
            cout << "[EFFECTIVE_CONFIG_WARN] ANTI_UAV_AUDIO_GUIDANCE_ENABLED is set but this binary currently only applies --no-audio-guidance; final guidance remains ON." << endl;
        }
    }
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_RTSP_BACKEND env="
         << env_text("ANTI_UAV_RTSP_BACKEND", "gstreamer_mpp")
         << " ANTI_UAV_GST_LATENCY_MS env=" << env_text("ANTI_UAV_GST_LATENCY_MS", "50") << endl;
    cout << "[EFFECTIVE_CONFIG] ANTI_UAV_FRAME_STREAM_VIEW env="
         << env_text("ANTI_UAV_FRAME_STREAM_VIEW", "letterbox640")
         << " final=" << (g_frame_stream_letterbox640 ? "letterbox640" : "full")
         << " ANTI_UAV_FRAME_STREAM_SEND_BUFFER env="
         << env_text("ANTI_UAV_FRAME_STREAM_SEND_BUFFER", "262144")
         << " final=" << g_frame_stream_send_buffer << endl;
    cout << "[EFFECTIVE_CONFIG] search_conf=" << fixed << setprecision(3) << g_search_conf_thresh.load()
         << " track_conf=" << fixed << setprecision(3) << g_track_conf_thresh.load()
         << " alarm_conf=" << fixed << setprecision(3) << g_alarm_conf_thresh.load()
         << " auto_track_enabled=" << (ptz_control_enabled.load() ? 1 : 0)
         << " auto_zoom_enabled=" << (auto_zoom_enabled.load() ? 1 : 0)
         << " auto_focus_enabled=" << (g_autofocus_enabled.load() ? 1 : 0)
         << " audio_guidance_enabled=" << (g_audio_guidance_enabled.load() ? 1 : 0)
         << endl;

    vector<unsigned char> model_data;
    log_model_file_metadata(g_model_path);
    if (!load_model_bytes(g_model_path, model_data)) {
        return -1;
    }

    rknn_context ctx_rknn;
    int ret = rknn_init(&ctx_rknn,
                        model_data.data(),
                        static_cast<uint32_t>(model_data.size()),
                        0,
                        NULL);
    if (ret < 0) {
        cerr << "❌ RKNPU 初始化失败，ret=" << ret << endl;
        return -1;
    }
    cout << "✅ RKNN load result：ret=" << ret
         << " model_bytes=" << model_data.size() << endl;
    rknn_set_core_mask(ctx_rknn, RKNN_NPU_CORE_0);

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx_rknn, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret == 0 && io_num.n_output > 0) {
        g_rknn_output_count = static_cast<int>(io_num.n_output);
        cout << "✅ RKNN 输出数量：" << g_rknn_output_count << endl;
        for (int i = 0; i < g_rknn_output_count; ++i) {
            rknn_tensor_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.index = i;
            int attr_ret = rknn_query(ctx_rknn, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (attr_ret == 0) {
                cout << "   output" << i << " dims=[";
                for (int d = 0; d < attr.n_dims; ++d) {
                    if (d > 0) cout << ",";
                    cout << attr.dims[d];
                }
                cout << "] size=" << attr.size << endl;
            }
        }
    } else {
        g_rknn_output_count = metric_model_mode_cpp() == "INT8" ? 6 : 1;
        cerr << "[RKNN] query output count failed ret=" << ret
             << ", fallback output_count=" << g_rknn_output_count << endl;
    }

    if (model_info_only) {
        rknn_destroy(ctx_rknn);
        cout << "[MAIN] model-info-only done" << endl;
        return 0;
    }

    if (model_self_test) {
        vector<unsigned char> dummy(IMG_SIZE * IMG_SIZE * 3, 0);

        rknn_input input;
        memset(&input, 0, sizeof(input));
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.size = IMG_SIZE * IMG_SIZE * 3;
        input.fmt = RKNN_TENSOR_NHWC;
        input.buf = dummy.data();

        ret = rknn_inputs_set(ctx_rknn, 1, &input);
        if (ret < 0) {
            cerr << "[SELFTEST] rknn_inputs_set failed: " << ret << endl;
            rknn_destroy(ctx_rknn);
            return 3;
        }

        ret = rknn_run(ctx_rknn, NULL);
        if (ret < 0) {
            cerr << "[SELFTEST] rknn_run failed: " << ret << endl;
            rknn_destroy(ctx_rknn);
            return 3;
        }

        vector<rknn_output> outputs(g_rknn_output_count);
        memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
        for (int i = 0; i < g_rknn_output_count; ++i) {
            outputs[i].want_float = 1;
        }

        ret = rknn_outputs_get(ctx_rknn, g_rknn_output_count, outputs.data(), NULL);
        if (ret < 0) {
            cerr << "[SELFTEST] rknn_outputs_get failed: " << ret << endl;
            rknn_destroy(ctx_rknn);
            return 3;
        }

        for (int i = 0; i < g_rknn_output_count; ++i) {
            cout << "[SELFTEST] output" << i << " size=" << outputs[i].size
                 << " want_float=1" << endl;
        }

        vector<ObjectBox> boxes;
        if (g_rknn_output_count == 1) {
            boxes = decode_yolo11_single_output(outputs.data(), IMG_SIZE);
        } else if (g_rknn_output_count == 6) {
            boxes = decode_yolo11_raw_head_outputs(outputs.data(), g_rknn_output_count, IMG_SIZE);
        }

        cout << "[SELFTEST] decoded_boxes=" << boxes.size() << endl;
        rknn_outputs_release(ctx_rknn, g_rknn_output_count, outputs.data());
        rknn_destroy(ctx_rknn);
        cout << "[MAIN] model-self-test done" << endl;
        return 0;
    }

    if (FORCE_DAY_MODE_ON_START) {
        cout << "[CAMERA_PROFILE] startup default profile scheduled after 1s" << endl;
        this_thread::sleep_for(chrono::seconds(1));
        apply_camera_profile(DEFAULT_CAMERA_PROFILE, "startup_default_profile");
    }

    thread t_capture(capture_thread, rtsp_url);
    thread t_infer0(inference_thread, 0, ctx_rknn);
    thread t_post(postprocess_thread);
    thread t_frame_stream(frame_stream_thread);
    thread t_display(display_thread);
    thread t_audio(audio_fusion_receiver_thread);
    thread t_qt_ctrl(qt_control_receiver_thread);
    thread t_ptz(ptz_control_thread);

    t_capture.join();
    t_infer0.join();
    t_post.join();
    latest_stream_cv.notify_all();
    t_frame_stream.join();
    t_display.join();
    t_audio.join();
    t_qt_ctrl.join();
    t_ptz.join();

    safe_shutdown_hikvision_camera("vision_main_exit");

    rknn_destroy(ctx_rknn);

    cout << "\n[MAIN] exited safely" << endl;
    return 0;
}
