#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <atomic>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <array>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "yolo_decoder.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;
using namespace cv;

static string env_string_or_default(const char* name, const string& default_value) {
    const char* value = getenv(name);
    if (value == nullptr || strlen(value) == 0) return default_value;
    return string(value);
}

// ============================================================
// 1. 基础配置：海康 RTSP + RKNN YOLO
// ============================================================

const string MODEL_PATH = env_string_or_default(
    "ANTI_UAV_YOLO_MODEL",
    "models/best_uav_headless_i8.rknn");

// 默认不内置摄像头凭据；通过 ANTI_UAV_RTSP_URL 或第一个启动参数传入完整 RTSP。
const string DEFAULT_RTSP_URL = env_string_or_default("ANTI_UAV_RTSP_URL", "");

const int IMG_SIZE = 640;
const float IOU_THRESH = 0.40f;

// 摄像头倒装时保持 true；如果后面物理正装，改成 false。
const bool ROTATE_IMAGE_180 = true;

// 是否写入共享内存，兼容你之前的 Qt / 上位机显示逻辑。
const bool ENABLE_SHM = true;
const char* SHM_NAME = "/rk3588_vision";

// 是否显示 OpenCV 调试窗口。没有桌面环境时建议 false。
const bool ENABLE_IMSHOW = true;

// UDP 输出给云台控制 / Qt 程序。
const char* UDP_IP = "127.0.0.1";
const int UDP_PORT = 5005;

atomic<bool> is_running(true);

// ============================================================
// 1.1 FP 低频推理 + 海康 PTZ 粗闭环控制配置
// ============================================================

// FP RKNN 较慢，默认每 3 帧推理一次。卡顿明显改 5；想更灵敏改 2 或 1。
const int INFER_FRAME_STRIDE = 2;

// 按 C 开/关云台自动跟踪。默认关闭，避免程序启动后云台乱动。
atomic<bool> ptz_control_enabled(false);

// 海康球机控制参数
const string HIK_IP = env_string_or_default("ANTI_UAV_HIK_IP", "CAMERA_IP");
const string HIK_USER = env_string_or_default("ANTI_UAV_CAMERA_USER", "YOUR_USERNAME");
const string HIK_PASS = env_string_or_default("ANTI_UAV_CAMERA_PASSWORD", "");
const int HIK_CHANNEL = 1;

const string PTZ_STATUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/status";
const string PTZ_ABSOLUTE_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/absolute";
const string PTZ_CONTINUOUS_URL = "http://" + HIK_IP + "/ISAPI/PTZCtrl/channels/1/continuous";

// 你前面实测的 absolute 方向
const int ABS_PAN_DIR = -1;
const int ABS_TILT_DIR =1;

// 粗闭环参数：按 C 后使用 continuous 小速度脉冲控制，反应比 absolute + 查询状态快得多。
// 目标：只要把无人机拉回中心小框附近，不做像素级精准对中。
const float PTZ_CONTROL_PERIOD_SEC = 0.12f;     // 控制周期，越小反应越快；太小可能抖。建议 0.15~0.25
const float PTZ_TARGET_MAX_AGE_SEC = 0.75f;     // 检测结果超过这个时间就不再控制，防止追旧目标

// 中心允许范围：你说框太大，这里已经缩小。
// 704x576 下约为 x±70px, y±52px；同时设置较小的最小值。
const float CENTER_DEAD_ZONE_X_RATIO = 0.06f;
const float CENTER_DEAD_ZONE_Y_RATIO = 0.06f;
const float CENTER_DEAD_ZONE_X_MIN_PX = 38.0f;
const float CENTER_DEAD_ZONE_Y_MIN_PX = 30.0f;

// continuous 控制方向：沿用你之前 Python 控制脚本里的方向。
const int CONT_PAN_DIR = -1;
const int CONT_TILT_DIR = 1;
const int CONT_ZOOM_DIR = 1;

// continuous 速度。越大反应越快，也越容易过冲。
const int PTZ_MIN_SPEED = 15;
const int PTZ_MAX_SPEED = 60;
const int PTZ_MAX_SPEED_HIGH_ZOOM = 28;

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
const int CONFIRM_FRAMES = 2;

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

// 显示线程用：采集线程每帧更新，不依赖 YOLO 推理频率，避免 FP 推理慢导致窗口卡成几 fps。
mutex latest_camera_mtx;
Mat latest_camera_frame;
int latest_camera_id = -1;

struct SharedDetection {
    bool valid = false;
    bool confirmed = false;
    int frame_id = -1;
    int frame_w = 0;
    int frame_h = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float conf = 0.0f;
    double ts = 0.0;
};

mutex latest_det_mtx;
SharedDetection latest_det;



// ============================================================
// 3.1 PTZ 控制工具函数：curl + Digest Auth，无需修改 CMake 链接库
// ============================================================

double now_sec() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::steady_clock::now().time_since_epoch()
    ).count() / 1000000.0;
}

float clampf(float v, float lo, float hi) {
    return max(lo, min(hi, v));
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

struct PTZState {
    bool ok = false;
    int azimuth = 0;
    int elevation = 0;
    int absolute_zoom = 10;
};

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

bool send_ptz_absolute(int azimuth, int elevation, int absolute_zoom) {
    azimuth = wrap_azimuth(azimuth);
    elevation = max(-900, min(900, elevation));

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

void update_shared_detection(bool valid, bool confirmed, int frame_id, int frame_w, int frame_h, const DetBox* d) {
    lock_guard<mutex> lock(latest_det_mtx);

    latest_det.valid = valid;
    latest_det.confirmed = confirmed;
    latest_det.frame_id = frame_id;
    latest_det.frame_w = frame_w;
    latest_det.frame_h = frame_h;
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

    cout << "[VIDEO] try OpenCV FFmpeg TCP..." << endl;

    // 保守 TCP 参数。过激低延迟参数可能导致某些 OpenCV/FFmpeg 组合打开失败。
    setenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;3000000",
        1
    );

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

    cout << "[VIDEO] CAP_ANY failed, try GStreamer pipeline..." << endl;

    std::string gst =
        "rtspsrc location=" + url +
        " protocols=tcp latency=80 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! appsink sync=false max-buffers=1 drop=true";

    if (cap.open(gst, cv::CAP_GSTREAMER)) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cout << "[VIDEO] opened by CAP_GSTREAMER" << endl;
        return true;
    }

    cap.release();

    cerr << "[VIDEO] all RTSP open methods failed" << endl;
    return false;
}

// ============================================================
// 6. RTSP 采集线程
// ============================================================

void capture_thread(string rtsp_url) {
    // OpenCV FFmpeg 低延迟参数。
    setenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;100000|probesize;2048|max_delay;0",
        1
    );

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
        cout << "[VIDEO] opening: " << rtsp_url << endl;

        VideoCapture cap;

        if (!open_rtsp_capture(cap, rtsp_url)) {
            cerr << "❌ RTSP 打开失败，1 秒后重试。请检查 IP、账号密码、网络脚本。" << endl;
            this_thread::sleep_for(chrono::seconds(1));
            continue;
        }

        cout << "[VIDEO] opened" << endl;

        while (is_running.load()) {
            Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                cerr << "⚠️ RTSP 读取失败，准备重连" << endl;
                break;
            }

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

            // FP 模型较慢：低频送入 YOLO，避免 NPU/CPU 后处理拖慢画面。
            if (this_frame_id % INFER_FRAME_STRIDE != 0) {
                continue;
            }

            FrameContext ctx;
            ctx.frame_id = this_frame_id;
            ctx.src_w = frame.cols;
            ctx.src_h = frame.rows;

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

        ret = rknn_run(ctx_rknn, NULL);
        if (ret < 0) {
            cerr << "[RKNN] run failed: " << ret << endl;
            continue;
        }

        // YOLO11n FP RKNN 单输出：output0 = [1,5,8400]
        // 固定 RGB_CHW_a5_c4_sig0，不猜 HWC，避免把坐标误当 conf=1.0。
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;

        ret = rknn_outputs_get(ctx_rknn, 1, outputs, NULL);
        if (ret < 0) {
            cerr << "[RKNN] outputs_get failed: " << ret << endl;
            continue;
        }

        ctx.decoded_boxes = decode_yolo11_single_output(outputs, IMG_SIZE);
        rknn_outputs_release(ctx_rknn, 1, outputs);

        ctx.infer_rgb.release();
        output_queue.push_latest(std::move(ctx));
    }
}

// ============================================================
// 8. 后处理线程：取消环境学习，输出稳定目标
// ============================================================

void send_udp_json(int sock, const sockaddr_in& addr, const string& json) {
    sendto(sock, json.c_str(), json.size(), 0, (const struct sockaddr*)&addr, sizeof(addr));
}

string make_json_no_target(int frame_id, int frame_w, int frame_h) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"frame_id\":%d,\"detected\":0,\"tracking\":0,\"frame_w\":%d,\"frame_h\":%d,"
             "\"bbox_x\":0,\"bbox_y\":0,\"bbox_w\":0,\"bbox_h\":0,"
             "\"cx\":0,\"cy\":0,\"dx\":0,\"dy\":0,\"conf\":0}",
             frame_id, frame_w, frame_h);
    return string(buf);
}

string make_json_target(int frame_id, int frame_w, int frame_h, const DetBox& d, bool tracking) {
    int x = (int)round(d.cx - d.w / 2.0f);
    int y = (int)round(d.cy - d.h / 2.0f);
    int w = (int)round(d.w);
    int h = (int)round(d.h);

    float dx = d.cx - frame_w / 2.0f;
    float dy = d.cy - frame_h / 2.0f;

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"frame_id\":%d,\"detected\":1,\"tracking\":%d,\"frame_w\":%d,\"frame_h\":%d,"
             "\"bbox_x\":%d,\"bbox_y\":%d,\"bbox_w\":%d,\"bbox_h\":%d,"
             "\"cx\":%.1f,\"cy\":%.1f,\"dx\":%.1f,\"dy\":%.1f,\"conf\":%.3f,"
             "\"ui_cx\":%.1f,\"ui_cy\":%.1f,\"ui_w\":%.1f,\"ui_h\":%.1f}",
             frame_id,
             tracking ? 1 : 0,
             frame_w,
             frame_h,
             x, y, w, h,
             d.cx, d.cy, dx, dy, d.conf,
             d.ui_cx, d.ui_cy, d.ui_w, d.ui_h);
    return string(buf);
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

bool is_reasonable_box(const DetBox& d, int frame_w, int frame_h) {
    if (d.conf < 0.0f) return false;
    if (d.w < 2.0f || d.h < 2.0f) return false;

    float frame_area = max(1.0f, (float)(frame_w * frame_h));
    float area = d.w * d.h;
    float area_ratio = area / frame_area;

    if (area_ratio < MIN_BOX_AREA_RATIO) return false;
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
    KalmanTracker tracker;
    TrackState state = TrackState::SEARCHING;

    int confirm_count = 0;
    int candidate_miss = 0;
    int lost_frames = 0;

    DetBox last_candidate;
    DetBox last_confirmed;
    Point2f last_candidate_center(0, 0);

    auto start_time = chrono::high_resolution_clock::now();
    int highest_processed_id = -1;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(UDP_PORT);
    servaddr.sin_addr.s_addr = inet_addr(UDP_IP);

    cout << "✅ 后处理启动：Candidate/Confirmed/Lost 三状态，低置信误检不会驱动云台" << endl;
    cout << "✅ ACQUIRE_CONF=" << ACQUIRE_CONF
         << ", TRACK_CONF=" << TRACK_CONF
         << ", CONFIRM_FRAMES=" << CONFIRM_FRAMES
         << ", MAX_LOST_FRAMES=" << MAX_LOST_FRAMES << endl;

    while (is_running.load()) {
        FrameContext ctx;
        if (!output_queue.pop(ctx)) continue;

        if (ctx.frame_id <= highest_processed_id) continue;
        highest_processed_id = ctx.frame_id;

        // 1. 将 RKNN 解码框映射回海康原始画面，并做几何过滤。
        vector<Rect> nms_boxes;
        vector<float> nms_scores;
        vector<DetBox> dets;

        // NMS 前先保留所有 >= TRACK_CONF 的检测。首次捕获时后面还会再要求 >= ACQUIRE_CONF。
        for (const auto& raw : ctx.decoded_boxes) {
            if (raw.conf < TRACK_CONF) continue;

            DetBox d = map_box_to_original(raw, ctx);
            if (!is_reasonable_box(d, ctx.src_w, ctx.src_h)) continue;

            dets.push_back(d);
            nms_boxes.push_back(det_to_rect(d, ctx.src_w, ctx.src_h));
            nms_scores.push_back(d.conf);
        }

        vector<int> indices;
        if (!nms_boxes.empty()) {
            dnn::NMSBoxes(nms_boxes, nms_scores, TRACK_CONF, IOU_THRESH, indices);
        }

        // 2. 根据当前状态选择目标。
        bool output_target = false;
        bool is_confirmed_output = false;
        DetBox target;

        Point2f pred_pt(0, 0);
        if (state == TrackState::CONFIRMED && tracker.initialized) {
            pred_pt = tracker.predict();
        }

        if (state == TrackState::SEARCHING) {
            int best_idx = -1;
            float best_conf = -1.0f;

            for (int idx : indices) {
                const DetBox& d = dets[idx];
                if (d.conf < ACQUIRE_CONF) continue;

                if (d.conf > best_conf) {
                    best_conf = d.conf;
                    best_idx = idx;
                }
            }

            if (best_idx >= 0) {
                last_candidate = dets[best_idx];
                last_candidate_center = Point2f(last_candidate.cx, last_candidate.cy);
                confirm_count = 1;
                candidate_miss = 0;
                state = TrackState::CANDIDATE;

                target = last_candidate;
                output_target = true;
                is_confirmed_output = false;
            }
        }
        else if (state == TrackState::CANDIDATE) {
            int best_idx = -1;
            float best_score = -999.0f;

            for (int idx : indices) {
                const DetBox& d = dets[idx];
                if (d.conf < ACQUIRE_CONF) continue;

                Point2f c(d.cx, d.cy);
                float dist = norm(c - last_candidate_center);
                if (dist > CANDIDATE_GATE_PX) continue;

                if (!size_is_consistent(last_candidate, d)) continue;

                float score = d.conf - 0.35f * (dist / CANDIDATE_GATE_PX);
                if (score > best_score) {
                    best_score = score;
                    best_idx = idx;
                }
            }

            if (best_idx >= 0) {
                last_candidate = dets[best_idx];
                last_candidate_center = Point2f(last_candidate.cx, last_candidate.cy);
                confirm_count++;
                candidate_miss = 0;

                target = last_candidate;
                output_target = true;
                is_confirmed_output = false;

                if (confirm_count >= CONFIRM_FRAMES) {
                    tracker.init(last_candidate.cx, last_candidate.cy);
                    last_confirmed = last_candidate;
                    state = TrackState::CONFIRMED;
                    lost_frames = 0;

                    target = last_candidate;
                    output_target = true;
                    is_confirmed_output = true;
                }
            } else {
                candidate_miss++;

                if (candidate_miss > CANDIDATE_MISS_MAX) {
                    state = TrackState::SEARCHING;
                    confirm_count = 0;
                    candidate_miss = 0;
                    tracker.initialized = false;
                }
            }
        }
        else if (state == TrackState::CONFIRMED) {
            int best_idx = -1;
            float best_score = -999.0f;

            for (int idx : indices) {
                const DetBox& d = dets[idx];
                if (d.conf < TRACK_CONF) continue;

                Point2f c(d.cx, d.cy);
                float dist = norm(c - pred_pt);
                if (dist > TRACK_GATE_PX) continue;

                if (!size_is_consistent(last_confirmed, d)) continue;

                float score = d.conf - 0.45f * (dist / TRACK_GATE_PX);
                if (score > best_score) {
                    best_score = score;
                    best_idx = idx;
                }
            }

            if (best_idx >= 0) {
                DetBox measured = dets[best_idx];
                Point2f corrected = tracker.update(measured.cx, measured.cy);

                target = measured;
                target.cx = clamp_float(corrected.x, 0.0f, (float)(ctx.src_w - 1));
                target.cy = clamp_float(corrected.y, 0.0f, (float)(ctx.src_h - 1));

                last_confirmed = target;
                lost_frames = 0;

                output_target = true;
                is_confirmed_output = true;
            } else {
                lost_frames++;

                // 关键修改：短时丢失时内部保留状态，但不输出 conf=0.01 的假目标，避免云台追 ghosts。
                output_target = false;
                is_confirmed_output = false;

                if (lost_frames > MAX_LOST_FRAMES) {
                    state = TrackState::SEARCHING;
                    confirm_count = 0;
                    candidate_miss = 0;
                    lost_frames = 0;
                    tracker.initialized = false;
                }
            }
        }

        // 3. UDP 输出 + 共享给显示/云台控制线程。
        string json;
        if (output_target) {
            update_shared_detection(true, is_confirmed_output, ctx.frame_id, ctx.src_w, ctx.src_h, &target);
            json = make_json_target(ctx.frame_id, ctx.src_w, ctx.src_h, target, is_confirmed_output);
            send_udp_json(sock, servaddr, json);
            draw_debug_on_ui(ctx, &target, is_confirmed_output);
        } else {
            update_shared_detection(false, false, ctx.frame_id, ctx.src_w, ctx.src_h, nullptr);
            json = make_json_no_target(ctx.frame_id, ctx.src_w, ctx.src_h);
            send_udp_json(sock, servaddr, json);
            draw_debug_on_ui(ctx, nullptr, false);
        }

        auto now = chrono::high_resolution_clock::now();
        float elapsed = chrono::duration_cast<chrono::microseconds>(now - start_time).count() / 1000000.0f;
        float fps = (highest_processed_id + 1) / max(0.001f, elapsed);

        char status[320];
        if (output_target) {
            snprintf(status, sizeof(status),
                     "FPS:%4.1f | %s | conf:%.2f | cx:%4.0f cy:%4.0f | dx:%+5.0f dy:%+5.0f | hits:%d lost:%d | frame:%dx%d",
                     fps,
                     is_confirmed_output ? "TRACK" : "CAND",
                     target.conf,
                     target.cx,
                     target.cy,
                     target.cx - ctx.src_w / 2.0f,
                     target.cy - ctx.src_h / 2.0f,
                     confirm_count,
                     lost_frames,
                     ctx.src_w,
                     ctx.src_h);
        } else {
            snprintf(status, sizeof(status),
                     "FPS:%4.1f | %s | no output | hits:%d lost:%d | frame:%dx%d",
                     fps,
                     state_to_string(state),
                     confirm_count,
                     lost_frames,
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

            int dzx = max((int)CENTER_DEAD_ZONE_X_MIN_PX, (int)round(w * CENTER_DEAD_ZONE_X_RATIO));
            int dzy = max((int)CENTER_DEAD_ZONE_Y_MIN_PX, (int)round(h * CENTER_DEAD_ZONE_Y_RATIO));

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
                     "Frame:%d | infer_stride:%d | PTZ:%s | Press C toggle PTZ, Q quit",
                     cam_id,
                     INFER_FRAME_STRIDE,
                     ptz_control_enabled.load() ? "ON" : "OFF");
            putText(img, info1, Point(12, 28), FONT_HERSHEY_SIMPLEX, 0.55,
                    ptz_control_enabled.load() ? Scalar(0, 255, 0) : Scalar(0, 255, 255),
                    2, LINE_AA);

            putText(img, text, Point(12, 58), FONT_HERSHEY_SIMPLEX, 0.50,
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
        } else if (key == ' ') {
            ptz_control_enabled.store(false);
            cout << "\n[KEY] PTZ auto tracking OFF" << endl;
        }

        this_thread::sleep_for(chrono::milliseconds(15));
    }

    destroyAllWindows();
}

void ptz_control_thread() {
    cout << "[PTZ] fast continuous control thread started. Press C to enable/disable." << endl;

    double last_cmd_time = 0.0;
    int last_pan = 0;
    int last_tilt = 0;
    int print_count = 0;

    auto send_stop_if_needed = [&]() {
        if (last_pan != 0 || last_tilt != 0) {
            send_ptz_continuous(0, 0, 0);
            last_pan = 0;
            last_tilt = 0;
        }
    };

    while (is_running.load()) {
        if (!ptz_control_enabled.load()) {
            send_stop_if_needed();
            this_thread::sleep_for(chrono::milliseconds(60));
            continue;
        }

        double now = now_sec();
        if (now - last_cmd_time < PTZ_CONTROL_PERIOD_SEC) {
            this_thread::sleep_for(chrono::milliseconds(20));
            continue;
        }

        SharedDetection det;
        {
            lock_guard<mutex> lock(latest_det_mtx);
            det = latest_det;
        }

        // 只用已经确认 TRACK 的目标控制，避免候选误检直接带动云台。
        if (!det.valid || !det.confirmed || (now - det.ts) > PTZ_TARGET_MAX_AGE_SEC) {
            send_stop_if_needed();
            this_thread::sleep_for(chrono::milliseconds(40));
            continue;
        }

        int fw = max(1, det.frame_w);
        int fh = max(1, det.frame_h);

        float dx = det.cx - fw / 2.0f;
        float dy = det.cy - fh / 2.0f;

        float dead_x = max(CENTER_DEAD_ZONE_X_MIN_PX, fw * CENTER_DEAD_ZONE_X_RATIO);
        float dead_y = max(CENTER_DEAD_ZONE_Y_MIN_PX, fh * CENTER_DEAD_ZONE_Y_RATIO);

        int pan_user = 0;
        int tilt_user = 0;

        if (fabs(dx) > dead_x) {
            float usable = max(1.0f, fw * 0.5f - dead_x);
            float e = clampf((fabs(dx) - dead_x) / usable, 0.0f, 1.0f);
            int speed = (int)round(PTZ_MIN_SPEED + e * (PTZ_MAX_SPEED - PTZ_MIN_SPEED));
            pan_user = (dx > 0) ? speed : -speed;
        }

        if (fabs(dy) > dead_y) {
            float usable = max(1.0f, fh * 0.5f - dead_y);
            float e = clampf((fabs(dy) - dead_y) / usable, 0.0f, 1.0f);
            int speed = (int)round(PTZ_MIN_SPEED + e * (PTZ_MAX_SPEED - PTZ_MIN_SPEED));
            tilt_user = (dy > 0) ? speed : -speed;
        }

        // 高倍率时速度自动压低，减少过冲。
        // 这里不每次查询 PTZ 状态，保持快速响应；仅按检测框大小做粗略降速。
        float box_side_ratio = max(det.w / fw, det.h / fh);
        if (box_side_ratio > 0.18f) {
            pan_user = max(-PTZ_MAX_SPEED_HIGH_ZOOM, min(PTZ_MAX_SPEED_HIGH_ZOOM, pan_user));
            tilt_user = max(-PTZ_MAX_SPEED_HIGH_ZOOM, min(PTZ_MAX_SPEED_HIGH_ZOOM, tilt_user));
        }

        bool ok = true;
        if (pan_user == 0 && tilt_user == 0) {
            send_stop_if_needed();
        } else {
            ok = send_ptz_continuous(pan_user, tilt_user, 0);
            last_pan = pan_user;
            last_tilt = tilt_user;
        }

        last_cmd_time = now_sec();

        print_count++;
        if (print_count % 6 == 1 || !ok) {
            cout << "\n[PTZ-CONT] " << (ok ? "OK" : "FAIL")
                 << " dx=" << dx << " dy=" << dy
                 << " dead=(" << dead_x << "," << dead_y << ")"
                 << " speed=(" << pan_user << "," << tilt_user << ")"
                 << " conf=" << det.conf
                 << endl;
        }

        this_thread::sleep_for(chrono::milliseconds(15));
    }

    send_ptz_continuous(0, 0, 0);
}

// ============================================================
// 10. 主程序
// ============================================================



void signal_handler(int) {
    is_running.store(false);
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    string rtsp_url = DEFAULT_RTSP_URL;
    if (argc >= 2) {
        rtsp_url = argv[1];
    }
    if (rtsp_url.empty()) {
        cerr << "❌ 未设置 RTSP 输入源。请设置 ANTI_UAV_RTSP_URL，或用第一个启动参数传入完整地址。" << endl;
        return 2;
    }

    cout << "\n🚀 启动海康球机 RKNN YOLO 无人机识别程序" << endl;
    cout << "✅ 已取消环境适应学习 / 死物黑名单" << endl;
    cout << "✅ 已加入 Candidate/Confirmed/Lost 三状态，低置信误检不会进入 TRACK" << endl;
    cout << "✅ FP 模型低频推理，显示线程与推理解耦，按 C 开/关云台粗闭环" << endl;
    cout << "✅ 模型：" << MODEL_PATH << endl;
    cout << "✅ 输入源：" << rtsp_url << endl;
    cout << "✅ 输出 UDP：" << UDP_IP << ":" << UDP_PORT << endl;
    cout << "✅ NPU：Core 0 单线程 FP 推理，Core 2 保留给音频 YAMNet" << endl;
    cout << "按 q 或 Ctrl+C 退出。\n" << endl;

    rknn_context ctx_rknn;
    int ret = rknn_init(&ctx_rknn, (void*)MODEL_PATH.c_str(), 0, 0, NULL);
    if (ret < 0) {
        cerr << "❌ RKNPU 初始化失败，ret=" << ret << endl;
        return -1;
    }
    rknn_set_core_mask(ctx_rknn, RKNN_NPU_CORE_0);

    thread t_capture(capture_thread, rtsp_url);
    thread t_infer0(inference_thread, 0, ctx_rknn);
    thread t_post(postprocess_thread);
    thread t_display(display_thread);
    thread t_ptz(ptz_control_thread);

    t_capture.join();
    t_infer0.join();
    t_post.join();
    t_display.join();
    t_ptz.join();

    rknn_destroy(ctx_rknn);

    cout << "\n[MAIN] exited safely" << endl;
    return 0;
}
