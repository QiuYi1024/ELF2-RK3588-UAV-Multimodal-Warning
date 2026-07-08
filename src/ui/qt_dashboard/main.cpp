#include <QApplication>
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QLineEdit>
#include <QProgressBar>
#include <QSlider>
#include <QSpinBox>
#include <QScrollArea>
#include <QSizePolicy>
#include <QUdpSocket>
#include <QTcpSocket>
#include <QThread>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QByteArray>
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QDateTime>
#include <QKeyEvent>
#include <QMutex>
#include <QMutexLocker>
#include <QNetworkProxy>
#include <QNetworkProxyFactory>
#include <QPixmap>
#include <QSet>
#include <QHostAddress>
#include <QTimer>
#include <QUrl>
#include <QStorageInfo>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QStringList>
#include <QFont>
#include <QMouseEvent>
#include <QScreen>
#include <QStackedWidget>
#include <QButtonGroup>
#include <QTableWidget>
#include <QHeaderView>
#include <QHash>
#include <QProcess>
#include <QCloseEvent>
#include <QMessageBox>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <algorithm>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#endif

#include "VideoWidget.h"
#include "RadarWidget.h"
#include "DataManager.h"

// ============================================================
// 1. 海康威视视频源配置
// ============================================================
// 默认不内置摄像头凭据；通过 ANTI_UAV_QT_RTSP_URL 或 ANTI_UAV_RTSP_URL 配置视频源。
static const std::string HIKVISION_RTSP_URL = "";

// 你当前代码里要求这两个参数：
// continuous 控制方向留在识别/控制程序中；Qt 只负责显示。
// 图像显示方向这里单独控制。
static const bool ROTATE_IMAGE_180 = false;

// UDP 端口：视觉识别程序发送检测框到 5005；音频雷达发送角度到 5006；融合事件监听 5007。
static const quint16 VISION_UDP_PORT = 5005;
static const quint16 AUDIO_UDP_PORT  = 5006;
static const quint16 FUSION_UDP_PORT = 5007;
static const quint16 RID900_UDP_PORT = 5009;

// ============================================================
// 2. 全局共享状态定义
// ============================================================

SensorMeta g_metaRingBuffer[META_BUFFER_SIZE];
std::atomic<int> g_yolo_max_frame{-1};

std::shared_ptr<cv::Mat> g_latestVideoFrame = nullptr;     // RGB Mat
std::shared_ptr<SensorMeta> g_latestSensorMeta = nullptr;
std::shared_ptr<VideoFrameTiming> g_latestVideoTiming = nullptr;
std::atomic<int> g_visionOverlayMaxAgeMs{250};
std::atomic<int> g_frameStreamBacklogBytes{0};
std::atomic<int> g_frameStreamDroppedOldFrames{0};
std::atomic<int> g_frameStreamDecodedFrames{0};
std::atomic<int> g_frameStreamPublishedFrames{0};
std::atomic<int> g_frameStreamLastDisplayFrameId{-1};
std::atomic<int> g_frameStreamLastRecordFrameId{-1};
std::atomic<int> g_frameStreamRecordingQueueSize{0};
std::atomic<int> g_frameStreamRecordingDroppedFrames{0};
std::atomic<int> g_frameStreamRecordingWriteMs{-1};

std::atomic<bool> g_hasVideoSignal{false};
std::atomic<int> g_videoWidth{0};
std::atomic<int> g_videoHeight{0};
std::atomic<bool> g_expectRkFrameStream{false};
std::atomic<bool> g_videoUsingRkStream{false};
static std::atomic<bool> g_preferFrameStreamMeta{false};

std::atomic<float> g_latestAudioAngle{0.0f};
std::atomic<bool> g_hasAudioSignal{false};

std::atomic<qint64> g_lastVideoFrameMs{-1};
std::atomic<qint64> g_lastVisionRecvMs{-1};
std::atomic<qint64> g_lastAudioRecvMs{-1};

std::atomic<bool> g_latestVisionDetected{false};
std::atomic<bool> g_latestVisionTracking{false};
std::atomic<double> g_latestVisionConf{0.0};
std::atomic<double> g_latestVisionFps{0.0};
std::atomic<double> g_latestVisionInferMs{0.0};

std::atomic<bool> g_latestAudioDetected{false};
std::atomic<bool> g_latestAudioDoaValid{false};
std::atomic<bool> g_latestAudioStableDoa{false};
std::atomic<double> g_latestAudioScore{0.0};
std::atomic<double> g_latestAudioScoreEma{0.0};
std::atomic<double> g_latestAudioDoaStability{0.0};
std::atomic<double> g_latestAudioDoaConfidence{0.0};
std::atomic<double> g_latestAudioRmsDbfs{-120.0};

std::atomic<bool> g_latestAudioGuided{false};
std::atomic<bool> g_audioGuidanceRequested{false};
std::atomic<double> g_latestCameraAzimuthDeg{99999.0};
std::atomic<int> g_latestCameraAzimuthUnit{0};
std::atomic<int> g_latestCameraElevationUnit{0};
std::atomic<int> g_latestCameraAbsoluteZoom{10};
std::atomic<bool> g_latestAutoTrackingEnabled{true};
std::atomic<bool> g_latestAutoZoomEnabled{true};
std::atomic<double> g_latestZoomRatio{-1.0};
std::atomic<double> g_latestZoomSpeed{0.0};
std::atomic<qint64> g_latestFocusRequestMs{0};
std::atomic<qint64> g_latestFocusCooldownRemainingMs{0};
std::atomic<double> g_latestVisualAzimuthDeg{99999.0};
std::atomic<double> g_latestAudioWorldAzimuthDeg{99999.0};
std::atomic<double> g_latestAudioPanErrDeg{99999.0};
std::atomic<double> g_latestAudioVisualErrDeg{99999.0};
std::atomic<double> g_latestMicOffsetDeg{0.0};
std::atomic<double> g_latestOffsetConfidence{0.0};
std::atomic<int> g_latestCalibrationSamples{0};

static QMutex g_dashboardTextMutex;
static QString g_latestFusionMode = "IDLE";
static QString g_latestFusionState = "IDLE";
static QString g_latestControlSource = "none";
static qint64 g_qtControlSeq = 0;
static QString g_latestSearchState = "none";
static QString g_latestFocusState = "DISABLED";
static QString g_latestOffsetConfidenceLabel = "low";
static QString g_latestAudioGuidanceState = "disabled";
static QString g_latestAudioRejectReason = "none";
static QString g_latestAudioCalibrationState = "uncalibrated";
static QString g_latestAudioState = "IDLE";
static QString g_latestCameraProfile = "--";
static QString g_latestImageQualityWarning = "OK";
static QString g_latestProfileSwitchReason = "--";
std::atomic<double> g_latestVerticalSweepDeltaDeg{0.0};
std::atomic<double> g_latestBrightnessMean{-1.0};
std::atomic<double> g_latestContrastStd{-1.0};
std::atomic<double> g_latestSaturationMean{-1.0};
std::atomic<double> g_latestBlurLaplacianVar{-1.0};
std::atomic<bool> g_latestGrayscaleLike{false};

std::atomic<bool> g_recordingRequested{false};
std::atomic<bool> g_recordingActive{false};
std::atomic<bool> g_recordingPaused{false};
static QMutex g_recordingPathMutex;
static QString g_currentRecordingPath;
static QString g_lastRecordingPath;
std::atomic<bool> g_rawRecordingRequested{false};
std::atomic<bool> g_rawRecordingActive{false};
static QMutex g_rawRecordingPathMutex;
static QString g_currentRawRecordingPath;
std::atomic<bool> g_detectionEnabledRequested{true};
std::atomic<bool> g_detectionEnabledReported{true};
std::atomic<bool> g_audioRawRecordingRequested{false};
std::atomic<bool> g_audioRawRecordingActive{false};
static QMutex g_audioRawRecordingPathMutex;
static QString g_currentAudioRawRecordingPath;
static QString g_currentAudioRawRecordingSessionId;
std::atomic<bool> g_fusedRecordingRequested{false};
std::atomic<bool> g_fusedRecordingActive{false};
std::atomic<bool> g_displayEcoMode{false};
std::atomic<double> g_latestSystemCpuPercent{-1.0};
std::atomic<double> g_latestSystemMemPercent{-1.0};
std::atomic<double> g_latestNpuLoadPercent{-1.0};
std::atomic<double> g_latestNpuFreqMhz{-1.0};
std::atomic<double> g_latestMaxTemperatureC{-1.0};
static QMutex g_captureSessionMutex;
static QString g_currentCaptureSessionId;
static QString g_currentCaptureManifestPath;
static QString g_currentRawRecordingSessionId;
static qint64 g_captureSessionStartWallMs = -1;
static qint64 g_captureSessionStartMonoNs = -1;
static QMutex g_audioSenderMutex;
static QString g_latestAudioSenderHost;
static std::atomic<qint64> g_lastAudioControlRetryMs{0};
static QMutex g_qtControlStatusMutex;
static QString g_lastQtPtzCmd = "--";
static QString g_lastQtCameraModeCmd = "--";

struct RidTargetState {
    QString serialNumber;
    QString vendor;
    QString productType;
    int rssiDbm = 0;
    int snrDb = 0;
    double longitude = 0.0;
    double latitude = 0.0;
    double heightM = 0.0;
    double speedMps = 0.0;
    int yawDeg = 0;
    qint64 lastSeenMs = 0;
};

static QMutex g_ridMutex;
static QHash<QString, RidTargetState> g_ridTargets;
static QString g_ridStatusText = "等待 RID900";
static QString g_ridSource = "--";
static QString g_ridDeviceId = "--";
static QString g_ridModulePosition = "--";
static std::atomic<bool> g_ridConnected{false};
static std::atomic<qint64> g_lastRidRecvMs{-1};
static std::atomic<qint64> g_lastRidDataMs{-1};
static std::atomic<qint64> g_lastWindowsSyncMs{-1};
static QMutex g_syncMutex;
static QString g_windowsSyncHost;
static QString g_windowsLaunchStatus = "等待操作";
static QString g_windowsRecordingStatus = "未录制";
static std::atomic<bool> g_windowsRemoteRecordingRequested{false};
static std::atomic<bool> g_windowsAutoReconnect{true};
static QMutex g_syncTelemetryMutex;
static QHash<quint16, QByteArray> g_syncLatestTelemetry;

static bool dashboardIsRecent(qint64 timestampMs, qint64 nowMs, qint64 maxAgeMs);

static qint64 qtMonotonicMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void setDashboardText(QString* target, const QString& value) {
    QMutexLocker locker(&g_dashboardTextMutex);
    *target = value;
}

static QString dashboardText(const QString& value) {
    QMutexLocker locker(&g_dashboardTextMutex);
    return value;
}

static void setQtPtzStatusText(const QString& value) {
    QMutexLocker locker(&g_qtControlStatusMutex);
    g_lastQtPtzCmd = value;
}

static void setQtCameraModeStatusText(const QString& value) {
    QMutexLocker locker(&g_qtControlStatusMutex);
    g_lastQtCameraModeCmd = value;
}

static QString qtPtzStatusText() {
    QMutexLocker locker(&g_qtControlStatusMutex);
    return g_lastQtPtzCmd;
}

static QString qtCameraModeStatusText() {
    QMutexLocker locker(&g_qtControlStatusMutex);
    return g_lastQtCameraModeCmd;
}

static void setCurrentRecordingPath(const QString& path) {
    QMutexLocker locker(&g_recordingPathMutex);
    g_currentRecordingPath = path;
}

static QString currentRecordingPath() {
    QMutexLocker locker(&g_recordingPathMutex);
    return g_currentRecordingPath;
}

static void setLastRecordingPath(const QString& path) {
    if (path.trimmed().isEmpty()) return;
    QMutexLocker locker(&g_recordingPathMutex);
    g_lastRecordingPath = path;
}

static QString lastRecordingPath() {
    QMutexLocker locker(&g_recordingPathMutex);
    return g_lastRecordingPath;
}

static void setCurrentRawRecordingPath(const QString& path) {
    QMutexLocker locker(&g_rawRecordingPathMutex);
    g_currentRawRecordingPath = path;
}

static QString currentRawRecordingPath() {
    QMutexLocker locker(&g_rawRecordingPathMutex);
    return g_currentRawRecordingPath;
}

static void setCurrentAudioRawRecordingPath(const QString& path) {
    QMutexLocker locker(&g_audioRawRecordingPathMutex);
    g_currentAudioRawRecordingPath = path;
}

static QString currentAudioRawRecordingPath() {
    QMutexLocker locker(&g_audioRawRecordingPathMutex);
    return g_currentAudioRawRecordingPath;
}

static void setCurrentAudioRawRecordingSessionId(const QString& sessionId) {
    QMutexLocker locker(&g_audioRawRecordingPathMutex);
    g_currentAudioRawRecordingSessionId = sessionId;
}

static QString currentAudioRawRecordingSessionId() {
    QMutexLocker locker(&g_audioRawRecordingPathMutex);
    return g_currentAudioRawRecordingSessionId;
}

static void setCurrentCaptureSession(const QString& sessionId, const QString& manifestPath) {
    QMutexLocker locker(&g_captureSessionMutex);
    g_currentCaptureSessionId = sessionId;
    g_currentCaptureManifestPath = manifestPath;
}

static QString currentCaptureSessionId() {
    QMutexLocker locker(&g_captureSessionMutex);
    return g_currentCaptureSessionId;
}

static QString currentCaptureManifestPath() {
    QMutexLocker locker(&g_captureSessionMutex);
    return g_currentCaptureManifestPath;
}

static void setCurrentRawRecordingSessionId(const QString& sessionId) {
    QMutexLocker locker(&g_captureSessionMutex);
    g_currentRawRecordingSessionId = sessionId;
}

static QString currentRawRecordingSessionId() {
    QMutexLocker locker(&g_captureSessionMutex);
    return g_currentRawRecordingSessionId;
}

static void setLatestAudioSenderHost(const QHostAddress& sender) {
    if (sender.isNull()) return;
    QString host = sender.toString().trimmed();
    if (host.startsWith("::ffff:")) {
        host = host.mid(QString("::ffff:").size());
    }
    if (host.isEmpty() || host == "0.0.0.0" || host == "::") return;

    QMutexLocker locker(&g_audioSenderMutex);
    g_latestAudioSenderHost = host;
}

static QString latestAudioSenderHost() {
    QMutexLocker locker(&g_audioSenderMutex);
    return g_latestAudioSenderHost;
}

static QString sanitizePathSegment(QString value, const QString& fallback) {
    value = value.trimmed();
    if (value.isEmpty()) value = fallback;
    for (QChar& ch : value) {
        if (!(ch.isLetterOrNumber() || ch == '_' || ch == '-')) {
            ch = '_';
        }
    }
    return value.isEmpty() ? fallback : value;
}

static QString antiUavDataRootPath() {
    QString root = QString::fromUtf8(qgetenv("ANTI_UAV_DATA_ROOT")).trimmed();
    if (root.isEmpty()) {
        root = QString::fromUtf8(qgetenv("ANTIUAV_DATA_ROOT")).trimmed();
    }
    if (root.isEmpty()) {
#ifdef Q_OS_WIN
        root = "D:/codex/ELF2/AntiUAV_Data";
#else
        root = "/home/elf/AntiUAV_Data";
#endif
    }
    QDir().mkpath(root);
    return root;
}

static void ensureDataRootSkeleton(const QString& root) {
    const QStringList dirs = {
        "sessions", "incoming", "exports", "cache", "temp", "diagnostics",
        "diagnostics/runtime", "diagnostics/legacy_unclassified", "transferred", "index",
        "runtime", "runtime/pid", "runtime/status", "runtime/logs"
    };
    for (const QString& dir : dirs) {
        QDir().mkpath(root + "/" + dir);
    }
    const QString sessionsIndex = root + "/index/sessions_index.json";
    if (!QFile::exists(sessionsIndex)) {
        QFile f(sessionsIndex);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            f.write("{\n  \"created_by\": \"Codex\",\n  \"sessions\": []\n}\n");
        }
    }
    const QString transferIndex = root + "/index/transfer_index.json";
    if (!QFile::exists(transferIndex)) {
        QFile f(transferIndex);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            f.write("{\n  \"created_by\": \"Codex\",\n  \"transfers\": []\n}\n");
        }
    }
}

static QString captureSessionType() {
    const QString type = QString::fromUtf8(qgetenv("ANTI_UAV_SESSION_TYPE")).trimmed();
#ifdef Q_OS_WIN
    const QString fallback = "windows_only";
#else
    const QString fallback = "realtime_camera";
#endif
    return sanitizePathSegment(type.isEmpty() ? fallback : type, fallback);
}

static QString sessionRootDirPath(const QString& date, const QString& preferredSessionId = QString()) {
    const QString root = antiUavDataRootPath();
    ensureDataRootSkeleton(root);
    QString sessionId = preferredSessionId.trimmed();
    if (sessionId.isEmpty()) {
        sessionId = currentCaptureSessionId();
    }
    if (sessionId.isEmpty()) {
        sessionId = QString::fromUtf8(qgetenv("ANTI_UAV_SESSION_ID")).trimmed();
    }
    if (sessionId.isEmpty()) {
        sessionId = date + "_" + captureSessionType() + "_S0001";
    }
    sessionId = sanitizePathSegment(sessionId, date + "_realtime_camera_S0001");
    const QString dir = root + "/sessions/" + sessionId;
    const QStringList subdirs = {
        "video/raw", "video/annotated", "video/thumbnails",
        "audio/raw_6ch", "audio/selected_channel", "audio/features",
        "windows_recordings",
        "screenshots", "logs/system", "logs/yolo", "logs/audio", "logs/qt",
        "logs/ptz", "logs/rid900", "logs/transfer",
        "metadata", "config"
    };
    for (const QString& subdir : subdirs) {
        QDir().mkpath(dir + "/" + subdir);
    }
    return dir;
}

static QString recordingDirPath(const QString& date = QDateTime::currentDateTime().toString("yyyyMMdd")) {
    const QString dir = sessionRootDirPath(date) + "/video/annotated";
    QDir().mkpath(dir);
    return dir;
}

static QString rawVideoDirPath(const QString& date = QDateTime::currentDateTime().toString("yyyyMMdd")) {
    const QString dir = sessionRootDirPath(date) + "/video/raw";
    QDir().mkpath(dir);
    return dir;
}

static QString windowsRecordingDirPath(const QString& date = QDateTime::currentDateTime().toString("yyyyMMdd")) {
    const QString root = QString::fromUtf8(qgetenv("ANTI_UAV_WINDOWS_RECORDING_ROOT")).trimmed();
#ifdef Q_OS_WIN
    const QString dataRoot = antiUavDataRootPath();
    const QString base = root.isEmpty() ? dataRoot + "/windows_recordings" : root;
    const QDate parsedDate = QDate::fromString(date, "yyyyMMdd");
    const QDate day = parsedDate.isValid() ? parsedDate : QDate::currentDate();
    QString sessionId = currentCaptureSessionId();
    if (sessionId.isEmpty()) {
        sessionId = QString::fromUtf8(qgetenv("ANTI_UAV_SESSION_ID")).trimmed();
    }
    if (sessionId.isEmpty()) {
        sessionId = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + "_S0001";
    }
    const QString dir = QString("%1/%2/%3/%4/%5")
        .arg(base,
             day.toString("yyyy"),
             day.toString("yyyy-MM"),
             day.toString("yyyy-MM-dd"),
             sanitizePathSegment(sessionId, day.toString("yyyyMMdd") + "_S0001"));
#else
    const QString dir = sessionRootDirPath(date) + "/windows_recordings";
#endif
    QDir().mkpath(dir);
    return dir;
}

static QString windowsRecordingIndexPath() {
#ifdef Q_OS_WIN
    const QString dir = antiUavDataRootPath() + "/index";
    QDir().mkpath(dir);
    return dir + "/recordings_index.jsonl";
#else
    return windowsRecordingDirPath() + "/recordings_index.jsonl";
#endif
}

static QString sessionManifestPath(const QString& date, const QString& sessionId) {
    return sessionRootDirPath(date, sessionId) + "/manifest.json";
}

static qint64 metricsNowMs();
static qint64 monotonicNowNs();

static QString sessionSummaryPath(const QString& date, const QString& sessionId) {
    return sessionRootDirPath(date, sessionId) + "/session_summary.json";
}

static QString recordingTimelinePath(const QString& date, const QString& sessionId) {
    return sessionRootDirPath(date, sessionId) + "/metadata/recording_timeline.jsonl";
}

static QString ensureCaptureSessionForRecording(const QString& preferredSessionId = QString()) {
    const QDateTime now = QDateTime::currentDateTime();
    const QString date = now.toString("yyyyMMdd");
    QString sessionId = preferredSessionId.trimmed();
    if (sessionId.isEmpty()) {
        sessionId = currentCaptureSessionId();
    }
    if (sessionId.isEmpty()) {
        sessionId = QString::fromUtf8(qgetenv("ANTI_UAV_SESSION_ID")).trimmed();
    }
    if (sessionId.isEmpty()) {
        sessionId = now.toString("yyyyMMdd_HHmmss")
#ifdef Q_OS_WIN
            + "_S0001";
#else
            + "_" + captureSessionType() + "_S0001";
#endif
    }
    sessionId = sanitizePathSegment(sessionId,
#ifdef Q_OS_WIN
                                    date + "_S0001"
#else
                                    date + "_" + captureSessionType() + "_S0001"
#endif
                                    );
    const QString previousSessionId = currentCaptureSessionId();
    setCurrentCaptureSession(sessionId, sessionManifestPath(date, sessionId));
    if (g_captureSessionStartWallMs <= 0 || previousSessionId != sessionId) {
        g_captureSessionStartWallMs = metricsNowMs();
        g_captureSessionStartMonoNs = monotonicNowNs();
    }
    return sessionId;
}

static QString diagnosticsRunDirPath(const QString& stamp) {
    const QString root = antiUavDataRootPath();
    ensureDataRootSkeleton(root);
    const QString envSessionId = QString::fromUtf8(qgetenv("ANTI_UAV_SESSION_ID")).trimmed();
    const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
    const QString dir = envSessionId.isEmpty()
        ? root + "/diagnostics/runtime/qt/metrics_logs/run_" + stamp
        : sessionRootDirPath(date, envSessionId) + "/logs/qt/metrics_logs/run_" + stamp;
    QDir().mkpath(dir);
    return dir;
}

static int envIntBounded(const char* name, int defaultValue, int minValue, int maxValue) {
    const QByteArray raw = qgetenv(name).trimmed();
    if (raw.isEmpty()) {
        return defaultValue;
    }
    bool ok = false;
    int value = QString::fromUtf8(raw).toInt(&ok);
    if (!ok) {
        return defaultValue;
    }
    return qBound(minValue, value, maxValue);
}

static std::string envStringOrDefault(const char* name, const std::string& defaultValue) {
    const QByteArray raw = qgetenv(name).trimmed();
    if (raw.isEmpty()) {
        return defaultValue;
    }
    return QString::fromUtf8(raw).toStdString();
}

static QString antiUavNodeRole();
static QString defaultVideoSourceForThisNode();

struct WindowsRecordingFilePlan {
    QString recordId;
    QString sessionId;
    QString source;
    QString path;
    QDateTime startTime;
    int counter = 1;
};

static QString windowsRecordingSourceLabel() {
    QString source = QString::fromUtf8(qgetenv("ANTI_UAV_QT_VIDEO_SOURCE")).trimmed().toLower();
    if (source.isEmpty()) {
        source = defaultVideoSourceForThisNode().toLower();
    }
    if (source == "auto" || source == "rk_stream" || source == "rk" || source == "frame_stream") {
        return QString::number(envIntBounded("ANTI_UAV_FRAME_STREAM_PORT", 5010, 1, 65535));
    }
    if (source == "rtsp") return "rtsp";
    if (source == "fallback" || source == "test" || source == "sim") return "fallback";
    return sanitizePathSegment(source, "source");
}

static int nextWindowsRecordingCounter(const QString& dir,
                                       const QString& sessionId,
                                       const QString& source) {
    int maxCounter = 0;
    const QString prefix = QString("qt_record_%1_").arg(sessionId);
    const QString marker = QString("_%1_").arg(source);
    QDir recordingDir(dir);
    const QStringList files = recordingDir.entryList(
        QStringList() << "qt_record_*.mp4" << "qt_record_*.avi",
        QDir::Files,
        QDir::Name);
    for (const QString& fileName : files) {
        if (!fileName.startsWith(prefix) || !fileName.contains(marker)) continue;
        const QString baseName = QFileInfo(fileName).completeBaseName();
        const int lastUnderscore = baseName.lastIndexOf('_');
        if (lastUnderscore < 0) continue;
        bool ok = false;
        const int value = baseName.mid(lastUnderscore + 1).toInt(&ok);
        if (ok) maxCounter = std::max(maxCounter, value);
    }
    return maxCounter + 1;
}

static QString makeUniquePath(const QString& dir,
                              const QString& stem,
                              const QString& extension,
                              int startCounter,
                              int* selectedCounter = nullptr) {
    QDir().mkpath(dir);
    int counter = qMax(1, startCounter);
    for (int attempts = 0; attempts < 9999; ++attempts, ++counter) {
        const QString counterText = QString("%1").arg(counter, 4, 10, QChar('0'));
        const QString path = QString("%1/%2_%3.%4").arg(dir, stem, counterText, extension);
        if (!QFileInfo::exists(path)) {
            if (selectedCounter) {
                *selectedCounter = counter;
            }
            return path;
        }
    }
    qWarning() << "[REC] no unique recording path available in" << dir << "for" << stem;
    return QString();
}

static WindowsRecordingFilePlan makeWindowsRecordingFilePlan(const QString& extension,
                                                              const WindowsRecordingFilePlan* previousPlan = nullptr) {
    WindowsRecordingFilePlan plan;
    plan.startTime = previousPlan ? previousPlan->startTime : QDateTime::currentDateTime();
    plan.sessionId = previousPlan && !previousPlan->sessionId.isEmpty()
        ? previousPlan->sessionId
        : ensureCaptureSessionForRecording();
    plan.source = previousPlan && !previousPlan->source.isEmpty()
        ? previousPlan->source
        : windowsRecordingSourceLabel();
    const QString date = plan.startTime.toString("yyyyMMdd");
    const QString dir = windowsRecordingDirPath(date);
    plan.counter = previousPlan ? previousPlan->counter : nextWindowsRecordingCounter(dir, plan.sessionId, plan.source);
    const QString stamp = plan.startTime.toString("yyyyMMdd_HHmmss_zzz");

    int selectedCounter = plan.counter;
    const QString stem = QString("qt_record_%1_%2_%3").arg(plan.sessionId, stamp, plan.source);
    plan.path = makeUniquePath(dir, stem, extension, plan.counter, &selectedCounter);
    if (!plan.path.isEmpty()) {
        plan.counter = selectedCounter;
        const QString counterText = QString("%1").arg(plan.counter, 4, 10, QChar('0'));
        plan.recordId = QString("%1_%2").arg(stamp, counterText);
    }
    return plan;
}

static void appendWindowsRecordingIndexEvent(const QString& event,
                                             const QString& recordId,
                                             const QString& sessionId,
                                             const QString& source,
                                             const QDateTime& startTime,
                                             const QDateTime& endTime,
                                             const QString& filePath,
                                             const QString& codec,
                                             const cv::Size& size,
                                             int frames,
                                             const QString& endReason) {
    if (antiUavNodeRole() != "windows") return;
    QFile f(windowsRecordingIndexPath());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) return;

    QJsonObject obj;
    obj["event"] = event;
    obj["record_id"] = recordId;
    obj["session_id"] = sessionId;
    obj["source"] = source;
    obj["start_time"] = startTime.isValid() ? startTime.toString("yyyy-MM-dd HH:mm:ss.zzz") : "";
    obj["end_time"] = endTime.isValid() ? endTime.toString("yyyy-MM-dd HH:mm:ss.zzz") : "";
    obj["file_path"] = filePath;
    obj["codec"] = codec;
    obj["width"] = size.width;
    obj["height"] = size.height;
    obj["fps"] = 25;
    obj["frame_count"] = frames;
    obj["dropped_recording_frames"] = g_frameStreamRecordingDroppedFrames.load();
    obj["end_reason"] = endReason;
    obj["bytes"] = static_cast<double>(QFileInfo(filePath).exists() ? QFileInfo(filePath).size() : 0);
    obj["ts"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    f.write("\n");
}

static int displayFrameIntervalMs() {
    if (g_displayEcoMode.load()) {
        return envIntBounded("ANTI_UAV_QT_LIGHT_DISPLAY_INTERVAL_MS", 40, 16, 66);
    }
    return envIntBounded("ANTI_UAV_QT_DISPLAY_INTERVAL_MS", 40, 16, 120);
}

static bool shouldPublishDisplayFrame(qint64 nowMs, qint64& lastPublishMs) {
    const int intervalMs = displayFrameIntervalMs();
    if (lastPublishMs <= 0 || nowMs - lastPublishMs >= intervalMs) {
        lastPublishMs = nowMs;
        return true;
    }
    return false;
}

// ============================================================
// 3. 小工具：兼容不同 JSON 字段名
// ============================================================

static int jsonIntAny(const QJsonObject& obj,
                      std::initializer_list<const char*> keys,
                      int defaultValue = 0) {
    for (const char* k : keys) {
        if (obj.contains(k)) return obj.value(k).toInt(defaultValue);
    }
    return defaultValue;
}

static double jsonDoubleAny(const QJsonObject& obj,
                            std::initializer_list<const char*> keys,
                            double defaultValue = 0.0) {
    for (const char* k : keys) {
        if (obj.contains(k)) return obj.value(k).toDouble(defaultValue);
    }
    return defaultValue;
}

static bool jsonBoolAny(const QJsonObject& obj,
                        std::initializer_list<const char*> keys,
                        bool defaultValue = false) {
    for (const char* k : keys) {
        if (obj.contains(k)) return obj.value(k).toBool(defaultValue);
    }
    return defaultValue;
}


// ============================================================
// METRICS LOGGER PATCH BEGIN
// Qt 实验数据记录中心：vision/audio/system/fusion CSV
// ============================================================

static QString metricsCsvEscape(QString s) {
    s.replace("\"", "\"\"");
    return "\"" + s + "\"";
}

static QString metricsEnvString(const char* key) {
    return qEnvironmentVariable(key);
}

static qint64 metricsNowMs() {
    return QDateTime::currentMSecsSinceEpoch();
}

static qint64 monotonicNowNs() {
    return static_cast<qint64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static QString antiUavNodeRole() {
    const QString role = QString::fromUtf8(qgetenv("ANTI_UAV_NODE_ROLE")).trimmed().toLower();
    if (!role.isEmpty()) return role;
#ifdef Q_OS_WIN
    return "windows";
#else
    return "elf2";
#endif
}

static std::string defaultBoardHostForThisNode() {
#ifdef Q_OS_WIN
    return "DEVICE_IP";
#else
    return "127.0.0.1";
#endif
}

static QString defaultVideoSourceForThisNode() {
#ifdef Q_OS_WIN
    return "rk_stream";
#else
    return "auto";
#endif
}

static QString antiUavYoloModelName() {
    QString model = QString::fromUtf8(qgetenv("ANTI_UAV_YOLO_MODEL")).trimmed();
    if (model.isEmpty()) {
        model = "models/best_uav_headless_i8.rknn";
    }
    return QFileInfo(model).fileName();
}

static qint64 dataDiskFreeBytes() {
    const QString root = antiUavDataRootPath();
    QDir().mkpath(root);
    QStorageInfo storage(root);
    storage.refresh();
    return storage.isValid() ? storage.bytesAvailable() : -1;
}

static QString formatBytesForUi(qint64 bytes) {
    if (bytes < 0) return "未知";
    const double mb = static_cast<double>(bytes) / 1024.0 / 1024.0;
    if (mb < 1024.0) return QString("%1 MB").arg(mb, 0, 'f', 0);
    return QString("%1 GB").arg(mb / 1024.0, 0, 'f', 1);
}

static QString dataDiskStatusText() {
    const qint64 freeBytes = dataDiskFreeBytes();
    QString status = QString("剩余 %1").arg(formatBytesForUi(freeBytes));
    if (freeBytes >= 0 && freeBytes < 100LL * 1024 * 1024) {
        status += " / 禁止完整会话";
    } else if (freeBytes >= 0 && freeBytes < 200LL * 1024 * 1024) {
        status += " / 禁止视频音频";
    } else if (freeBytes >= 0 && freeBytes < 500LL * 1024 * 1024) {
        status += " / 仅短测试";
    }
    return status;
}

static bool recordingStorageAllowed(const QString& feature, bool fullSession) {
    const qint64 freeBytes = dataDiskFreeBytes();
    QString reason;
    if (freeBytes >= 0 && freeBytes < 100LL * 1024 * 1024) {
        reason = "磁盘不足：剩余低于100MB，禁止完整会话";
    } else if (freeBytes >= 0 && freeBytes < 200LL * 1024 * 1024) {
        reason = "磁盘不足：剩余低于200MB，禁止视频/音频录制";
    } else if (freeBytes >= 0 && freeBytes < 500LL * 1024 * 1024 &&
               qgetenv("ANTI_UAV_ALLOW_SHORT_RECORDING").trimmed() != "1") {
        reason = "磁盘不足：剩余低于500MB，仅允许显式短测试";
    } else if (fullSession && freeBytes >= 0 && freeBytes < 500LL * 1024 * 1024) {
        reason = "磁盘不足：剩余低于500MB，禁止长会话";
    }
    if (reason.isEmpty()) return true;
    qWarning() << "[REC]" << feature << reason << "free=" << freeBytes;
    {
        QMutexLocker locker(&g_syncMutex);
        g_windowsLaunchStatus = feature + " " + reason;
        g_windowsRecordingStatus = "磁盘不足";
    }
    return false;
}

static void setWindowsSyncHost(const QString& host) {
    QMutexLocker locker(&g_syncMutex);
    g_windowsSyncHost = host;
}

static QString windowsSyncHost() {
    QMutexLocker locker(&g_syncMutex);
    return g_windowsSyncHost;
}

static void setWindowsLaunchStatus(const QString& status) {
    QMutexLocker locker(&g_syncMutex);
    g_windowsLaunchStatus = status;
}

static QString windowsLaunchStatus() {
    QMutexLocker locker(&g_syncMutex);
    return g_windowsLaunchStatus;
}

static void setWindowsRecordingStatus(const QString& status) {
    QMutexLocker locker(&g_syncMutex);
    g_windowsRecordingStatus = status;
}

static QString windowsRecordingStatus() {
    QMutexLocker locker(&g_syncMutex);
    return g_windowsRecordingStatus;
}

static void appendWindowsRecordingEvent(const QString& event, const QString& path) {
    if (antiUavNodeRole() != "windows") return;
    const QString dir = QFileInfo(path).absolutePath();
    QDir().mkpath(dir);
    QFile f(dir + "/windows_dashboard_events.jsonl");
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) return;
    QJsonObject obj;
    obj["type"] = event;
    obj["ts"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    obj["ts_ms"] = static_cast<double>(metricsNowMs());
    obj["path"] = path;
    obj["session_id"] = currentCaptureSessionId();
    obj["data_disk"] = dataDiskStatusText();
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    f.write("\n");
}

static void writeWindowsRecordingSummary(const QString& path, qint64 startMs, int frames, const QString& status) {
    if (antiUavNodeRole() != "windows") return;
    const QString dir = QFileInfo(path).absolutePath();
    QDir().mkpath(dir);
    QJsonObject root;
    QFileInfo info(path);
    root["created_by"] = "Codex";
    root["updated_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    root["status"] = status;
    root["session_id"] = currentCaptureSessionId();
    root["dashboard_path"] = path;
    root["bytes"] = static_cast<double>(info.exists() ? info.size() : 0);
    root["duration_ms"] = static_cast<double>(startMs > 0 ? metricsNowMs() - startMs : 0);
    root["frames"] = frames;
    root["data_root"] = antiUavDataRootPath();
    root["data_disk"] = dataDiskStatusText();
    QFile f(dir + "/windows_recording_summary.json");
    if (f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
}

static void mirrorTelemetryDatagram(QUdpSocket* socket,
                                    const QByteArray& datagram,
                                    quint16 port) {
    Q_UNUSED(socket);
    if (antiUavNodeRole() != "elf2") return;
    const qint64 lastSyncMs = g_lastWindowsSyncMs.load();
    if (lastSyncMs <= 0 || metricsNowMs() - lastSyncMs > 4000) return;
    QMutexLocker locker(&g_syncTelemetryMutex);
    g_syncLatestTelemetry.insert(port, datagram);
}

static QHash<quint16, QByteArray> takeLatestSyncTelemetry() {
    QMutexLocker locker(&g_syncTelemetryMutex);
    QHash<quint16, QByteArray> pending;
    pending.swap(g_syncLatestTelemetry);
    return pending;
}

static void sendAudioRawRecordingControlDatagram(const QString& host,
                                                 int port,
                                                 bool enabled,
                                                 const QString& recordId,
                                                 const QString& sessionId,
                                                 const QString& source) {
    if (host.trimmed().isEmpty()) return;

    QJsonObject obj;
    obj["type"] = "audio_control";
    obj["command"] = "set_raw_recording";
    obj["raw_recording"] = enabled ? 1 : 0;
    obj["record_id"] = recordId;
    obj["session_id"] = sessionId;
    obj["source"] = source;
    obj["ts_ms"] = static_cast<double>(metricsNowMs());

    QUdpSocket socket;
    const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
    const qint64 sent = socket.writeDatagram(payload, QHostAddress(host), port);
    if (sent < 0) {
        qWarning() << "[CTRL] audio raw retry failed" << host << port << socket.errorString();
    } else {
        qInfo() << "[CTRL] audio raw retry sent" << host << port << payload;
    }
}

static double metricsJsonDoubleAny(const QJsonObject& obj,
                                   std::initializer_list<const char*> keys,
                                   double defaultValue = 0.0) {
    for (const char* k : keys) {
        if (obj.contains(k)) return obj.value(k).toDouble(defaultValue);
    }
    return defaultValue;
}

static int metricsJsonIntAny(const QJsonObject& obj,
                             std::initializer_list<const char*> keys,
                             int defaultValue = 0) {
    for (const char* k : keys) {
        if (obj.contains(k)) {
            QJsonValue v = obj.value(k);
            if (v.isBool()) return v.toBool() ? 1 : 0;
            if (v.isDouble()) return v.toInt(defaultValue);
            if (v.isString()) {
                QString s = v.toString().trimmed().toLower();
                if (s == "true" || s == "yes" || s == "track" || s == "detected") return 1;
                if (s == "false" || s == "no" || s == "search" || s == "none") return 0;
                bool ok = false;
                int x = s.toInt(&ok);
                if (ok) return x;
            }
            return defaultValue;
        }
    }
    return defaultValue;
}

static QString metricsJsonStringAny(const QJsonObject& obj,
                                    std::initializer_list<const char*> keys,
                                    const QString& defaultValue = "") {
    for (const char* k : keys) {
        if (obj.contains(k)) return obj.value(k).toString(defaultValue);
    }
    return defaultValue;
}

static QString metricsReadText(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return "";
    return QString::fromUtf8(f.readAll()).trimmed();
}

struct ExperimentSelection {
    QString experimentId;
    QString sceneType;
    QString location;
    QString distanceM;
    QString noiseType;
    QString targetPresent;
    QString methodLabel;
    QString moduleLabel;
    QString trialId;
    QString powerW;
};

static QString experimentEnvOrDefault(const char* key, const QString& fallback = "") {
    const QString value = metricsEnvString(key);
    return value.isEmpty() ? fallback : value;
}

static ExperimentSelection defaultExperimentSelection() {
    ExperimentSelection s;
    s.experimentId = experimentEnvOrDefault("ANTI_UAV_EXPERIMENT_ID");
    s.sceneType = experimentEnvOrDefault("ANTI_UAV_SCENE_TYPE");
    s.location = experimentEnvOrDefault("ANTI_UAV_EXPERIMENT_LOCATION");
    s.distanceM = experimentEnvOrDefault("ANTI_UAV_DISTANCE_M");
    s.noiseType = experimentEnvOrDefault("ANTI_UAV_NOISE_TYPE");
    s.targetPresent = experimentEnvOrDefault("ANTI_UAV_TARGET_PRESENT");
    s.methodLabel = experimentEnvOrDefault("ANTI_UAV_EXPERIMENT_METHOD", "all_methods_auto");
    s.moduleLabel = experimentEnvOrDefault("ANTI_UAV_EXPERIMENT_MODULE", "auto_multimodal");
    s.trialId = experimentEnvOrDefault("ANTI_UAV_TRIAL_ID");
    s.powerW = experimentEnvOrDefault("ANTI_UAV_POWER_W");
    return s;
}

static QMutex& experimentSelectionMutex() {
    static QMutex m;
    return m;
}

static ExperimentSelection& experimentSelectionStorage() {
    static ExperimentSelection selection = defaultExperimentSelection();
    return selection;
}

static ExperimentSelection currentExperimentSelection() {
    QMutexLocker locker(&experimentSelectionMutex());
    return experimentSelectionStorage();
}

static void setCurrentExperimentSelection(const ExperimentSelection& selection) {
    QMutexLocker locker(&experimentSelectionMutex());
    experimentSelectionStorage() = selection;
}

static void exportExperimentSelectionToEnv(const ExperimentSelection& selection) {
    qputenv("ANTI_UAV_EXPERIMENT_ID", selection.experimentId.toUtf8());
    qputenv("ANTI_UAV_SCENE_TYPE", selection.sceneType.toUtf8());
    qputenv("ANTI_UAV_EXPERIMENT_LOCATION", selection.location.toUtf8());
    qputenv("ANTI_UAV_DISTANCE_M", selection.distanceM.toUtf8());
    qputenv("ANTI_UAV_NOISE_TYPE", selection.noiseType.toUtf8());
    qputenv("ANTI_UAV_TARGET_PRESENT", selection.targetPresent.toUtf8());
    qputenv("ANTI_UAV_EXPERIMENT_METHOD", selection.methodLabel.toUtf8());
    qputenv("ANTI_UAV_EXPERIMENT_MODULE", selection.moduleLabel.toUtf8());
    qputenv("ANTI_UAV_TRIAL_ID", selection.trialId.toUtf8());
    qputenv("ANTI_UAV_POWER_W", selection.powerW.toUtf8());
}

static void writeCaptureSessionManifest(const QString& state) {
    const QString path = currentCaptureManifestPath();
    if (path.isEmpty()) {
        return;
    }

    const ExperimentSelection selection = currentExperimentSelection();
    QJsonObject root;
    root["created_by"] = "Codex";
    root["updated_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    root["state"] = state;
    root["session_id"] = currentCaptureSessionId();
    root["experiment_id"] = selection.experimentId;
    root["scene_type"] = selection.sceneType;
    root["distance_m"] = selection.distanceM;
    root["noise_type"] = selection.noiseType;
    root["target_present"] = selection.targetPresent;
    root["trial_id"] = selection.trialId;
    root["method_label"] = selection.methodLabel;
    root["module_label"] = selection.moduleLabel;
    root["dashboard_recording_path"] = currentRecordingPath();
    root["raw_video_path"] = currentRawRecordingPath();
    root["raw_audio_path"] = currentAudioRawRecordingPath();
    root["windows_recording_dir"] = sessionRootDirPath(QDateTime::currentDateTime().toString("yyyyMMdd"), currentCaptureSessionId()) + "/windows_recordings";
    root["raw_video_requested"] = g_rawRecordingRequested.load();
    root["raw_audio_requested"] = g_audioRawRecordingRequested.load();
    root["dashboard_recording_requested"] = g_recordingRequested.load();
    root["windows_recording_requested"] = g_windowsRemoteRecordingRequested.load();
    root["session_start_wall_time"] = QDateTime::fromMSecsSinceEpoch(g_captureSessionStartWallMs).toString(Qt::ISODateWithMs);
    root["session_start_wall_time_ms"] = static_cast<double>(g_captureSessionStartWallMs);
    root["session_start_monotonic_ns"] = static_cast<double>(g_captureSessionStartMonoNs);
    root["recording_timeline"] = recordingTimelinePath(QDateTime::currentDateTime().toString("yyyyMMdd"), currentCaptureSessionId());
    root["ts_ms"] = static_cast<double>(metricsNowMs());
    root["note"] = "同步开始原始音视频采集只表示 RK 端原始视频、原始六通道音频、主通道音频和元数据共享 session_id、session_start_monotonic_ns 与 session_elapsed_ms；Windows本地录像单独控制。";

    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        f.close();
    }
}

static void appendRecordingTimelineEvent(const QString& event,
                                         const QString& source,
                                         const QString& path = QString(),
                                         const QJsonObject& extra = QJsonObject()) {
    const QString sessionId = currentCaptureSessionId();
    if (sessionId.isEmpty()) return;
    const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
    const qint64 nowWallMs = metricsNowMs();
    const qint64 nowMonoNs = monotonicNowNs();
    if (g_captureSessionStartWallMs <= 0) {
        g_captureSessionStartWallMs = nowWallMs;
    }
    if (g_captureSessionStartMonoNs <= 0) {
        g_captureSessionStartMonoNs = nowMonoNs;
    }

    QJsonObject obj = extra;
    obj["event"] = event;
    obj["wall_time"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    obj["wall_time_ms"] = static_cast<double>(nowWallMs);
    obj["monotonic_ns"] = static_cast<double>(nowMonoNs);
    obj["session_elapsed_ms"] = static_cast<double>((nowMonoNs - g_captureSessionStartMonoNs) / 1000000LL);
    obj["session_id"] = sessionId;
    obj["source"] = source;
    obj["path"] = path;
    obj["raw_video_requested"] = g_rawRecordingRequested.load();
    obj["raw_audio_requested"] = g_audioRawRecordingRequested.load();
    obj["dashboard_recording_requested"] = g_recordingRequested.load();
    obj["windows_recording_requested"] = g_windowsRemoteRecordingRequested.load();

    QFile f(recordingTimelinePath(date, sessionId));
    if (f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        f.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
        f.write("\n");
    }
}

static QString sha256HexForFile(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return "";
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!f.atEnd()) {
        hash.addData(f.read(1024 * 1024));
    }
    return QString::fromLatin1(hash.result().toHex());
}

static void writeSessionSha256Manifest(const QString& sessionId) {
    if (sessionId.isEmpty()) return;
    const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
    const QString sessionDir = sessionRootDirPath(date, sessionId);
    QDir().mkpath(sessionDir + "/metadata");

    QJsonArray files;
    QDir rootDir(sessionDir);
    QDirIterator it(sessionDir, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString relativePath = rootDir.relativeFilePath(path);
        if (relativePath == "metadata/sha256_manifest.json") continue;
        QFileInfo info(path);
        QJsonObject item;
        item["relative_path"] = relativePath;
        item["path"] = path;
        item["bytes"] = static_cast<double>(info.size());
        item["sha256"] = sha256HexForFile(path);
        files.append(item);
    }

    QJsonObject root;
    root["created_by"] = "Codex";
    root["updated_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    root["session_id"] = sessionId;
    root["file_count"] = files.size();
    root["files"] = files;

    QFile f(sessionDir + "/metadata/sha256_manifest.json");
    if (f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
}

static void writeCaptureSessionSummary(const QString& state) {
    const QString sessionId = currentCaptureSessionId();
    if (sessionId.isEmpty()) return;
    const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
    const QString sessionDir = sessionRootDirPath(date, sessionId);
    QJsonObject root;
    root["created_by"] = "Codex";
    root["updated_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    root["state"] = state;
    root["session_id"] = sessionId;
    root["data_root"] = antiUavDataRootPath();
    root["session_dir"] = sessionDir;
    root["session_start_wall_time_ms"] = static_cast<double>(g_captureSessionStartWallMs);
    root["session_start_monotonic_ns"] = static_cast<double>(g_captureSessionStartMonoNs);
    root["raw_video_path"] = currentRawRecordingPath();
    root["raw_audio_path"] = currentAudioRawRecordingPath();
    root["dashboard_recording_path"] = currentRecordingPath();
    root["windows_recording_dir"] = sessionDir + "/windows_recordings";
    root["recording_timeline"] = recordingTimelinePath(date, sessionId);
    root["sync_capture_semantics"] =
        "RK端原始视频、原始六通道音频、主通道音频和元数据共享 session_id 与 monotonic 时间基准；Windows本地录像是独立能力。";

    QFile f(sessionSummaryPath(date, sessionId));
    if (f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
    writeSessionSha256Manifest(sessionId);
}

static void sendRkAudioRawRecordingCommand(bool enabled,
                                           const QString& recordId,
                                           const QString& sessionId,
                                           const QString& source) {
    const int audioPort = envIntBounded("ANTI_UAV_AUDIO_CONTROL_PORT", 5008, 1, 65535);
    const QString learnedAudioHost = latestAudioSenderHost();
    if (!learnedAudioHost.isEmpty() && learnedAudioHost != "127.0.0.1" &&
        learnedAudioHost != "localhost") {
        sendAudioRawRecordingControlDatagram(
            learnedAudioHost, audioPort, enabled, recordId, sessionId, source);
    }
    sendAudioRawRecordingControlDatagram(
        "127.0.0.1", audioPort, enabled, recordId, sessionId, source);
}

static void handleRkSyncCaptureControl(const QString& action,
                                       const QString& preferredSessionId,
                                       const QString& source) {
    if (antiUavNodeRole() != "elf2") return;

    if (action == "start") {
        if (g_fusedRecordingRequested.load()) return;
        if (!recordingStorageAllowed("RK原始音视频同步采集", true)) return;
        const QString sessionId = ensureCaptureSessionForRecording(preferredSessionId);
        g_recordingPaused.store(false);
        g_fusedRecordingRequested.store(true);
        g_fusedRecordingActive.store(true);
        setCurrentRawRecordingPath("");
        setCurrentRawRecordingSessionId(sessionId);
        g_rawRecordingRequested.store(true);
        setCurrentAudioRawRecordingPath("");
        setCurrentAudioRawRecordingSessionId(sessionId);
        g_audioRawRecordingRequested.store(true);
        appendRecordingTimelineEvent("sync_raw_capture_start", source);
        appendRecordingTimelineEvent("raw_audio_command_start", "qt_audio_control", currentAudioRawRecordingPath());
        sendRkAudioRawRecordingCommand(true, sessionId, sessionId, source);
        writeCaptureSessionManifest("recording");
        writeCaptureSessionSummary("recording");
        qInfo() << "[REC] RK raw audio/video sync capture started by control:" << sessionId;
        return;
    }

    if (action == "pause") {
        const QString sessionId = currentCaptureSessionId();
        if (!g_fusedRecordingRequested.load() || sessionId.isEmpty() || g_recordingPaused.load()) return;
        appendRecordingTimelineEvent("sync_raw_capture_pause", source);
        g_recordingPaused.store(true);
        g_rawRecordingRequested.store(false);
        appendRecordingTimelineEvent("raw_audio_command_stop", "qt_audio_control", currentAudioRawRecordingPath());
        g_audioRawRecordingRequested.store(false);
        sendRkAudioRawRecordingCommand(false, sessionId, sessionId, source);
        writeCaptureSessionManifest("paused");
        writeCaptureSessionSummary("paused");
        qInfo() << "[REC] RK raw audio/video sync capture paused by control:" << sessionId;
        return;
    }

    if (action == "resume") {
        const QString sessionId = currentCaptureSessionId();
        if (!g_fusedRecordingRequested.load() || sessionId.isEmpty() || !g_recordingPaused.load()) return;
        appendRecordingTimelineEvent("sync_raw_capture_resume", source);
        g_recordingPaused.store(false);
        setCurrentRawRecordingPath("");
        setCurrentRawRecordingSessionId(sessionId);
        g_rawRecordingRequested.store(true);
        setCurrentAudioRawRecordingPath("");
        setCurrentAudioRawRecordingSessionId(sessionId);
        g_audioRawRecordingRequested.store(true);
        appendRecordingTimelineEvent("raw_audio_command_start", "qt_audio_control", currentAudioRawRecordingPath());
        sendRkAudioRawRecordingCommand(true, sessionId, sessionId, source);
        writeCaptureSessionManifest("recording");
        writeCaptureSessionSummary("recording");
        qInfo() << "[REC] RK raw audio/video sync capture resumed by control:" << sessionId;
        return;
    }

    if (action == "stop") {
        const QString sessionId = currentCaptureSessionId().isEmpty()
                                      ? preferredSessionId
                                      : currentCaptureSessionId();
        if (sessionId.isEmpty()) return;
        appendRecordingTimelineEvent("sync_raw_capture_stop", source);
        g_recordingPaused.store(false);
        g_fusedRecordingRequested.store(false);
        g_rawRecordingRequested.store(false);
        setCurrentRawRecordingSessionId("");
        appendRecordingTimelineEvent("raw_audio_command_stop", "qt_audio_control", currentAudioRawRecordingPath());
        g_audioRawRecordingRequested.store(false);
        sendRkAudioRawRecordingCommand(false, sessionId, sessionId, source);
        setCurrentAudioRawRecordingSessionId("");
        g_fusedRecordingActive.store(false);
        writeCaptureSessionManifest("stopped");
        writeCaptureSessionSummary("stopped");
        qInfo() << "[REC] RK raw audio/video sync capture stopped by control:" << sessionId;
    }
}

static QString chooseExperimentString(const QJsonObject& obj,
                                      std::initializer_list<const char*> keys,
                                      const QString& uiValue,
                                      const QString& fallback = "") {
    if (!uiValue.isEmpty()) return uiValue;
    const QString jsonValue = metricsJsonStringAny(obj, keys, fallback);
    return jsonValue.isEmpty() ? fallback : jsonValue;
}

static double experimentDoubleValue(const QString& text, double fallback) {
    bool ok = false;
    const double v = text.toDouble(&ok);
    return ok ? v : fallback;
}

static int experimentIntValue(const QString& text, int fallback) {
    bool ok = false;
    const int v = text.toInt(&ok);
    return ok ? v : fallback;
}

class MetricsLogger {
public:
    static MetricsLogger& instance() {
        static MetricsLogger logger;
        return logger;
    }

    ~MetricsLogger() {
        finalize();
    }

    void logVision(const QJsonObject& obj) {
        QMutexLocker locker(&mtx);
        const qint64 recv_ms = metricsNowMs();

        const QString raw = QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact));
        const int frame_id = metricsJsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
        const int detected = metricsJsonIntAny(obj, {"detected", "has_detection", "visual_detected"}, 0);
        const int tracking = metricsJsonIntAny(obj, {"tracking", "track", "confirmed"}, 0);
        const int track_id = metricsJsonIntAny(obj, {"track_id", "tid"}, -1);
        const int lost_frames = metricsJsonIntAny(obj, {"lost_frames", "lost_count"}, 0);
        const QString state = metricsJsonStringAny(obj, {"state", "status", "tracker_state"}, tracking ? "TRACK" : "SEARCH");

        const double conf = metricsJsonDoubleAny(obj, {"conf", "score", "confidence"}, 0);
        const double infer_ms = metricsJsonDoubleAny(obj, {"infer_ms", "yolo_infer_ms"}, -1);
        const double fps = metricsJsonDoubleAny(obj, {"fps", "vision_fps"}, -1);

        const double bbox_x = metricsJsonDoubleAny(obj, {"bbox_x", "x", "left", "cx"}, 0);
        const double bbox_y = metricsJsonDoubleAny(obj, {"bbox_y", "y", "top", "cy"}, 0);
        const double bbox_w = metricsJsonDoubleAny(obj, {"bbox_w", "w", "width", "bw"}, 0);
        const double bbox_h = metricsJsonDoubleAny(obj, {"bbox_h", "h", "height", "bh"}, 0);
        const double frame_w = metricsJsonDoubleAny(obj, {"frame_w", "fw", "coord_w", "image_w"}, 0);
        const double frame_h = metricsJsonDoubleAny(obj, {"frame_h", "fh", "coord_h", "image_h"}, 0);
        const double dx_px = metricsJsonDoubleAny(obj, {"dx_px", "dx", "err_x_px"}, 0);
        const double dy_px = metricsJsonDoubleAny(obj, {"dy_px", "dy", "err_y_px"}, 0);
        const double pan_err_deg = metricsJsonDoubleAny(obj, {"pan_err_deg", "err_pan_deg"}, 99999);
        const double tilt_err_deg = metricsJsonDoubleAny(obj, {"tilt_err_deg", "err_tilt_deg"}, 99999);
        const double visual_az = metricsJsonDoubleAny(obj, {"visual_azimuth_deg", "target_azimuth_deg", "azimuth_deg"}, 99999);
        const double pan_speed = metricsJsonDoubleAny(obj, {"pan_speed"}, 0);
        const double tilt_speed = metricsJsonDoubleAny(obj, {"tilt_speed"}, 0);
        const double zoom_speed = metricsJsonDoubleAny(obj, {"zoom_speed"}, 0);
        const double zoom_ratio = metricsJsonDoubleAny(obj, {"zoom_ratio", "zoom"}, -1);
        const double hfov_deg = metricsJsonDoubleAny(obj, {"hfov_deg", "HFOV", "hfov"}, -1);
        const double vfov_deg = metricsJsonDoubleAny(obj, {"vfov_deg", "VFOV", "vfov"}, -1);
        const QString fusion_mode = metricsJsonStringAny(obj, {"fusion_mode"}, "IDLE");
        const QString fusion_state = metricsJsonStringAny(obj, {"fusion_state"}, fusion_mode);
        const QString control_source = metricsJsonStringAny(obj, {"control_source"}, "");
        const QString search_state = metricsJsonStringAny(obj, {"search_state"}, "");
        const double vertical_sweep_delta = metricsJsonDoubleAny(obj, {"vertical_sweep_delta_deg"}, 0);
        const int audio_guided = metricsJsonIntAny(obj, {"audio_guided"}, 0);
        const double camera_az = metricsJsonDoubleAny(obj, {"camera_azimuth_deg"}, 99999);
        const double audio_world_az = metricsJsonDoubleAny(obj, {"audio_world_azimuth_deg"}, 99999);
        const double audio_pan_err = metricsJsonDoubleAny(obj, {"audio_pan_err_deg"}, 99999);
        const double av_angle_err = metricsJsonDoubleAny(obj, {"audio_visual_angle_error_deg"}, 99999);
        const double offset = metricsJsonDoubleAny(obj, {"mic_to_camera_offset_deg"}, 0);
        const double offset_conf = metricsJsonDoubleAny(obj, {"offset_confidence"}, 0);
        const QString offset_label = metricsJsonStringAny(obj, {"offset_confidence_label"}, "low");
        const int calibration_samples = metricsJsonIntAny(obj, {"calibration_samples"}, 0);
        const double doa_stability = metricsJsonDoubleAny(obj, {"doa_stability"}, 0);
        const QString camera_profile = metricsJsonStringAny(obj, {"camera_profile"}, "");
        const double brightness_mean = metricsJsonDoubleAny(obj, {"brightness_mean"}, -1);
        const double contrast_std = metricsJsonDoubleAny(obj, {"contrast_std"}, -1);
        const double saturation_mean = metricsJsonDoubleAny(obj, {"saturation_mean"}, -1);
        const double blur_laplacian_var = metricsJsonDoubleAny(obj, {"blur_laplacian_var"}, -1);
        const int is_grayscale_like = metricsJsonIntAny(obj, {"is_grayscale_like", "is_grayscale"}, 0);
        const QString image_quality_warning = metricsJsonStringAny(obj, {"image_quality_warning"}, "OK");
        const double center_error_px = std::sqrt(dx_px * dx_px + dy_px * dy_px);

        visionPackets += 1;
        detectedFrames += detected ? 1 : 0;
        trackedFrames += tracking ? 1 : 0;
        appendMetric(visionInferMs, infer_ms);
        appendMetric(visionFps, fps);

        QStringList fields;
        fields << QString::number(recv_ms)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(frame_id)
               << QString::number(detected)
               << QString::number(tracking)
               << QString::number(track_id)
               << QString::number(lost_frames)
               << metricsCsvEscape(state)
               << QString::number(conf, 'f', 4)
               << QString::number(infer_ms, 'f', 3)
               << QString::number(fps, 'f', 3)
               << QString::number(bbox_x, 'f', 2)
               << QString::number(bbox_y, 'f', 2)
               << QString::number(bbox_w, 'f', 2)
               << QString::number(bbox_h, 'f', 2)
               << QString::number(frame_w, 'f', 0)
               << QString::number(frame_h, 'f', 0)
               << QString::number(dx_px, 'f', 2)
               << QString::number(dy_px, 'f', 2)
               << QString::number(center_error_px, 'f', 2)
               << QString::number(pan_err_deg, 'f', 4)
               << QString::number(tilt_err_deg, 'f', 4)
               << QString::number(visual_az, 'f', 4)
               << QString::number(hfov_deg, 'f', 4)
               << QString::number(vfov_deg, 'f', 4)
               << QString::number(pan_speed, 'f', 1)
               << QString::number(tilt_speed, 'f', 1)
               << QString::number(zoom_speed, 'f', 1)
               << QString::number(zoom_ratio, 'f', 3)
               << metricsCsvEscape(fusion_mode)
               << metricsCsvEscape(fusion_state)
               << metricsCsvEscape(control_source)
               << metricsCsvEscape(search_state)
               << QString::number(vertical_sweep_delta, 'f', 3)
               << QString::number(audio_guided)
               << QString::number(camera_az, 'f', 4)
               << QString::number(audio_world_az, 'f', 4)
               << QString::number(audio_pan_err, 'f', 4)
               << QString::number(av_angle_err, 'f', 4)
               << QString::number(offset, 'f', 4)
               << QString::number(offset_conf, 'f', 4)
               << metricsCsvEscape(offset_label)
               << QString::number(calibration_samples)
               << QString::number(doa_stability, 'f', 4)
               << metricsCsvEscape(camera_profile)
               << QString::number(brightness_mean, 'f', 3)
               << QString::number(contrast_std, 'f', 3)
               << QString::number(saturation_mean, 'f', 3)
               << QString::number(blur_laplacian_var, 'f', 3)
               << QString::number(is_grayscale_like)
               << metricsCsvEscape(image_quality_warning)
               << metricsCsvEscape(raw);

        writeLine(visionFile, fields.join(","));
        if (!camera_profile.isEmpty() && camera_profile != lastCameraProfileLogged) {
            lastCameraProfileLogged = camera_profile;
            writeEventUnlocked(recv_ms, "CAMERA_PROFILE_SWITCH", "info",
                               -1.0, conf, "", "", raw);
        }
        if (!image_quality_warning.isEmpty() && image_quality_warning != "OK") {
            if (image_quality_warning != lastImageQualityWarningLogged ||
                recv_ms - lastImageQualityEventMs > 5000) {
                lastImageQualityWarningLogged = image_quality_warning;
                lastImageQualityEventMs = recv_ms;
                writeEventUnlocked(recv_ms, image_quality_warning, "warning",
                                   -1.0, conf, "", "", raw);
            }
        }
        logTrackerUnlocked(recv_ms, frame_id, track_id, bbox_x, bbox_y, bbox_w, bbox_h,
                           bbox_x + bbox_w * 0.5, bbox_y + bbox_h * 0.5,
                           conf, state, lost_frames, tracking, raw);
        logZoomUnlocked(recv_ms, obj, bbox_w, bbox_h, frame_w, frame_h, zoom_ratio, zoom_speed, raw);

        if (std::fabs(pan_speed) > 0.001 || std::fabs(tilt_speed) > 0.001 || std::fabs(zoom_speed) > 0.001) {
            logPtzObservedUnlocked(recv_ms, "vision_udp", pan_speed, tilt_speed, zoom_speed,
                                   -1.0, 0, 99999.0, 99999.0, zoom_ratio,
                                   pan_speed, tilt_speed, zoom_ratio, raw);
            if (lastAudioTriggerRecvMs > 0) {
                const qint64 latency = recv_ms - lastAudioTriggerRecvMs;
                appendMetric(audioToPtzLatencyMs, latency);
                fusionSuccessEvents += 1;
                writeFusionMetricUnlocked(recv_ms, "audio_to_ptz_latency_ms", latency,
                                          lastAudioAngle, visual_az, raw);
            }
        }

        if (detected && lastAudioTriggerRecvMs > 0) {
            const qint64 latency = recv_ms - lastAudioTriggerRecvMs;
            appendMetric(audioToVisionLatencyMs, latency);
            writeFusionMetricUnlocked(recv_ms, "audio_to_vision_confirm_latency_ms", latency,
                                      lastAudioAngle, visual_az, raw);
        }

        if (tracking && lastAudioTriggerRecvMs > 0) {
            const qint64 latency = recv_ms - lastAudioTriggerRecvMs;
            appendMetric(audioToStableTrackLatencyMs, latency);
            stableTrackEvents += 1;
            writeFusionMetricUnlocked(recv_ms, "audio_to_stable_track_latency_ms", latency,
                                      lastAudioAngle, visual_az, raw);
        }

        if (visual_az < 99998 && lastAudioAngleValid) {
            double err = visual_az - lastAudioAngle;
            while (err > 180.0) err -= 360.0;
            while (err < -180.0) err += 360.0;
            writeFusionMetricUnlocked(recv_ms, "doa_visual_angle_error_deg", qAbs(err),
                                      lastAudioAngle, visual_az, raw);
        }
        maybeWriteSummaryUnlocked(recv_ms);
    }

    void logAudio(const QJsonObject& obj) {
        QMutexLocker locker(&mtx);
        const qint64 recv_ms = metricsNowMs();
        const QString raw = QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact));

        const int frame_id = metricsJsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
        const int audio_detected = metricsJsonIntAny(obj, {"audio_detected", "detected", "trigger"}, 1);
        const double angle = metricsJsonDoubleAny(obj, {"doa_deg", "angle", "azimuth_deg"}, 99999);
        const double conf = metricsJsonDoubleAny(obj, {"confidence", "conf", "yamnet_score", "score"}, -1);
        const double infer_ms = metricsJsonDoubleAny(obj, {"audio_infer_ms", "infer_ms", "yamnet_infer_ms"}, -1);
        const QString audio_state = metricsJsonStringAny(obj, {"audio_state"}, audio_detected ? "CONFIRMED" : "IDLE");
        const double raw_doa = metricsJsonDoubleAny(obj, {"raw_doa_deg"}, 99999);
        const double smooth_doa = metricsJsonDoubleAny(obj, {"smooth_doa_deg", "azimuth_deg"}, angle);
        const int doa_valid = metricsJsonIntAny(obj, {"doa_valid"}, angle < 99998 ? 1 : 0);
        const int stable_doa = metricsJsonIntAny(obj, {"stable_doa"}, audio_detected);
        const double doa_confidence = metricsJsonDoubleAny(obj, {"doa_confidence"}, metricsJsonDoubleAny(obj, {"doa_stability"}, 0));
        const double stability = metricsJsonDoubleAny(obj, {"doa_stability"}, 0);
        const double score_ema = metricsJsonDoubleAny(obj, {"score_ema"}, conf);
        const double rms_dbfs = metricsJsonDoubleAny(obj, {"rms_dbfs"}, -120);
        const double ts_mono_ms = metricsJsonDoubleAny(obj, {"ts_mono_ms"}, -1);
        const double prob_raw = metricsJsonDoubleAny(obj, {"prob_raw"}, conf);
        const double prob_beam = metricsJsonDoubleAny(obj, {"prob_beam"}, conf);
        const double prob_final = metricsJsonDoubleAny(obj, {"prob_final"}, conf);
        const double srp_score = metricsJsonDoubleAny(obj, {"srp_score"}, stability);
        const double snr_db = metricsJsonDoubleAny(obj, {"snr_db", "snr"}, 99999);
        const int beamforming_enabled = metricsJsonIntAny(obj, {"beamforming_enabled"}, 0);
        const double raw_infer_ms = metricsJsonDoubleAny(obj, {"raw_infer_ms"}, -1);
        const double beam_infer_ms = metricsJsonDoubleAny(obj, {"beam_infer_ms"}, -1);
        const double doa_compute_ms = metricsJsonDoubleAny(obj, {"doa_compute_ms", "doa_ms"}, -1);
        const double beamform_ms = metricsJsonDoubleAny(obj, {"beamform_ms"}, -1);
        const int ptz_triggered = metricsJsonIntAny(obj, {"ptz_triggered", "trigger_ptz", "ptz_trigger"}, 0);
        const int alarm = metricsJsonIntAny(obj, {"alarm"}, audio_detected);
        const QString doaSource = metricsJsonStringAny(obj, {"doa_source"}, "");
        const double hardwareDoa = metricsJsonDoubleAny(obj, {"hardware_doa_deg"}, -1);
        const int hardwareVoice = metricsJsonIntAny(obj, {"hardware_voice"}, 0);
        const int postFilterEnabled = metricsJsonIntAny(obj, {"post_filter_enabled"}, 0);
        const int rawRecordingActive = metricsJsonIntAny(obj, {"raw_recording_active"}, 0);
        const QString rawRecordingPath = metricsJsonStringAny(obj, {"raw_recording_path"}, "");
        const QString rawRecordingSessionId = metricsJsonStringAny(obj, {"raw_recording_session_id"}, "");
        const int rawRecordingSamples = metricsJsonIntAny(obj, {"raw_recording_samples"}, 0);
        const ExperimentSelection selection = currentExperimentSelection();
        const QString experimentId = chooseExperimentString(obj, {"experiment_id"}, selection.experimentId);
        const QString sceneType = chooseExperimentString(obj, {"scene_type", "scenario"}, selection.sceneType);
        const double distanceM = !selection.distanceM.isEmpty()
            ? experimentDoubleValue(selection.distanceM, -1.0)
            : metricsJsonDoubleAny(obj, {"distance_m"}, -1.0);
        const QString noiseType = chooseExperimentString(obj, {"noise_type", "background_noise"}, selection.noiseType);
        const int targetPresent = !selection.targetPresent.isEmpty()
            ? experimentIntValue(selection.targetPresent, -1)
            : metricsJsonIntAny(obj, {"target_present"}, -1);
        const QString methodLabel = chooseExperimentString(obj, {"method_label"}, selection.methodLabel);
        const QString moduleLabel = chooseExperimentString(obj, {"module_label"}, selection.moduleLabel);
        const QString trialId = chooseExperimentString(obj, {"trial_id"}, selection.trialId);
        const double powerW = !selection.powerW.isEmpty()
            ? experimentDoubleValue(selection.powerW, -1.0)
            : metricsJsonDoubleAny(obj, {"power_w"}, -1.0);

        audioPackets += 1;
        audioTriggerCount += (audio_detected || alarm) ? 1 : 0;
        appendMetric(audioInferMs, infer_ms);
        appendMetric(rawProbs, prob_raw);
        appendMetric(beamProbs, prob_beam);
        appendMetric(finalProbs, prob_final);

        QStringList fields;
        fields << QString::number(recv_ms)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(frame_id)
               << QString::number(audio_detected)
               << QString::number(angle, 'f', 4)
               << QString::number(raw_doa, 'f', 4)
               << QString::number(smooth_doa, 'f', 4)
               << QString::number(doa_valid)
               << QString::number(stable_doa)
               << QString::number(doa_confidence, 'f', 4)
               << QString::number(stability, 'f', 4)
               << QString::number(srp_score, 'f', 4)
               << QString::number(snr_db, 'f', 3)
               << QString::number(conf, 'f', 4)
               << QString::number(prob_raw, 'f', 4)
               << QString::number(prob_beam, 'f', 4)
               << QString::number(prob_final, 'f', 4)
               << QString::number(score_ema, 'f', 4)
               << QString::number(rms_dbfs, 'f', 2)
               << metricsCsvEscape(audio_state)
               << QString::number(infer_ms, 'f', 3)
               << QString::number(raw_infer_ms, 'f', 3)
               << QString::number(beam_infer_ms, 'f', 3)
               << QString::number(doa_compute_ms, 'f', 3)
               << QString::number(beamform_ms, 'f', 3)
               << QString::number(beamforming_enabled)
               << QString::number(postFilterEnabled)
               << metricsCsvEscape(doaSource)
               << QString::number(hardwareDoa, 'f', 4)
               << QString::number(hardwareVoice)
               << QString::number(ptz_triggered)
               << QString::number(alarm)
               << QString::number(ts_mono_ms, 'f', 0)
               << metricsCsvEscape(experimentId)
               << metricsCsvEscape(sceneType)
               << QString::number(distanceM, 'f', 2)
               << metricsCsvEscape(noiseType)
               << QString::number(targetPresent)
               << metricsCsvEscape(methodLabel)
               << metricsCsvEscape(moduleLabel)
               << metricsCsvEscape(trialId)
               << QString::number(powerW, 'f', 3)
               << QString::number(rawRecordingActive)
               << metricsCsvEscape(rawRecordingPath)
               << metricsCsvEscape(rawRecordingSessionId)
               << QString::number(rawRecordingSamples)
               << metricsCsvEscape(raw);

        writeLine(audioFile, fields.join(","));

        if (audio_detected || alarm) {
            lastAudioRecvMs = recv_ms;
            lastAudioTriggerRecvMs = recv_ms;
            if (angle < 99998) {
                lastAudioAngle = angle;
                lastAudioAngleValid = true;
            }
            writeEventUnlocked(recv_ms, "audio_trigger", alarm ? "alarm" : "info",
                               prob_final, -1.0, "", "", raw);
        }
        maybeWriteSummaryUnlocked(recv_ms);
    }

    void logSystem(const QString& csvLine) {
        QMutexLocker locker(&mtx);
        writeLine(systemFile, csvLine);
        const QStringList p = csvLine.split(',');
        if (p.size() >= 4) {
            appendMetric(systemCpuPercent, p.value(2).toDouble());
            appendMetric(systemMemPercent, p.value(3).toDouble());
        }
        maybeWriteSummaryUnlocked(metricsNowMs());
    }

    void logSystemFromJson(const QJsonObject& obj, qint64 recvMs) {
        if (recvMs - lastSystemJsonWriteMs < 1000) {
            return;
        }
        const double cpu = metricsJsonDoubleAny(obj, {"cpu_percent", "rk_cpu_percent", "system_cpu_percent"}, -1.0);
        const double memPercent = metricsJsonDoubleAny(obj, {"mem_used_percent", "memory_percent", "rk_mem_used_percent"}, -1.0);
        const double memUsedMb = metricsJsonDoubleAny(obj, {"mem_used_mb", "memory_used_mb", "rk_mem_used_mb"}, -1.0);
        const double memTotalMb = metricsJsonDoubleAny(obj, {"mem_total_mb", "memory_total_mb", "rk_mem_total_mb"}, -1.0);
        const double npuFreq = metricsJsonDoubleAny(obj, {"npu_freq_mhz", "rk_npu_freq_mhz"}, -1.0);
        const double npuLoad = metricsJsonDoubleAny(obj, {"npu_load_percent", "rk_npu_load_percent"}, -1.0);
        const QString npuGov = metricsJsonStringAny(obj, {"npu_governor", "rk_npu_governor"}, "");
        const QString temp = metricsJsonStringAny(obj, {"temp_summary", "temperature_summary", "rk_temp_summary"}, "");
        const QString cpuFreq = metricsJsonStringAny(obj, {"cpu_freq_summary", "rk_cpu_freq_summary"}, "");
        const double processRssMb = metricsJsonDoubleAny(obj, {"process_rss_mb", "rk_process_rss_mb"}, -1.0);
        const int processThreads = metricsJsonIntAny(obj, {"process_threads", "rk_process_threads"}, -1);

        const bool hasAny =
            cpu >= 0.0 || memPercent >= 0.0 || memUsedMb >= 0.0 || memTotalMb >= 0.0 ||
            npuFreq >= 0.0 || npuLoad >= 0.0 || !npuGov.isEmpty() || !temp.isEmpty() ||
            !cpuFreq.isEmpty() || processRssMb >= 0.0 || processThreads >= 0;
        if (!hasAny) {
            return;
        }
        lastSystemJsonWriteMs = recvMs;

        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(cpu, 'f', 2)
               << QString::number(memPercent, 'f', 2)
               << QString::number(memUsedMb, 'f', 2)
               << QString::number(memTotalMb, 'f', 2)
               << QString::number(npuFreq, 'f', 2)
               << QString::number(npuLoad, 'f', 2)
               << metricsCsvEscape(npuGov)
               << metricsCsvEscape(temp)
               << metricsCsvEscape(cpuFreq)
               << QString::number(processRssMb, 'f', 2)
               << QString::number(processThreads);
        logSystem(fields.join(","));
    }

    void logFusionUdp(const QJsonObject& obj) {
        QMutexLocker locker(&mtx);
        const qint64 recv_ms = metricsNowMs();
        const QString raw = QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact));
        const QString eventType = metricsJsonStringAny(obj, {"event", "type", "fusion_event"}, "fusion_udp");
        const double value = metricsJsonDoubleAny(obj, {"value", "latency_ms", "confidence"}, -1);
        const double audioAngle = metricsJsonDoubleAny(obj, {"audio_angle_deg", "doa_deg", "angle"}, 99999);
        const double visualAngle = metricsJsonDoubleAny(obj, {"visual_angle_deg", "visual_azimuth_deg"}, 99999);
        writeFusionMetricUnlocked(recv_ms, eventType, value, audioAngle, visualAngle, raw);
        maybeWriteSummaryUnlocked(recv_ms);
    }

    void logFrameStreamEvent(const QString& eventType,
                             const QString& host,
                             int port,
                             int frameId,
                             int jsonLen,
                             int jpegLen,
                             int frameW,
                             int frameH,
                             const QString& status,
                             const QString& errorText) {
        QMutexLocker locker(&mtx);
        const qint64 recv_ms = metricsNowMs();
        frameStreamEvents += 1;
        if (eventType == "connected") frameStreamConnects += 1;
        if (eventType == "disconnected" || eventType == "connect_failed") frameStreamDisconnects += 1;
        if (eventType == "frame") frameStreamFrames += 1;

        QStringList fields;
        fields << QString::number(recv_ms)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << metricsCsvEscape(eventType)
               << metricsCsvEscape(host)
               << QString::number(port)
               << QString::number(frameId)
               << QString::number(jsonLen)
               << QString::number(jpegLen)
               << QString::number(frameW)
               << QString::number(frameH)
               << QString::number(frameStreamFrames)
               << metricsCsvEscape(status)
               << metricsCsvEscape(errorText);
        writeLine(frameStreamFile, fields.join(","));
        maybeWriteSummaryUnlocked(recv_ms);
    }

    void logVideoLatency(const VideoFrameTiming& timing,
                         qint64 paintStartMonoMs,
                         qint64 paintEndMonoMs) {
        QMutexLocker locker(&mtx);
        const auto delta = [](qint64 end, qint64 start) -> qint64 {
            return end >= 0 && start >= 0 && end >= start ? end - start : -1;
        };
        const qint64 rtspReceiveMs =
            delta(timing.rtsp_grab_done_mono_ms, timing.rtsp_grab_start_mono_ms);
        const qint64 decodeMs =
            delta(timing.decode_done_mono_ms, timing.rtsp_grab_done_mono_ms);
        const qint64 inferMs =
            delta(timing.infer_end_mono_ms, timing.infer_start_mono_ms);
        const qint64 postprocessMs =
            delta(timing.postprocess_done_mono_ms, timing.infer_end_mono_ms);
        const qint64 yoloToQtMs =
            delta(timing.qt_receive_done_mono_ms, timing.tcp_send_start_mono_ms);
        const qint64 qtDecodeMs =
            delta(timing.qt_decode_end_mono_ms, timing.qt_decode_start_mono_ms);
        const qint64 qtQueueMs =
            delta(paintStartMonoMs, timing.qt_publish_mono_ms);
        const qint64 qtRecvToPaintMs =
            delta(paintStartMonoMs, timing.qt_receive_done_mono_ms);
        const qint64 displayTotalMs =
            delta(paintStartMonoMs, timing.camera_estimated_mono_ms);
        const qint64 paintMs = delta(paintEndMonoMs, paintStartMonoMs);

        QStringList fields;
        fields << QString::number(metricsNowMs())
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(timing.frame_id)
               << QString::number(rtspReceiveMs)
               << QString::number(decodeMs)
               << QString::number(inferMs)
               << QString::number(postprocessMs)
               << QString::number(yoloToQtMs)
               << QString::number(qtDecodeMs)
               << QString::number(qtQueueMs)
               << QString::number(paintMs)
               << QString::number(displayTotalMs)
               << QString::number(timing.camera_estimated_mono_ms)
               << QString::number(timing.rtsp_grab_start_mono_ms)
               << QString::number(timing.rtsp_grab_done_mono_ms)
               << QString::number(timing.decode_done_mono_ms)
               << QString::number(timing.infer_start_mono_ms)
               << QString::number(timing.infer_end_mono_ms)
               << QString::number(timing.postprocess_done_mono_ms)
               << QString::number(timing.stream_enqueue_mono_ms)
               << QString::number(timing.jpeg_start_mono_ms)
               << QString::number(timing.jpeg_end_mono_ms)
               << QString::number(timing.tcp_send_start_mono_ms)
               << QString::number(timing.yolo_stream_send_ts_ms)
               << QString::number(timing.payload_size)
               << QString::number(timing.qt_receive_done_mono_ms)
               << QString::number(timing.qt_decode_start_mono_ms)
               << QString::number(timing.qt_decode_end_mono_ms)
               << QString::number(timing.qt_publish_mono_ms)
               << QString::number(paintStartMonoMs)
               << QString::number(paintEndMonoMs)
               << QString::number(timing.display_frame_id)
               << QString::number(timing.record_frame_id)
               << QString::number(timing.display_dropped_old_frames)
               << QString::number(timing.recording_queue_size)
               << QString::number(timing.recording_dropped_frames)
               << QString::number(timing.recording_write_ms)
               << QString::number(qtRecvToPaintMs)
               << QString::number(displayTotalMs);
        writeLine(videoLatencyFile, fields.join(","));
    }

    void logPtzCommand(const QString& source,
                       int pan,
                       int tilt,
                       int zoom,
                       double sendMs,
                       int returnCode,
                       const QString& errorText) {
        QMutexLocker locker(&mtx);
        const qint64 recv_ms = metricsNowMs();
        QJsonObject raw;
        raw["source"] = source;
        raw["pan"] = pan;
        raw["tilt"] = tilt;
        raw["zoom"] = zoom;
        raw["send_ms"] = sendMs;
        raw["return_code"] = returnCode;
        raw["error"] = errorText;
        logPtzObservedUnlocked(recv_ms, source, pan, tilt, zoom, sendMs, returnCode,
                               99999.0, 99999.0, 99999.0,
                               pan, tilt, zoom,
                               QString::fromUtf8(QJsonDocument(raw).toJson(QJsonDocument::Compact)));
        maybeWriteSummaryUnlocked(recv_ms);
    }

    void updateExperimentSelection(const ExperimentSelection& selection) {
        QMutexLocker locker(&mtx);
        exportExperimentSelectionToEnv(selection);
        writeExperimentConfig(runStamp);

        QJsonObject raw;
        raw["experiment_id"] = selection.experimentId;
        raw["scene_type"] = selection.sceneType;
        raw["distance_m"] = selection.distanceM;
        raw["noise_type"] = selection.noiseType;
        raw["target_present"] = selection.targetPresent;
        raw["method_label"] = selection.methodLabel;
        raw["module_label"] = selection.moduleLabel;
        raw["trial_id"] = selection.trialId;
        const QString json = QString::fromUtf8(QJsonDocument(raw).toJson(QJsonDocument::Compact));
        writeEventUnlocked(metricsNowMs(), "experiment_selection", "info",
                           -1.0, -1.0, "", "", json);
    }

    void finalize() {
        QMutexLocker locker(&mtx);
        if (finalized) return;
        finalized = true;

        writeSummaryUnlocked();

        closeIfOpen(visionFile);
        closeIfOpen(trackerFile);
        closeIfOpen(audioFile);
        closeIfOpen(ptzFile);
        closeIfOpen(zoomFile);
        closeIfOpen(fusionFile);
        closeIfOpen(systemFile);
        closeIfOpen(eventFile);
        closeIfOpen(frameStreamFile);
        closeIfOpen(videoLatencyFile);
    }

private:
    MetricsLogger() {
        const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        runStamp = stamp;
        runDir = diagnosticsRunDirPath(stamp);
        QDir().mkpath(runDir);

        openCsv(visionFile, runDir + "/vision_log.csv",
                "recv_ms,qt_time,frame_id,detected,tracking,track_id,lost_frames,state,conf,infer_ms,fps,bbox_x,bbox_y,bbox_w,bbox_h,frame_w,frame_h,dx_px,dy_px,center_error_px,pan_err_deg,tilt_err_deg,visual_azimuth_deg,hfov_deg,vfov_deg,pan_speed,tilt_speed,zoom_speed,zoom_ratio,fusion_mode,fusion_state,control_source,search_state,vertical_sweep_delta_deg,audio_guided,camera_azimuth_deg,audio_world_azimuth_deg,audio_pan_err_deg,audio_visual_angle_error_deg,mic_to_camera_offset_deg,offset_confidence,offset_confidence_label,calibration_samples,doa_stability,camera_profile,brightness_mean,contrast_std,saturation_mean,blur_laplacian_var,is_grayscale_like,image_quality_warning,raw_json");
        openCsv(trackerFile, runDir + "/tracker_log.csv",
                "recv_ms,qt_time,frame_id,track_id,bbox_x,bbox_y,bbox_w,bbox_h,center_x,center_y,score,state,lost_frames,tracking,raw_json");
        openCsv(audioFile, runDir + "/audio_log.csv",
                "recv_ms,qt_time,frame_id,audio_detected,doa_deg,raw_doa_deg,smooth_doa_deg,doa_valid,stable_doa,doa_confidence,doa_stability,srp_score,snr_db,confidence,prob_raw,prob_beam,prob_final,score_ema,rms_dbfs,audio_state,audio_infer_ms,raw_infer_ms,beam_infer_ms,doa_compute_ms,beamform_ms,beamforming_enabled,post_filter_enabled,doa_source,hardware_doa_deg,hardware_voice,ptz_triggered,alarm,ts_mono_ms,experiment_id,scene_type,distance_m,noise_type,target_present,method_label,module_label,trial_id,power_w,raw_recording_active,raw_recording_path,raw_recording_session_id,raw_recording_samples,raw_json");
        openCsv(ptzFile, runDir + "/ptz_log.csv",
                "recv_ms,qt_time,source,pan_cmd,tilt_cmd,zoom_cmd,send_ms,return_code,pan_before,tilt_before,zoom_before,pan_after,tilt_after,zoom_after,in_dead_zone,high_zoom_limited,raw_json");
        openCsv(zoomFile, runDir + "/zoom_log.csv",
                "recv_ms,qt_time,frame_id,target_box_ratio,zoom_ratio,zoom_speed,zoom_state,trigger_reason,stable_zone,in_cooldown,manual_override,raw_json");
        openCsv(fusionFile, runDir + "/fusion_log.csv",
                "recv_ms,qt_time,event_type,value,audio_angle_deg,visual_angle_deg,raw_json");
        openCsv(systemFile, runDir + "/system_log.csv",
                "recv_ms,qt_time,cpu_percent,mem_used_percent,mem_used_mb,mem_total_mb,npu_freq_mhz,npu_load_percent,npu_governor,temp_summary,cpu_freq_summary,process_rss_mb,process_threads");
        openCsv(eventFile, runDir + "/event_log.csv",
                "recv_ms,qt_time,event_type,alarm_level,audio_confidence,vision_confidence,screenshot_path,video_clip_path,standard_json");
        openCsv(frameStreamFile, runDir + "/frame_stream_log.csv",
                 "recv_ms,qt_time,event_type,host,port,frame_id,json_len,jpeg_len,frame_w,frame_h,total_frames,status,error_text");
        openCsv(videoLatencyFile, runDir + "/video_latency_log.csv",
                "recv_ms,qt_time,frame_id,rtsp_receive_ms,decode_ms,infer_ms,postprocess_ms,yolo_to_qt_ms,qt_decode_ms,qt_queue_ms,paint_ms,display_total_ms,camera_estimated_mono_ms,rtsp_grab_start_mono_ms,rtsp_grab_done_mono_ms,decode_done_mono_ms,infer_start_mono_ms,infer_end_mono_ms,postprocess_done_mono_ms,stream_enqueue_mono_ms,jpeg_start_mono_ms,jpeg_end_mono_ms,tcp_send_start_mono_ms,yolo_stream_send_ts_ms,payload_size,qt_receive_done_mono_ms,qt_decode_start_mono_ms,qt_decode_end_mono_ms,qt_publish_mono_ms,paint_start_mono_ms,paint_end_mono_ms,display_frame_id,record_frame_id,display_dropped_old_frames,recording_queue_size,recording_dropped_frames,recording_write_ms,qt_recv_to_paint_ms,qt_paint_frame_age_ms");
        writeExperimentConfig(stamp);

        qInfo() << "[METRICS] logging to:" << runDir;
    }

    void openCsv(QFile& f, const QString& path, const QString& header) {
        f.setFileName(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            qWarning() << "[METRICS] open failed:" << path << f.errorString();
            return;
        }
        writeLine(f, header);
    }

    void writeLine(QFile& f, const QString& line) {
        if (!f.isOpen()) return;
        f.write(line.toUtf8());
        f.write("\n");
        const qint64 nowMs = metricsNowMs();
        const int flushIntervalMs = envIntBounded("ANTI_UAV_METRICS_FLUSH_INTERVAL_MS", 1000, 100, 5000);
        if (lastMetricsFlushMs <= 0 || nowMs - lastMetricsFlushMs >= flushIntervalMs) {
            f.flush();
            lastMetricsFlushMs = nowMs;
        }
    }

    static void closeIfOpen(QFile& f) {
        if (f.isOpen()) {
            f.flush();
            f.close();
        }
    }

    static void appendMetric(std::vector<double>& values, double v) {
        if (std::isfinite(v) && v >= 0.0 && v < 99998.0) {
            values.push_back(v);
        }
    }

    static double meanMetric(const std::vector<double>& values) {
        if (values.empty()) return -1.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / static_cast<double>(values.size());
    }

    static double p95Metric(std::vector<double> values) {
        if (values.empty()) return -1.0;
        std::sort(values.begin(), values.end());
        const size_t idx = static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1;
        return values[std::min(idx, values.size() - 1)];
    }

    void writeExperimentConfig(const QString& stamp) {
        const ExperimentSelection selection = currentExperimentSelection();
        QJsonObject root;
        root["created_by"] = "Codex";
        root["created_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
        root["run_id"] = "run_" + stamp;
        root["run_dir"] = runDir;
        root["vision_udp_port"] = static_cast<int>(VISION_UDP_PORT);
        root["audio_udp_port"] = static_cast<int>(AUDIO_UDP_PORT);
        root["fusion_udp_port"] = static_cast<int>(FUSION_UDP_PORT);
        root["rk_frame_tcp_port"] = 5010;
        root["video_source"] = metricsEnvString("ANTI_UAV_QT_VIDEO_SOURCE");
        root["rtsp_url"] = QString::fromStdString(HIKVISION_RTSP_URL);
        root["experiment_id"] = selection.experimentId;
        root["scene_type"] = selection.sceneType;
        root["location"] = selection.location;
        root["distance_m"] = selection.distanceM;
        root["drone_model"] = metricsEnvString("ANTI_UAV_DRONE_MODEL");
        root["flight_mode"] = metricsEnvString("ANTI_UAV_FLIGHT_MODE");
        root["background"] = metricsEnvString("ANTI_UAV_BACKGROUND");
        root["noise_type"] = selection.noiseType;
        root["wind_level"] = metricsEnvString("ANTI_UAV_WIND_LEVEL");
        root["light_condition"] = metricsEnvString("ANTI_UAV_LIGHT_CONDITION");
        root["mic_array"] = metricsEnvString("ANTI_UAV_MIC_ARRAY");
        root["beamforming"] = metricsEnvString("ANTI_UAV_BEAMFORMING");
        root["auto_ptz"] = metricsEnvString("ANTI_UAV_AUTO_PTZ");
        root["auto_zoom"] = metricsEnvString("ANTI_UAV_AUTO_ZOOM");
        root["method_label"] = selection.methodLabel;
        root["module_label"] = selection.moduleLabel;
        root["target_present"] = selection.targetPresent;
        root["trial_id"] = selection.trialId;
        root["power_w"] = selection.powerW;
        QFile f(runDir + "/experiment_config.json");
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
            f.close();
        }
    }

    void logTrackerUnlocked(qint64 recvMs,
                            int frameId,
                            int trackId,
                            double bboxX,
                            double bboxY,
                            double bboxW,
                            double bboxH,
                            double centerX,
                            double centerY,
                            double score,
                            const QString& state,
                            int lostFrames,
                            int tracking,
                            const QString& raw) {
        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(frameId)
               << QString::number(trackId)
               << QString::number(bboxX, 'f', 2)
               << QString::number(bboxY, 'f', 2)
               << QString::number(bboxW, 'f', 2)
               << QString::number(bboxH, 'f', 2)
               << QString::number(centerX, 'f', 2)
               << QString::number(centerY, 'f', 2)
               << QString::number(score, 'f', 4)
               << metricsCsvEscape(state)
               << QString::number(lostFrames)
               << QString::number(tracking)
               << metricsCsvEscape(raw);
        writeLine(trackerFile, fields.join(","));
    }

    void logZoomUnlocked(qint64 recvMs,
                         const QJsonObject& obj,
                         double bboxW,
                         double bboxH,
                         double frameW,
                         double frameH,
                         double zoomRatio,
                         double zoomSpeed,
                         const QString& raw) {
        const double denom = frameW > 0.0 && frameH > 0.0 ? frameW * frameH : -1.0;
        const double targetRatio = denom > 0.0 ? (bboxW * bboxH) / denom : -1.0;
        const QString zoomState = metricsJsonStringAny(obj, {"zoom_state"}, std::fabs(zoomSpeed) > 0.001 ? "ACTIVE" : "IDLE");
        const QString reason = metricsJsonStringAny(obj, {"zoom_reason", "trigger_reason"}, "vision_udp");
        const int stableZone = metricsJsonIntAny(obj, {"stable_zone", "zoom_stable_zone"}, 0);
        const int cooldown = metricsJsonIntAny(obj, {"in_cooldown", "zoom_cooldown"}, 0);
        const int manualOverride = metricsJsonIntAny(obj, {"manual_override", "zoom_manual_override"}, 0);

        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << QString::number(metricsJsonIntAny(obj, {"frame_id", "frame", "id"}, -1))
               << QString::number(targetRatio, 'f', 6)
               << QString::number(zoomRatio, 'f', 3)
               << QString::number(zoomSpeed, 'f', 2)
               << metricsCsvEscape(zoomState)
               << metricsCsvEscape(reason)
               << QString::number(stableZone)
               << QString::number(cooldown)
               << QString::number(manualOverride)
               << metricsCsvEscape(raw);
        writeLine(zoomFile, fields.join(","));
    }

    void logPtzObservedUnlocked(qint64 recvMs,
                                const QString& source,
                                double pan,
                                double tilt,
                                double zoom,
                                double sendMs,
                                int returnCode,
                                double panBefore,
                                double tiltBefore,
                                double zoomBefore,
                                double panAfter,
                                double tiltAfter,
                                double zoomAfter,
                                const QString& raw) {
        const int inDeadZone = (std::fabs(pan) < 0.001 && std::fabs(tilt) < 0.001 && std::fabs(zoom) < 0.001) ? 1 : 0;
        const int highZoomLimited = 0;
        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << metricsCsvEscape(source)
               << QString::number(pan, 'f', 2)
               << QString::number(tilt, 'f', 2)
               << QString::number(zoom, 'f', 2)
               << QString::number(sendMs, 'f', 3)
               << QString::number(returnCode)
               << QString::number(panBefore, 'f', 3)
               << QString::number(tiltBefore, 'f', 3)
               << QString::number(zoomBefore, 'f', 3)
               << QString::number(panAfter, 'f', 3)
               << QString::number(tiltAfter, 'f', 3)
               << QString::number(zoomAfter, 'f', 3)
               << QString::number(inDeadZone)
               << QString::number(highZoomLimited)
               << metricsCsvEscape(raw);
        writeLine(ptzFile, fields.join(","));
    }

    void writeFusionMetricUnlocked(qint64 recvMs,
                                   const QString& eventType,
                                   double value,
                                   double audioAngle,
                                   double visualAngle,
                                   const QString& raw) {
        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << metricsCsvEscape(eventType)
               << QString::number(value, 'f', 4)
               << QString::number(audioAngle, 'f', 3)
               << QString::number(visualAngle, 'f', 3)
               << metricsCsvEscape(raw);
        writeLine(fusionFile, fields.join(","));
    }

    void writeEventUnlocked(qint64 recvMs,
                            const QString& eventType,
                            const QString& level,
                            double audioConfidence,
                            double visionConfidence,
                            const QString& screenshot,
                            const QString& videoClip,
                            const QString& standardJson) {
        QStringList fields;
        fields << QString::number(recvMs)
               << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
               << metricsCsvEscape(eventType)
               << metricsCsvEscape(level)
               << QString::number(audioConfidence, 'f', 4)
               << QString::number(visionConfidence, 'f', 4)
               << metricsCsvEscape(screenshot)
               << metricsCsvEscape(videoClip)
               << metricsCsvEscape(standardJson);
        writeLine(eventFile, fields.join(","));
    }

    void writeSummaryUnlocked() {
        const ExperimentSelection selection = currentExperimentSelection();
        QJsonObject summary;
        summary["created_by"] = "Codex";
        summary["created_at"] = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
        summary["run_dir"] = runDir;
        summary["experiment_id"] = selection.experimentId;
        summary["scene_type"] = selection.sceneType;
        summary["distance_m"] = selection.distanceM;
        summary["noise_type"] = selection.noiseType;
        summary["target_present"] = selection.targetPresent;
        summary["method_label"] = selection.methodLabel;
        summary["module_label"] = selection.moduleLabel;
        summary["trial_id"] = selection.trialId;
        summary["vision_packets"] = visionPackets;
        summary["audio_packets"] = audioPackets;
        summary["audio_triggers"] = audioTriggerCount;
        summary["frame_stream_events"] = frameStreamEvents;
        summary["frame_stream_connects"] = frameStreamConnects;
        summary["frame_stream_disconnects"] = frameStreamDisconnects;
        summary["frame_stream_frames"] = frameStreamFrames;
        summary["avg_yolo_infer_ms"] = meanMetric(visionInferMs);
        summary["p95_yolo_infer_ms"] = p95Metric(visionInferMs);
        summary["avg_audio_infer_ms"] = meanMetric(audioInferMs);
        summary["p95_audio_infer_ms"] = p95Metric(audioInferMs);
        summary["avg_fps"] = meanMetric(visionFps);
        summary["detection_success_rate"] = visionPackets > 0 ? static_cast<double>(detectedFrames) / visionPackets : 0.0;
        summary["track_ratio"] = visionPackets > 0 ? static_cast<double>(trackedFrames) / visionPackets : 0.0;
        summary["avg_prob_raw"] = meanMetric(rawProbs);
        summary["avg_prob_beam"] = meanMetric(beamProbs);
        summary["avg_prob_final"] = meanMetric(finalProbs);
        summary["beam_gain_percent"] = meanMetric(beamProbs) >= 0 && meanMetric(rawProbs) >= 0 ? meanMetric(beamProbs) - meanMetric(rawProbs) : -1.0;
        summary["avg_audio_to_ptz_latency_ms"] = meanMetric(audioToPtzLatencyMs);
        summary["p95_audio_to_ptz_latency_ms"] = p95Metric(audioToPtzLatencyMs);
        summary["avg_audio_to_vision_confirm_latency_ms"] = meanMetric(audioToVisionLatencyMs);
        summary["p95_audio_to_vision_confirm_latency_ms"] = p95Metric(audioToVisionLatencyMs);
        summary["avg_audio_to_stable_track_latency_ms"] = meanMetric(audioToStableTrackLatencyMs);
        summary["p95_audio_to_stable_track_latency_ms"] = p95Metric(audioToStableTrackLatencyMs);
        summary["linkage_success_rate"] = audioTriggerCount > 0 ? std::min(1.0, static_cast<double>(stableTrackEvents) / audioTriggerCount) : 0.0;
        summary["avg_cpu_percent"] = meanMetric(systemCpuPercent);
        summary["avg_mem_used_percent"] = meanMetric(systemMemPercent);

        QFile sf(runDir + "/summary.json");
        if (sf.open(QIODevice::WriteOnly | QIODevice::Text)) {
            sf.write(QJsonDocument(summary).toJson(QJsonDocument::Indented));
            sf.close();
        }

        QFile pf(runDir + "/paper_tables.md");
        if (pf.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QString md;
            md += "# Paper Tables\n\n";
            md += QString("日期：%1  \n执行者：Codex\n\n").arg(QDateTime::currentDateTime().toString("yyyy-MM-dd"));
            md += "| 指标 | 数值 |\n| --- | ---: |\n";
            md += QString("| 平均 YOLO 推理耗时 ms | %1 |\n").arg(meanMetric(visionInferMs), 0, 'f', 3);
            md += QString("| P95 YOLO 推理耗时 ms | %1 |\n").arg(p95Metric(visionInferMs), 0, 'f', 3);
            md += QString("| 平均音频推理耗时 ms | %1 |\n").arg(meanMetric(audioInferMs), 0, 'f', 3);
            md += QString("| P95 音频推理耗时 ms | %1 |\n").arg(p95Metric(audioInferMs), 0, 'f', 3);
            md += QString("| 平均 FPS | %1 |\n").arg(meanMetric(visionFps), 0, 'f', 3);
            md += QString("| RK 同帧流帧数 | %1 |\n").arg(frameStreamFrames);
            md += QString("| 检测成功率 | %1 |\n").arg(visionPackets > 0 ? static_cast<double>(detectedFrames) / visionPackets : 0.0, 0, 'f', 4);
            md += QString("| TRACK 比例 | %1 |\n").arg(visionPackets > 0 ? static_cast<double>(trackedFrames) / visionPackets : 0.0, 0, 'f', 4);
            md += QString("| raw 平均概率 | %1 |\n").arg(meanMetric(rawProbs), 0, 'f', 4);
            md += QString("| beam 平均概率 | %1 |\n").arg(meanMetric(beamProbs), 0, 'f', 4);
            md += QString("| final 平均概率 | %1 |\n").arg(meanMetric(finalProbs), 0, 'f', 4);
            md += QString("| 声学到 PTZ 平均延迟 ms | %1 |\n").arg(meanMetric(audioToPtzLatencyMs), 0, 'f', 3);
            md += QString("| 声学到视觉确认平均延迟 ms | %1 |\n").arg(meanMetric(audioToVisionLatencyMs), 0, 'f', 3);
            md += QString("| 声学到稳定跟踪平均延迟 ms | %1 |\n").arg(meanMetric(audioToStableTrackLatencyMs), 0, 'f', 3);
            md += QString("| 联动成功率 | %1 |\n").arg(audioTriggerCount > 0 ? std::min(1.0, static_cast<double>(stableTrackEvents) / audioTriggerCount) : 0.0, 0, 'f', 4);
            md += QString("| 平均 CPU 占用 % | %1 |\n").arg(meanMetric(systemCpuPercent), 0, 'f', 3);
            md += QString("| 平均内存占用 % | %1 |\n").arg(meanMetric(systemMemPercent), 0, 'f', 3);
            pf.write(md.toUtf8());
            pf.close();
        }
    }

    void maybeWriteSummaryUnlocked(qint64 nowMs) {
        if (nowMs - lastSummaryWriteMs < 1000) return;
        lastSummaryWriteMs = nowMs;
        writeSummaryUnlocked();
    }

    QMutex mtx;
    QString runStamp;
    QString runDir;
    QFile visionFile;
    QFile trackerFile;
    QFile audioFile;
    QFile ptzFile;
    QFile zoomFile;
    QFile systemFile;
    QFile fusionFile;
    QFile eventFile;
    QFile frameStreamFile;
    QFile videoLatencyFile;

    qint64 lastAudioRecvMs = -1;
    qint64 lastAudioTriggerRecvMs = -1;
    qint64 lastSummaryWriteMs = 0;
    qint64 lastSystemJsonWriteMs = 0;
    qint64 lastImageQualityEventMs = 0;
    qint64 lastMetricsFlushMs = 0;
    double lastAudioAngle = 0.0;
    bool lastAudioAngleValid = false;
    bool finalized = false;
    QString lastCameraProfileLogged;
    QString lastImageQualityWarningLogged = "OK";

    int visionPackets = 0;
    int detectedFrames = 0;
    int trackedFrames = 0;
    int audioPackets = 0;
    int audioTriggerCount = 0;
    int stableTrackEvents = 0;
    int fusionSuccessEvents = 0;
    int frameStreamEvents = 0;
    int frameStreamConnects = 0;
    int frameStreamDisconnects = 0;
    int frameStreamFrames = 0;

    std::vector<double> visionInferMs;
    std::vector<double> visionFps;
    std::vector<double> audioInferMs;
    std::vector<double> rawProbs;
    std::vector<double> beamProbs;
    std::vector<double> finalProbs;
    std::vector<double> audioToPtzLatencyMs;
    std::vector<double> audioToVisionLatencyMs;
    std::vector<double> audioToStableTrackLatencyMs;
    std::vector<double> systemCpuPercent;
    std::vector<double> systemMemPercent;
};

void logVideoDisplayTiming(const VideoFrameTiming& timing,
                           qint64 paintStartMonoMs,
                           qint64 paintEndMonoMs) {
    static std::atomic<int> lastLoggedFrame{-1};
    int previous = lastLoggedFrame.load();
    if (timing.frame_id <= previous ||
        !lastLoggedFrame.compare_exchange_strong(previous, timing.frame_id)) {
        return;
    }
    MetricsLogger::instance().logVideoLatency(timing, paintStartMonoMs, paintEndMonoMs);

    static std::atomic<qint64> lastConsoleMs{0};
    const qint64 now = qtMonotonicMs();
    qint64 last = lastConsoleMs.load();
    if (now - last >= 1000 && lastConsoleMs.compare_exchange_strong(last, now)) {
        const qint64 total = timing.camera_estimated_mono_ms >= 0
            ? paintStartMonoMs - timing.camera_estimated_mono_ms
            : -1;
        const qint64 queue = timing.qt_publish_mono_ms >= 0
            ? paintStartMonoMs - timing.qt_publish_mono_ms
            : -1;
        qInfo() << "[VIDEO_LATENCY] frame=" << timing.frame_id
                << "total_ms=" << total
                << "qt_queue_ms=" << queue
                << "backlog_bytes=" << g_frameStreamBacklogBytes.load()
                << "dropped_old_frames=" << g_frameStreamDroppedOldFrames.load()
                << "display_frame_id=" << g_frameStreamLastDisplayFrameId.load()
                << "record_frame_id=" << g_frameStreamLastRecordFrameId.load()
                << "recording_queue_size=" << g_frameStreamRecordingQueueSize.load()
                << "recording_dropped_frames=" << g_frameStreamRecordingDroppedFrames.load()
                << "recording_write_ms=" << g_frameStreamRecordingWriteMs.load()
                << "decoded_frames=" << g_frameStreamDecodedFrames.load()
                << "published_frames=" << g_frameStreamPublishedFrames.load();
    }
}

class SystemMetricsWorker : public QThread {
public:
    void run() override {
        while (!isInterruptionRequested()) {
            MetricsLogger::instance().logSystem(sampleSystemCsvLine());
            for (int i = 0; i < 10 && !isInterruptionRequested(); ++i) {
                QThread::msleep(100);
            }
        }
    }

private:
    long long prevTotal = -1;
    long long prevIdle = -1;
#ifdef Q_OS_WIN
    bool hasPrevWindowsCpu = false;
    quint64 prevWindowsIdle = 0;
    quint64 prevWindowsKernel = 0;
    quint64 prevWindowsUser = 0;
#endif

    QString sampleSystemCsvLine() {
        const qint64 t = metricsNowMs();
        const QString qtTime = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz");

        double cpu = sampleCpuPercent();
        double memTotal = 0, memUsed = 0, memPercent = 0;
        sampleMem(memTotal, memUsed, memPercent);

        QString npuGov = "";
        double npuFreq = readFirstDevfreqValue({"fdab0000.npu"}, {"cur_freq", "scaling_cur_freq"}, 1000000.0);
        double npuLoad = readNpuLoad(npuGov);

        QString thermalSummary = readThermalSummary();
        QString temps = metricsCsvEscape(thermalSummary);
        QString freqs = metricsCsvEscape(readCpuFreqSummary());
        double processRssMb = -1.0;
        int processThreads = -1;
        sampleCurrentProcess(processRssMb, processThreads);
        g_latestSystemCpuPercent.store(cpu);
        g_latestSystemMemPercent.store(memPercent);
        g_latestNpuFreqMhz.store(npuFreq);
        g_latestNpuLoadPercent.store(npuLoad);

        double maxTemperature = -1.0;
        const QStringList thermalEntries = thermalSummary.split('|', Qt::SkipEmptyParts);
        for (const QString& entry : thermalEntries) {
            const int colon = entry.lastIndexOf(':');
            const int suffix = entry.lastIndexOf('C');
            if (colon < 0 || suffix <= colon) continue;
            bool ok = false;
            const double value = entry.mid(colon + 1, suffix - colon - 1).toDouble(&ok);
            if (ok) maxTemperature = std::max(maxTemperature, value);
        }
        g_latestMaxTemperatureC.store(maxTemperature);

        return QString("%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13")
            .arg(t)
            .arg(qtTime)
            .arg(cpu, 0, 'f', 2)
            .arg(memPercent, 0, 'f', 2)
            .arg(memUsed, 0, 'f', 2)
            .arg(memTotal, 0, 'f', 2)
            .arg(npuFreq, 0, 'f', 2)
            .arg(npuLoad, 0, 'f', 2)
            .arg(metricsCsvEscape(npuGov))
            .arg(temps)
            .arg(freqs)
            .arg(processRssMb, 0, 'f', 2)
            .arg(processThreads);
    }

    double sampleCpuPercent() {
#ifdef Q_OS_WIN
        FILETIME idleTime, kernelTime, userTime;
        if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
            return -1.0;
        }

        auto toUInt64 = [](const FILETIME& ft) -> quint64 {
            ULARGE_INTEGER value;
            value.LowPart = ft.dwLowDateTime;
            value.HighPart = ft.dwHighDateTime;
            return value.QuadPart;
        };

        const quint64 idle = toUInt64(idleTime);
        const quint64 kernel = toUInt64(kernelTime);
        const quint64 user = toUInt64(userTime);
        double cpu = -1.0;
        if (hasPrevWindowsCpu) {
            const quint64 idleDelta = idle - prevWindowsIdle;
            const quint64 kernelDelta = kernel - prevWindowsKernel;
            const quint64 userDelta = user - prevWindowsUser;
            const quint64 totalDelta = kernelDelta + userDelta;
            if (totalDelta > 0) {
                cpu = 100.0 * static_cast<double>(totalDelta - idleDelta) /
                      static_cast<double>(totalDelta);
            }
        }
        prevWindowsIdle = idle;
        prevWindowsKernel = kernel;
        prevWindowsUser = user;
        hasPrevWindowsCpu = true;
        return cpu;
#else
        QFile f("/proc/stat");
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return -1;
        QString line = QString::fromUtf8(f.readLine()).trimmed();
        QStringList p = line.split(' ', Qt::SkipEmptyParts);
        if (p.size() < 5) return -1;

        long long user = p.value(1).toLongLong();
        long long nice = p.value(2).toLongLong();
        long long sys  = p.value(3).toLongLong();
        long long idle = p.value(4).toLongLong();
        long long iowait = p.size() > 5 ? p.value(5).toLongLong() : 0;
        long long irq = p.size() > 6 ? p.value(6).toLongLong() : 0;
        long long softirq = p.size() > 7 ? p.value(7).toLongLong() : 0;

        long long idleAll = idle + iowait;
        long long total = user + nice + sys + idle + iowait + irq + softirq;

        double cpu = -1;
        if (prevTotal > 0) {
            long long dt = total - prevTotal;
            long long di = idleAll - prevIdle;
            if (dt > 0) cpu = 100.0 * (dt - di) / dt;
        }

        prevTotal = total;
        prevIdle = idleAll;
        return cpu;
#endif
    }

    void sampleMem(double& totalMb, double& usedMb, double& percent) {
#ifdef Q_OS_WIN
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) {
            totalMb = usedMb = percent = -1.0;
            return;
        }
        totalMb = static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0);
        const double availMb = static_cast<double>(status.ullAvailPhys) / (1024.0 * 1024.0);
        usedMb = totalMb - availMb;
        percent = static_cast<double>(status.dwMemoryLoad);
#else
        QString txt = metricsReadText("/proc/meminfo");
        double totalKb = 0, availKb = 0;
        for (const QString& line : txt.split('\n')) {
            QStringList p = line.split(' ', Qt::SkipEmptyParts);
            if (p.size() >= 2) {
                if (p[0] == "MemTotal:") totalKb = p[1].toDouble();
                if (p[0] == "MemAvailable:") availKb = p[1].toDouble();
            }
        }
        totalMb = totalKb / 1024.0;
        usedMb = (totalKb - availKb) / 1024.0;
        percent = totalKb > 0 ? 100.0 * (totalKb - availKb) / totalKb : -1;
#endif
    }

    void sampleCurrentProcess(double& rssMb, int& threads) {
#ifdef Q_OS_WIN
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                 sizeof(pmc))) {
            rssMb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        } else {
            rssMb = -1.0;
        }

        threads = -1;
        const DWORD currentPid = GetCurrentProcessId();
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot != INVALID_HANDLE_VALUE) {
            THREADENTRY32 entry;
            entry.dwSize = sizeof(entry);
            int count = 0;
            if (Thread32First(snapshot, &entry)) {
                do {
                    if (entry.th32OwnerProcessID == currentPid) {
                        ++count;
                    }
                    entry.dwSize = sizeof(entry);
                } while (Thread32Next(snapshot, &entry));
            }
            CloseHandle(snapshot);
            threads = count;
        }
#else
        const QString txt = metricsReadText("/proc/self/status");
        if (txt.isEmpty()) {
            rssMb = -1.0;
            threads = -1;
            return;
        }
        rssMb = -1.0;
        threads = -1;
        for (const QString& line : txt.split('\n')) {
            QStringList p = line.split(' ', Qt::SkipEmptyParts);
            if (p.size() >= 2 && p[0] == "VmRSS:") {
                rssMb = p[1].toDouble() / 1024.0;
            }
            if (p.size() >= 2 && p[0] == "Threads:") {
                threads = p[1].toInt();
            }
        }
#endif
    }

    double readFirstDevfreqValue(std::initializer_list<const char*> nameHints,
                                 std::initializer_list<const char*> files,
                                 double scale) {
        QDir d("/sys/class/devfreq");
        QStringList entries = d.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        entries.prepend("fdab0000.npu");
        for (const QString& e : entries) {
            bool match = false;
            for (const char* h : nameHints) {
                if (e.contains(h, Qt::CaseInsensitive) || e.contains("npu", Qt::CaseInsensitive)) match = true;
            }
            if (!match) continue;
            for (const char* file : files) {
                QString v = metricsReadText("/sys/class/devfreq/" + e + "/" + file);
                if (!v.isEmpty()) return v.split(' ', Qt::SkipEmptyParts).value(0).toDouble() / scale;
            }
        }
        return -1;
    }

    double readNpuLoad(QString& gov) {
        QDir d("/sys/class/devfreq");
        QStringList entries = d.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        entries.prepend("fdab0000.npu");
        for (const QString& e : entries) {
            if (!e.contains("npu", Qt::CaseInsensitive) && e != "fdab0000.npu") continue;
            QString base = "/sys/class/devfreq/" + e + "/";
            QString g = metricsReadText(base + "governor");
            if (!g.isEmpty()) gov = g;
            QString load = metricsReadText(base + "load");
            if (!load.isEmpty()) {
                QString num;
                for (QChar c : load) {
                    if (c.isDigit() || c == '.') num.append(c);
                    else if (!num.isEmpty()) break;
                }
                if (!num.isEmpty()) return num.toDouble();
            }
        }
        return -1;
    }

    QString readThermalSummary() {
        QDir d("/sys/class/thermal");
        QStringList out;
        for (const QString& z : d.entryList(QStringList() << "thermal_zone*", QDir::Dirs | QDir::NoDotAndDotDot)) {
            QString type = metricsReadText("/sys/class/thermal/" + z + "/type");
            QString temp = metricsReadText("/sys/class/thermal/" + z + "/temp");
            if (!temp.isEmpty()) out << QString("%1:%2C").arg(type.isEmpty() ? z : type).arg(temp.toDouble()/1000.0, 0, 'f', 1);
        }
        return out.join("|");
    }

    QString readCpuFreqSummary() {
        QDir d("/sys/devices/system/cpu/cpufreq");
        QStringList out;
        for (const QString& p : d.entryList(QStringList() << "policy*", QDir::Dirs | QDir::NoDotAndDotDot)) {
            QString v = metricsReadText("/sys/devices/system/cpu/cpufreq/" + p + "/scaling_cur_freq");
            if (!v.isEmpty()) out << QString("%1:%2MHz").arg(p).arg(v.toDouble()/1000.0, 0, 'f', 0);
        }
        return out.join("|");
    }
};

// ============================================================
// METRICS LOGGER PATCH END
// ============================================================


static bool parseVisionMeta(const QJsonObject& obj, SensorMeta& meta) {
    meta.frame_id = jsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
    meta.source_ts_ms = static_cast<long long>(jsonDoubleAny(obj, {"ts_ms", "source_ts_ms"}, 0.0));
    meta.infer_ms = jsonDoubleAny(obj, {"infer_ms", "yolo_infer_ms"}, -1.0);

    // 兼容两种格式：
    // 1) {"bbox_x":..., "bbox_y":..., "bbox_w":..., "bbox_h":...}
    // 2) {"x":..., "y":..., "w":..., "h":...}
    meta.bbox_x = jsonIntAny(obj, {"bbox_x", "x", "left", "cx"}, 0);
    meta.bbox_y = jsonIntAny(obj, {"bbox_y", "y", "top", "cy"}, 0);
    meta.bbox_w = jsonIntAny(obj, {"bbox_w", "w", "width", "bw"}, 0);
    meta.bbox_h = jsonIntAny(obj, {"bbox_h", "h", "height", "bh"}, 0);

    // 可选 bbox 数组：[x,y,w,h] 或 [x1,y1,x2,y2]
    if (obj.contains("bbox") && obj.value("bbox").isArray()) {
        QJsonArray arr = obj.value("bbox").toArray();
        if (arr.size() >= 4) {
            int a = arr.at(0).toInt();
            int b = arr.at(1).toInt();
            int c = arr.at(2).toInt();
            int d = arr.at(3).toInt();

            meta.bbox_x = a;
            meta.bbox_y = b;

            // 如果 c/d 看起来像右下角，则转成 w/h；否则按 w/h 使用。
            if (c > a && d > b && (c > 300 || d > 300)) {
                meta.bbox_w = c - a;
                meta.bbox_h = d - b;
            } else {
                meta.bbox_w = c;
                meta.bbox_h = d;
            }
        }
    }

    meta.coord_w = jsonIntAny(obj, {"coord_w", "fw", "frame_w", "frame_width", "img_w", "image_w"}, 0);
    meta.coord_h = jsonIntAny(obj, {"coord_h", "fh", "frame_h", "frame_height", "img_h", "image_h"}, 0);

    meta.conf = static_cast<float>(jsonDoubleAny(obj, {"conf", "score", "confidence"}, 0.0));

    QString state = obj.value("state").toString();
    QString status = obj.value("status").toString();
    meta.tracking = jsonBoolAny(obj, {"tracking", "track", "confirmed"}, false)
                    || state.contains("TRACK", Qt::CaseInsensitive)
                    || status.contains("TRACK", Qt::CaseInsensitive);

    meta.has_detection = (meta.bbox_w > 0 && meta.bbox_h > 0)
                         || jsonBoolAny(obj, {"has_detection", "detected", "visual_detected"}, false);

    return meta.has_detection;
}

static void updateCameraQualityDashboardState(const QJsonObject& obj) {
    setDashboardText(&g_latestCameraProfile,
                     metricsJsonStringAny(obj, {"camera_profile"}, dashboardText(g_latestCameraProfile)));
    g_latestBrightnessMean.store(metricsJsonDoubleAny(obj, {"brightness_mean"}, g_latestBrightnessMean.load()));
    g_latestContrastStd.store(metricsJsonDoubleAny(obj, {"contrast_std"}, g_latestContrastStd.load()));
    g_latestSaturationMean.store(metricsJsonDoubleAny(obj, {"saturation_mean"}, g_latestSaturationMean.load()));
    g_latestBlurLaplacianVar.store(metricsJsonDoubleAny(obj, {"blur_laplacian_var"}, g_latestBlurLaplacianVar.load()));
    g_latestGrayscaleLike.store(metricsJsonIntAny(obj, {"is_grayscale_like", "is_grayscale"},
                                                 g_latestGrayscaleLike.load() ? 1 : 0) != 0);
    setDashboardText(&g_latestImageQualityWarning,
                     metricsJsonStringAny(obj, {"image_quality_warning"}, dashboardText(g_latestImageQualityWarning)));
    setDashboardText(&g_latestProfileSwitchReason,
                     metricsJsonStringAny(obj, {"last_profile_switch_reason"}, dashboardText(g_latestProfileSwitchReason)));
}

static void updateVisionDashboardState(const QJsonObject& obj, qint64 recvMs) {
    g_lastVisionRecvMs.store(recvMs);
    if (obj.contains("detection_enabled")) {
        g_detectionEnabledReported.store(metricsJsonIntAny(obj, {"detection_enabled"}, 1) != 0);
    }
    g_latestVisionDetected.store(metricsJsonIntAny(obj, {"detected", "has_detection", "visual_detected"}, 0) != 0);
    g_latestVisionTracking.store(metricsJsonIntAny(obj, {"tracking", "track", "confirmed"}, 0) != 0);
    g_latestVisionConf.store(metricsJsonDoubleAny(obj, {"conf", "score", "confidence"}, 0.0));
    g_latestVisionFps.store(metricsJsonDoubleAny(obj, {"fps", "vision_fps"}, g_latestVisionFps.load()));
    g_latestVisionInferMs.store(
        metricsJsonDoubleAny(obj, {"infer_ms", "yolo_infer_ms"}, g_latestVisionInferMs.load()));
    g_latestCameraAzimuthDeg.store(metricsJsonDoubleAny(obj, {"camera_azimuth_deg"}, 99999.0));
    g_latestCameraAzimuthUnit.store(metricsJsonIntAny(
        obj, {"camera_azimuth_unit"}, g_latestCameraAzimuthUnit.load()));
    g_latestCameraElevationUnit.store(metricsJsonIntAny(
        obj, {"camera_elevation_unit"}, g_latestCameraElevationUnit.load()));
    g_latestCameraAbsoluteZoom.store(metricsJsonIntAny(
        obj, {"camera_absolute_zoom"}, g_latestCameraAbsoluteZoom.load()));
    g_latestAutoTrackingEnabled.store(metricsJsonIntAny(
        obj, {"auto_tracking_enabled"}, g_latestAutoTrackingEnabled.load() ? 1 : 0) != 0);
    g_latestAutoZoomEnabled.store(metricsJsonIntAny(
        obj, {"auto_zoom_enabled"}, g_latestAutoZoomEnabled.load() ? 1 : 0) != 0);
    g_latestZoomRatio.store(metricsJsonDoubleAny(obj, {"zoom_ratio", "zoom"}, g_latestZoomRatio.load()));
    g_latestZoomSpeed.store(metricsJsonDoubleAny(obj, {"zoom_speed"}, g_latestZoomSpeed.load()));
    g_latestFocusRequestMs.store(static_cast<qint64>(metricsJsonDoubleAny(
        obj, {"last_focus_request_ms"}, static_cast<double>(g_latestFocusRequestMs.load()))));
    g_latestFocusCooldownRemainingMs.store(static_cast<qint64>(metricsJsonDoubleAny(
        obj, {"focus_cooldown_remaining_ms"}, static_cast<double>(g_latestFocusCooldownRemainingMs.load()))));
    g_latestVisualAzimuthDeg.store(metricsJsonDoubleAny(obj, {"visual_azimuth_deg", "target_azimuth_deg", "azimuth_deg"}, 99999.0));
    g_latestAudioWorldAzimuthDeg.store(metricsJsonDoubleAny(obj, {"audio_world_azimuth_deg"}, 99999.0));
    g_latestAudioPanErrDeg.store(metricsJsonDoubleAny(obj, {"audio_pan_err_deg"}, 99999.0));
    g_latestAudioVisualErrDeg.store(metricsJsonDoubleAny(obj, {"audio_visual_angle_error_deg"}, 99999.0));
    g_latestMicOffsetDeg.store(metricsJsonDoubleAny(obj, {"mic_to_camera_offset_deg"}, 0.0));
    g_latestOffsetConfidence.store(metricsJsonDoubleAny(obj, {"offset_confidence"}, 0.0));
    g_latestCalibrationSamples.store(metricsJsonIntAny(obj, {"calibration_samples"}, 0));
    g_latestAudioGuided.store(metricsJsonIntAny(obj, {"audio_guided"}, 0) != 0);
    setDashboardText(&g_latestFusionMode, metricsJsonStringAny(obj, {"fusion_mode"}, "IDLE"));
    setDashboardText(&g_latestFusionState, metricsJsonStringAny(obj, {"fusion_state"}, metricsJsonStringAny(obj, {"fusion_mode"}, "IDLE")));
    setDashboardText(&g_latestControlSource, metricsJsonStringAny(obj, {"control_source"}, "none"));
    setDashboardText(&g_latestSearchState, metricsJsonStringAny(obj, {"search_state"}, "none"));
    setDashboardText(&g_latestFocusState, metricsJsonStringAny(obj, {"focus_state"}, "DISABLED"));
    g_latestVerticalSweepDeltaDeg.store(metricsJsonDoubleAny(obj, {"vertical_sweep_delta_deg"}, 0.0));
    setDashboardText(&g_latestOffsetConfidenceLabel, metricsJsonStringAny(obj, {"offset_confidence_label"}, "low"));
    setDashboardText(&g_latestAudioGuidanceState,
                     metricsJsonStringAny(obj, {"audio_guidance_state"}, dashboardText(g_latestAudioGuidanceState)));
    setDashboardText(&g_latestAudioRejectReason,
                     metricsJsonStringAny(obj, {"audio_reject_reason"}, dashboardText(g_latestAudioRejectReason)));
    setDashboardText(&g_latestAudioCalibrationState,
                     metricsJsonStringAny(obj, {"audio_calibration_state"}, dashboardText(g_latestAudioCalibrationState)));
    updateCameraQualityDashboardState(obj);
}

static void publishVisionMetaFromJson(const QJsonObject& obj, qint64 recvMs) {
    SensorMeta meta;
    if (!parseVisionMeta(obj, meta)) {
        auto emptyMeta = std::make_shared<SensorMeta>();
        emptyMeta->frame_id = jsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
        emptyMeta->recv_ms = recvMs;
        emptyMeta->source_ts_ms = static_cast<long long>(metricsJsonDoubleAny(obj, {"ts_ms", "source_ts_ms"}, 0.0));
        emptyMeta->infer_ms = metricsJsonDoubleAny(obj, {"infer_ms", "yolo_infer_ms"}, -1.0);
        std::atomic_store(&g_latestSensorMeta, emptyMeta);
        return;
    }

    meta.recv_ms = recvMs;
    if (meta.coord_w <= 0) meta.coord_w = g_videoWidth.load();
    if (meta.coord_h <= 0) meta.coord_h = g_videoHeight.load();

    if (meta.frame_id >= 0) {
        g_metaRingBuffer[meta.frame_id % META_BUFFER_SIZE] = meta;
        if (meta.frame_id > g_yolo_max_frame.load()) {
            g_yolo_max_frame.store(meta.frame_id);
        }
    }

    auto metaPtr = std::make_shared<SensorMeta>(meta);
    g_latestVisionDetected.store(meta.has_detection);
    g_latestVisionTracking.store(meta.tracking);
    g_latestVisionConf.store(meta.conf);
    std::atomic_store(&g_latestSensorMeta, metaPtr);
}

// ============================================================
// 4. 视觉 UDP 接收：接入当前海康识别程序输出
// ============================================================

class UdpReceiverWorker : public QObject {
    Q_OBJECT
public:
    explicit UdpReceiverWorker(QObject* parent = nullptr) : QObject(parent) {}

public slots:
    void start() {
        socket = new QUdpSocket(this);
        bool ok = socket->bind(QHostAddress::AnyIPv4,
                               VISION_UDP_PORT,
                               QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
        if (!ok) {
            qWarning() << "[VISION UDP] bind failed on port" << VISION_UDP_PORT
                       << socket->errorString();
        } else {
            qInfo() << "[VISION UDP] listening on port" << VISION_UDP_PORT;
        }

        connect(socket, &QUdpSocket::readyRead,
                this, &UdpReceiverWorker::processPendingDatagrams);
    }

private slots:
    void processPendingDatagrams() {
        while (socket && socket->hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(static_cast<int>(socket->pendingDatagramSize()));
            socket->readDatagram(datagram.data(), datagram.size());
            mirrorTelemetryDatagram(socket, datagram, VISION_UDP_PORT);

            QJsonParseError err;
            QJsonDocument doc = QJsonDocument::fromJson(datagram, &err);
            if (!doc.isObject()) continue;

            QJsonObject obj = doc.object();
            const qint64 recvMs = metricsNowMs();
            MetricsLogger::instance().logVision(obj);
            MetricsLogger::instance().logSystemFromJson(obj, recvMs);
            updateVisionDashboardState(obj, recvMs);

            if (g_preferFrameStreamMeta.load()) {
                continue;
            }

            SensorMeta meta;
            if (!parseVisionMeta(obj, meta)) {
                // 没有框时也更新状态，防止 UI 长时间残留旧框
                auto emptyMeta = std::make_shared<SensorMeta>();
                emptyMeta->frame_id = jsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
                emptyMeta->recv_ms = recvMs;
                emptyMeta->source_ts_ms = static_cast<long long>(metricsJsonDoubleAny(obj, {"ts_ms", "source_ts_ms"}, 0.0));
                emptyMeta->infer_ms = metricsJsonDoubleAny(obj, {"infer_ms", "yolo_infer_ms"}, -1.0);
                std::atomic_store(&g_latestSensorMeta, emptyMeta);
                continue;
            }

            meta.recv_ms = recvMs;

            // 如果识别程序没有发坐标系尺寸，则默认用当前视频帧尺寸。
            if (meta.coord_w <= 0) meta.coord_w = g_videoWidth.load();
            if (meta.coord_h <= 0) meta.coord_h = g_videoHeight.load();

            if (meta.frame_id >= 0) {
                g_metaRingBuffer[meta.frame_id % META_BUFFER_SIZE] = meta;
                if (meta.frame_id > g_yolo_max_frame.load()) {
                    g_yolo_max_frame.store(meta.frame_id);
                }
            }

            auto metaPtr = std::make_shared<SensorMeta>(meta);
            g_latestVisionDetected.store(meta.has_detection);
            g_latestVisionTracking.store(meta.tracking);
            g_latestVisionConf.store(meta.conf);
            std::atomic_store(&g_latestSensorMeta, metaPtr);
        }
    }

private:
    QUdpSocket* socket = nullptr;
};

// ============================================================
// 5. 音频 UDP 接收：保留原功能
// ============================================================

class AudioReceiverWorker : public QObject {
    Q_OBJECT
public:
    explicit AudioReceiverWorker(QObject* parent = nullptr) : QObject(parent) {}

public slots:
    void start() {
        socket = new QUdpSocket(this);
        bool ok = socket->bind(QHostAddress::AnyIPv4,
                               AUDIO_UDP_PORT,
                               QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
        if (!ok) {
            qWarning() << "[AUDIO UDP] bind failed on port" << AUDIO_UDP_PORT
                       << socket->errorString();
        } else {
            qInfo() << "[AUDIO UDP] listening on port" << AUDIO_UDP_PORT;
        }

        connect(socket, &QUdpSocket::readyRead,
                this, &AudioReceiverWorker::processPendingDatagrams);
    }

private slots:
    void processPendingDatagrams() {
        while (socket && socket->hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(static_cast<int>(socket->pendingDatagramSize()));
            QHostAddress senderAddress;
            quint16 senderPort = 0;
            const qint64 bytesRead = socket->readDatagram(datagram.data(), datagram.size(), &senderAddress, &senderPort);
            if (bytesRead <= 0) continue;
            mirrorTelemetryDatagram(socket, datagram, AUDIO_UDP_PORT);
            setLatestAudioSenderHost(senderAddress);

            QJsonDocument doc = QJsonDocument::fromJson(datagram);
            if (!doc.isObject()) continue;

            QJsonObject obj = doc.object();
            MetricsLogger::instance().logAudio(obj);
            g_lastAudioRecvMs.store(metricsNowMs());

            const bool detected = metricsJsonIntAny(obj, {"audio_detected", "detected", "trigger"}, 0) != 0;
            const double smoothAngle = metricsJsonDoubleAny(obj, {"smooth_doa_deg", "doa_deg", "angle", "azimuth_deg"}, -1.0);
            const bool doaValid = metricsJsonIntAny(obj, {"doa_valid"}, (smoothAngle >= 0.0 && smoothAngle < 99998.0) ? 1 : 0) != 0;
            const bool stableDoa = metricsJsonIntAny(obj, {"stable_doa"}, detected ? 1 : 0) != 0;
            g_latestAudioDetected.store(detected);
            g_latestAudioScore.store(metricsJsonDoubleAny(obj, {"yamnet_score", "confidence", "score"}, 0.0));
            g_latestAudioScoreEma.store(metricsJsonDoubleAny(obj, {"score_ema"}, g_latestAudioScore.load()));
            g_latestAudioDoaStability.store(metricsJsonDoubleAny(obj, {"doa_stability"}, 0.0));
            g_latestAudioDoaValid.store(doaValid);
            g_latestAudioStableDoa.store(stableDoa);
            g_latestAudioDoaConfidence.store(metricsJsonDoubleAny(obj, {"doa_confidence"}, g_latestAudioDoaStability.load()));
            g_latestAudioRmsDbfs.store(metricsJsonDoubleAny(obj, {"rms_dbfs"}, -120.0));
            setDashboardText(&g_latestAudioState, metricsJsonStringAny(obj, {"audio_state"}, detected ? "CONFIRMED" : "IDLE"));
            if (obj.contains("raw_recording_active")) {
                g_audioRawRecordingActive.store(metricsJsonIntAny(obj, {"raw_recording_active"}, 0) != 0);
            }
            if (obj.contains("raw_recording_path")) {
                setCurrentAudioRawRecordingPath(obj.value("raw_recording_path").toString());
                if (g_fusedRecordingRequested.load()) {
                    writeCaptureSessionManifest("recording");
                }
            }
            const QString learnedAudioHost = latestAudioSenderHost();
            if (!learnedAudioHost.isEmpty()) {
                const bool requested = g_audioRawRecordingRequested.load();
                const bool active = g_audioRawRecordingActive.load();
                if (requested != active) {
                    const qint64 nowMs = metricsNowMs();
                    qint64 lastRetryMs = g_lastAudioControlRetryMs.load();
                    if (nowMs - lastRetryMs >= 1500 &&
                        g_lastAudioControlRetryMs.compare_exchange_strong(lastRetryMs, nowMs)) {
                        QString recordId = currentAudioRawRecordingSessionId();
                        const QString sessionId = currentCaptureSessionId();
                        if (recordId.isEmpty()) {
                            recordId = sessionId.isEmpty()
                                           ? QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")
                                           : sessionId;
                        }
                        const int retryPort = envIntBounded("ANTI_UAV_AUDIO_CONTROL_PORT", 5008, 1, 65535);
                        sendAudioRawRecordingControlDatagram(
                            learnedAudioHost,
                            retryPort,
                            requested,
                            recordId,
                            sessionId,
                            "qt_dashboard_auto_retry");
                    }
                }
            }

            if (obj.contains("audio_detected")) {
                const bool showSource = detected || (doaValid && stableDoa && g_latestAudioScoreEma.load() >= 0.15);
                g_hasAudioSignal.store(showSource);
                g_latestAudioAngle.store(static_cast<float>(smoothAngle));
            } else if (obj.contains("angle") || obj.contains("doa_deg")) {
                // 兼容只发 {"angle":xx} 或 {"doa_deg":xx} 的音频程序
                g_hasAudioSignal.store(true);
                g_latestAudioAngle.store(static_cast<float>(smoothAngle));
            }
        }
    }

private:
    QUdpSocket* socket = nullptr;
};

// ============================================================
// 6. 融合 UDP 接收：记录 YAMNet->YOLO 或融合模块上报事件
// ============================================================

class FusionReceiverWorker : public QObject {
    Q_OBJECT
public:
    explicit FusionReceiverWorker(QObject* parent = nullptr) : QObject(parent) {}

public slots:
    void start() {
        socket = new QUdpSocket(this);
        bool ok = socket->bind(QHostAddress::AnyIPv4,
                               FUSION_UDP_PORT,
                               QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
        if (!ok) {
            qWarning() << "[FUSION UDP] bind failed on port" << FUSION_UDP_PORT
                       << socket->errorString();
        } else {
            qInfo() << "[FUSION UDP] listening on port" << FUSION_UDP_PORT;
        }

        connect(socket, &QUdpSocket::readyRead,
                this, &FusionReceiverWorker::processPendingDatagrams);
    }

private slots:
    void processPendingDatagrams() {
        while (socket && socket->hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(static_cast<int>(socket->pendingDatagramSize()));
            socket->readDatagram(datagram.data(), datagram.size());
            mirrorTelemetryDatagram(socket, datagram, FUSION_UDP_PORT);

            QJsonParseError err;
            QJsonDocument doc = QJsonDocument::fromJson(datagram, &err);
            if (!doc.isObject()) continue;
            MetricsLogger::instance().logFusionUdp(doc.object());
        }
    }

private:
    QUdpSocket* socket = nullptr;
};

class RidReceiverWorker : public QObject {
    Q_OBJECT
public:
    explicit RidReceiverWorker(QObject* parent = nullptr) : QObject(parent) {}

public slots:
    void start() {
        socket = new QUdpSocket(this);
        const quint16 port = static_cast<quint16>(
            envIntBounded("ANTI_UAV_RID900_PORT", RID900_UDP_PORT, 1, 65535));
        const bool ok = socket->bind(
            QHostAddress::AnyIPv4,
            port,
            QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
        if (!ok) {
            qWarning() << "[RID900 UDP] bind failed on port" << port
                       << socket->errorString();
        } else {
            qInfo() << "[RID900 UDP] listening on port" << port;
        }
        connect(socket, &QUdpSocket::readyRead,
                this, &RidReceiverWorker::processPendingDatagrams);
    }

private slots:
    void processPendingDatagrams() {
        while (socket && socket->hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(static_cast<int>(socket->pendingDatagramSize()));
            socket->readDatagram(datagram.data(), datagram.size());
            mirrorTelemetryDatagram(socket, datagram, RID900_UDP_PORT);

            const QJsonDocument doc = QJsonDocument::fromJson(datagram);
            if (!doc.isObject()) continue;
            const QJsonObject obj = doc.object();
            const QString type = obj.value("type").toString();
            const qint64 now = metricsNowMs();
            g_lastRidRecvMs.store(now);

            QMutexLocker locker(&g_ridMutex);
            if (type == "rid900_status") {
                const bool connected = obj.value("connected").toInt() != 0;
                g_ridConnected.store(connected);
                g_ridSource = obj.value("source").toString("--");
                const QString message = obj.value("message").toString();
                g_ridStatusText = connected ? "RID900 已连接"
                                            : (message.isEmpty() ? "RID900 未连接" : message);
            } else if (type == "rid900_heartbeat") {
                g_ridConnected.store(true);
                g_lastRidDataMs.store(now);
                g_ridStatusText = "RID900 心跳正常";
                g_ridSource = obj.value("source").toString(g_ridSource);
                g_ridDeviceId = obj.value("device_id").toString("--");
                g_ridModulePosition = QString("%1, %2 / %3 m")
                                          .arg(obj.value("module_longitude").toDouble(), 0, 'f', 6)
                                          .arg(obj.value("module_latitude").toDouble(), 0, 'f', 6)
                                          .arg(obj.value("module_altitude_m").toDouble(), 0, 'f', 1);
            } else if (type == "rid900_target") {
                g_ridConnected.store(true);
                g_lastRidDataMs.store(now);
                g_ridStatusText = "收到 Remote ID 目标";
                g_ridSource = obj.value("source").toString(g_ridSource);
                RidTargetState target;
                target.serialNumber = obj.value("serial_number").toString();
                target.vendor = obj.value("vendor").toString();
                target.productType = obj.value("product_type").toString();
                target.rssiDbm = obj.value("detect_rssi_dbm").toInt();
                target.snrDb = obj.value("detect_snr_db").toInt();
                target.longitude = obj.value("drone_longitude").toDouble();
                target.latitude = obj.value("drone_latitude").toDouble();
                target.heightM = obj.value("height_m").toDouble();
                target.speedMps = obj.value("horizontal_speed_mps").toDouble();
                target.yawDeg = obj.value("yaw_deg").toInt();
                target.lastSeenMs = now;
                const QString key = target.serialNumber.isEmpty()
                                        ? QString("unknown_%1").arg(now)
                                        : target.serialNumber;
                g_ridTargets.insert(key, target);
            } else if (type == "rid900_parse_error") {
                g_ridStatusText = "RID900 报文解析异常";
            }

            for (auto it = g_ridTargets.begin(); it != g_ridTargets.end();) {
                if (now - it.value().lastSeenMs > 30000) {
                    it = g_ridTargets.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

private:
    QUdpSocket* socket = nullptr;
};

class SyncWorker : public QObject {
    Q_OBJECT
public:
    explicit SyncWorker(QObject* parent = nullptr) : QObject(parent) {}

public slots:
    void start() {
        socket = new QUdpSocket(this);
        const quint16 port = static_cast<quint16>(
            envIntBounded("ANTI_UAV_SYNC_PORT", 5012, 1, 65535));
        const bool ok = socket->bind(
            QHostAddress::AnyIPv4,
            port,
            QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
        if (!ok) {
            qWarning() << "[SYNC] bind failed on port" << port << socket->errorString();
        } else {
            qInfo() << "[SYNC] role=" << antiUavNodeRole() << "listening on" << port;
        }
        connect(socket, &QUdpSocket::readyRead,
                this, &SyncWorker::processPendingDatagrams);

        heartbeatTimer = new QTimer(this);
        heartbeatTimer->setInterval(1000);
        connect(heartbeatTimer, &QTimer::timeout, this, &SyncWorker::sendHeartbeat);
        heartbeatTimer->start();
        sendHeartbeat();

        telemetryTimer = new QTimer(this);
        telemetryTimer->setInterval(40);
        connect(telemetryTimer, &QTimer::timeout, this, &SyncWorker::flushTelemetry);
        telemetryTimer->start();
    }

private slots:
    void sendHeartbeat() {
        if (!socket || antiUavNodeRole() != "windows") return;
        QString boardHost =
            QString::fromUtf8(qgetenv("ANTI_UAV_SYNC_BOARD_HOST")).trimmed();
        if (boardHost.isEmpty()) boardHost = "DEVICE_IP";
        QJsonObject hello;
        hello["type"] = "antiuav_sync_hello";
        hello["node"] = "windows";
        hello["ts_ms"] = static_cast<double>(metricsNowMs());
        socket->writeDatagram(
            QJsonDocument(hello).toJson(QJsonDocument::Compact),
            QHostAddress(boardHost),
            static_cast<quint16>(envIntBounded("ANTI_UAV_SYNC_PORT", 5012, 1, 65535)));
    }

    void flushTelemetry() {
        if (!socket || antiUavNodeRole() != "elf2") return;
        const qint64 lastSyncMs = g_lastWindowsSyncMs.load();
        if (lastSyncMs <= 0 || metricsNowMs() - lastSyncMs > 4000) return;

        const QString peerHost = windowsSyncHost();
        if (peerHost.isEmpty()) return;
        const QHostAddress peerAddress(peerHost);
        if (peerAddress.isNull()) return;

        const quint16 syncPort = static_cast<quint16>(
            envIntBounded("ANTI_UAV_SYNC_PORT", 5012, 1, 65535));
        const QHash<quint16, QByteArray> pending = takeLatestSyncTelemetry();
        for (auto it = pending.constBegin(); it != pending.constEnd(); ++it) {
            QJsonObject wrapper;
            wrapper["type"] = "antiuav_sync_telemetry";
            wrapper["node"] = "elf2";
            wrapper["port"] = static_cast<int>(it.key());
            wrapper["ts_ms"] = static_cast<double>(metricsNowMs());

            const QJsonDocument payloadDoc = QJsonDocument::fromJson(it.value());
            if (payloadDoc.isObject()) {
                wrapper["payload"] = payloadDoc.object();
            } else {
                wrapper["payload_text"] = QString::fromUtf8(it.value());
            }
            socket->writeDatagram(
                QJsonDocument(wrapper).toJson(QJsonDocument::Compact),
                peerAddress,
                syncPort);
        }
    }

    void processPendingDatagrams() {
        while (socket && socket->hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(static_cast<int>(socket->pendingDatagramSize()));
            QHostAddress senderAddress;
            quint16 senderPort = 0;
            if (socket->readDatagram(
                    datagram.data(), datagram.size(), &senderAddress, &senderPort) <= 0) {
                continue;
            }
            const QJsonDocument doc = QJsonDocument::fromJson(datagram);
            if (!doc.isObject()) continue;
            const QJsonObject obj = doc.object();
            const QString type = obj.value("type").toString();
            if (type == "antiuav_rk_capture_control" && antiUavNodeRole() == "elf2") {
                const QString action = obj.value("action").toString().trimmed().toLower();
                const QString sessionId = obj.value("session_id").toString();
                const QString source = obj.value("source").toString("sync_control");
                handleRkSyncCaptureControl(action, sessionId, source);
                continue;
            }
            if (type == "antiuav_sync_recording_control" && antiUavNodeRole() == "windows") {
                const QString action = obj.value("action").toString();
                const QString sessionId = obj.value("session_id").toString();
                if (action == "start") {
                    if (recordingStorageAllowed("Windows本地录像", false)) {
                        ensureCaptureSessionForRecording(sessionId);
                        g_recordingPaused.store(false);
                        g_recordingRequested.store(true);
                        setWindowsRecordingStatus("录制中");
                    }
                } else if (action == "pause") {
                    if (g_recordingRequested.load()) {
                        g_recordingPaused.store(true);
                        setWindowsRecordingStatus("暂停");
                    }
                } else if (action == "resume") {
                    if (g_recordingRequested.load()) {
                        g_recordingPaused.store(false);
                        setWindowsRecordingStatus("录制中");
                    }
                } else if (action == "stop") {
                    g_recordingPaused.store(false);
                    g_recordingRequested.store(false);
                    setWindowsRecordingStatus("正在保存");
                }
                g_lastWindowsSyncMs.store(metricsNowMs());
                continue;
            }
            if (type == "antiuav_sync_telemetry" && antiUavNodeRole() == "windows") {
                const int port = obj.value("port").toInt();
                if (port != VISION_UDP_PORT &&
                    port != AUDIO_UDP_PORT &&
                    port != FUSION_UDP_PORT &&
                    port != RID900_UDP_PORT) {
                    continue;
                }

                QByteArray localPayload;
                if (obj.value("payload").isObject()) {
                    localPayload = QJsonDocument(obj.value("payload").toObject())
                                       .toJson(QJsonDocument::Compact);
                } else {
                    localPayload = obj.value("payload_text").toString().toUtf8();
                }
                if (!localPayload.isEmpty()) {
                    socket->writeDatagram(
                        localPayload,
                        QHostAddress::LocalHost,
                        static_cast<quint16>(port));
                    g_lastWindowsSyncMs.store(metricsNowMs());
                }
                continue;
            }
            if (type != "antiuav_sync_hello" && type != "antiuav_sync_ack") continue;

            g_lastWindowsSyncMs.store(metricsNowMs());
            setWindowsSyncHost(senderAddress.toString());
            if (antiUavNodeRole() == "elf2" && type == "antiuav_sync_hello") {
                QJsonObject ack;
                ack["type"] = "antiuav_sync_ack";
                ack["node"] = "elf2";
                ack["ts_ms"] = static_cast<double>(metricsNowMs());
                socket->writeDatagram(
                    QJsonDocument(ack).toJson(QJsonDocument::Compact),
                    senderAddress,
                    senderPort);
            }
        }
    }

private:
    QUdpSocket* socket = nullptr;
    QTimer* heartbeatTimer = nullptr;
    QTimer* telemetryTimer = nullptr;
};

class DashboardWindow : public QWidget {
public:
    explicit DashboardWindow(QWidget* parent = nullptr)
        : QWidget(parent) {
        setFocusPolicy(Qt::StrongFocus);
        qApp->installEventFilter(this);
        controlSocket.bind(
            QHostAddress(QHostAddress::AnyIPv4),
            static_cast<quint16>(0));
        connect(&controlSocket, &QUdpSocket::readyRead, this, [this]() {
            while (controlSocket.hasPendingDatagrams()) {
                QByteArray datagram;
                datagram.resize(static_cast<int>(controlSocket.pendingDatagramSize()));
                controlSocket.readDatagram(datagram.data(), datagram.size());
                const QJsonDocument doc = QJsonDocument::fromJson(datagram);
                if (!doc.isObject()) continue;
                const QJsonObject obj = doc.object();
                if (obj.value("type").toString() != "qt_ptz_ack") continue;
                const bool ok = obj.value("ok").toInt() != 0;
                const QString command = obj.value("command").toString();
                const QString message = obj.value("message").toString();
                g_latestCameraAzimuthUnit.store(obj.value("pan").toInt());
                g_latestCameraElevationUnit.store(obj.value("tilt").toInt());
                g_latestCameraAbsoluteZoom.store(obj.value("zoom").toInt(10));
                g_latestAutoTrackingEnabled.store(obj.value("auto_tracking").toInt() != 0);
                setQtPtzStatusText(QString("%1 %2: %3")
                                       .arg(ok ? "OK" : "FAIL", command, message));
            }
        });
        manualRepeatTimer.setInterval(MANUAL_REPEAT_MS);
        connect(&manualRepeatTimer, &QTimer::timeout, this, [this]() {
            if (touchMoveActive) {
                sendQtPtzCommand("move", touchPan, touchTilt, touchZoom,
                                 manualPtzSpeed, manualZoomSpeed, "touch_hold");
            } else if (activeKeys.isEmpty()) {
                sendQtManualStop("timer_empty_stop");
                return;
            } else {
                updatePtzCommand();
            }
        });
        windowsReconnectTimer.setInterval(3000);
        connect(&windowsReconnectTimer, &QTimer::timeout, this, [this]() {
            if (antiUavNodeRole() != "elf2" || !g_windowsAutoReconnect.load()) return;
            const qint64 now = metricsNowMs();
            if (dashboardIsRecent(g_lastWindowsSyncMs.load(), now, 5000)) return;
            sendWindowsAgentCommand("start", 700, 1200);
        });
        windowsReconnectTimer.start();
    }

    ~DashboardWindow() override {
        qApp->removeEventFilter(this);
    }

    void requestToggleRecording() {
        toggleRecording();
    }

    void requestToggleRawRecording() {
        toggleRawRecording();
    }

    void requestToggleAudioRawRecording() {
        toggleAudioRawRecording();
    }

    void requestToggleFusedRecording() {
        toggleFusedRecording();
    }

    void requestToggleWindowsRecording() {
        if (antiUavNodeRole() == "windows") {
            if (!g_recordingRequested.load() &&
                !recordingStorageAllowed("Windows本地录制", false)) {
                return;
            }
            ensureCaptureSessionForRecording();
            toggleRecording();
            setWindowsRecordingStatus(g_recordingRequested.load() ? "录制中" : "正在保存");
            return;
        }
        const bool next = !g_windowsRemoteRecordingRequested.load();
        const QString action = next ? "start" : "stop";
        if (sendWindowsRecordingControl(action)) {
            g_windowsRemoteRecordingRequested.store(next);
            setWindowsRecordingStatus(next ? "录制中" : "正在保存");
        }
    }

    void requestPauseWindowsRecording() {
        if (antiUavNodeRole() == "windows") {
            if (g_recordingRequested.load()) {
                g_recordingPaused.store(true);
                setWindowsRecordingStatus("暂停");
            }
            return;
        }
        if (sendWindowsRecordingControl("pause")) {
            setWindowsRecordingStatus("暂停");
        }
    }

    void requestResumeWindowsRecording() {
        if (antiUavNodeRole() == "windows") {
            if (g_recordingRequested.load()) {
                g_recordingPaused.store(false);
                setWindowsRecordingStatus("录制中");
            }
            return;
        }
        if (sendWindowsRecordingControl("resume")) {
            setWindowsRecordingStatus("录制中");
        }
    }

    void requestToggleSyncRecording() {
        const bool next = !g_fusedRecordingRequested.load();
        if (next) {
            if (!recordingStorageAllowed("RK原始音视频同步采集", true)) return;
            if (!g_fusedRecordingRequested.load()) {
                toggleFusedRecording();
            }
            setWindowsRecordingStatus("Windows本地录像独立控制");
        } else {
            if (g_fusedRecordingRequested.load()) {
                toggleFusedRecording();
            }
        }
    }

    void requestToggleRkCapturePause() {
        if (!g_fusedRecordingRequested.load()) {
            return;
        }
        const QString sessionId = currentCaptureSessionId();
        if (sessionId.isEmpty()) {
            return;
        }
        if (!g_recordingPaused.load()) {
            appendRecordingTimelineEvent("sync_raw_capture_pause", "qt");
            g_recordingPaused.store(true);
            stopRawVideoRecording();
            stopAudioRawRecording(sessionId);
            writeCaptureSessionManifest("paused");
            writeCaptureSessionSummary("paused");
            qInfo() << "[REC] RK raw audio/video sync capture paused:" << sessionId;
        } else {
            appendRecordingTimelineEvent("sync_raw_capture_resume", "qt");
            g_recordingPaused.store(false);
            startRawVideoRecording(sessionId);
            startAudioRawRecording(sessionId);
            writeCaptureSessionManifest("recording");
            writeCaptureSessionSummary("recording");
            qInfo() << "[REC] RK raw audio/video sync capture resumed:" << sessionId;
        }
    }

    void requestStopAllRecordings() {
        if (g_fusedRecordingRequested.load()) {
            toggleFusedRecording();
        } else {
            if (g_rawRecordingRequested.load()) stopRawVideoRecording();
            if (g_audioRawRecordingRequested.load()) {
                stopAudioRawRecording(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
            }
        }
        g_recordingPaused.store(false);
        g_recordingRequested.store(false);
        if (g_windowsRemoteRecordingRequested.load()) {
            sendWindowsRecordingControl("stop");
            g_windowsRemoteRecordingRequested.store(false);
        }
        setWindowsRecordingStatus("正在保存");
    }

    void requestOpenRecordingDirectory() {
        QString path = currentRecordingPath();
        if (path.isEmpty()) path = lastRecordingPath();
        QString dir = QFileInfo(path).absolutePath();
        if (dir.isEmpty() || dir == ".") {
            dir = windowsRecordingDirPath();
        }
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    }

    void requestToggleDetectionEnabled() {
        toggleDetectionEnabled();
    }

    void requestToggleDisplayEcoMode() {
        toggleDisplayEcoMode();
    }

    void requestEmergencyStop() {
        activeKeys.clear();
        touchMoveActive = false;
        sendPtzAction("emergency_stop");
        sendQtManualStop("ui_emergency_stop");
    }

    void requestExitToDesktop() {
        if (exitToDesktopRequested) {
            return;
        }
        exitToDesktopRequested = true;
        requestEmergencyStop();
        QString returnFlagPath =
            QString::fromUtf8(qgetenv("ANTI_UAV_RETURN_TO_MATRIX_FILE")).trimmed();
        if (returnFlagPath.isEmpty()) {
            returnFlagPath = antiUavDataRootPath() + "/runtime/pid/return_to_matrix.flag";
        }
        if (!returnFlagPath.isEmpty()) {
            QDir().mkpath(QFileInfo(returnFlagPath).absolutePath());
            QFile flag(returnFlagPath);
            if (flag.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                flag.write("return_to_matrix\n");
            }
        }
#ifndef Q_OS_WIN
        QString root = QString::fromUtf8(qgetenv("ANTI_UAV_ROOT")).trimmed();
        if (root.isEmpty()) root = "/home/elf/AntiUAV_Data";
        QProcess::startDetached("/bin/bash", {root + "/bin/stop_all.sh", root});
        QTimer::singleShot(2500, qApp, &QCoreApplication::quit);
#else
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
#endif
    }

    void requestStopSystem() {
        requestEmergencyStop();
        setCurrentCaptureSession("", "");
        g_hasVideoSignal.store(false);
        g_fusedRecordingActive.store(false);
#ifdef Q_OS_WIN
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
#else
        QString root = QString::fromUtf8(qgetenv("ANTI_UAV_ROOT")).trimmed();
        if (root.isEmpty()) root = "/home/elf/AntiUAV_Data";
        QProcess::startDetached("/bin/bash", {root + "/bin/stop_all.sh", "--keep-qt", root});
#endif
    }

    void requestRestoreSystem() {
#ifdef Q_OS_WIN
        return;
#else
        setCurrentCaptureSession("", "");
        QString root = QString::fromUtf8(qgetenv("ANTI_UAV_ROOT")).trimmed();
        if (root.isEmpty()) root = "/home/elf/AntiUAV_Data";
        QProcess::startDetached(
            "/bin/bash",
            {root + "/bin/run_all.sh", "--skip-qt", "--new-session"});
#endif
    }

    void requestStartWindowsClient() {
        if (antiUavNodeRole() == "windows") {
            setWindowsLaunchStatus("本端已经是 Windows 界面");
            return;
        }
        sendWindowsAgentCommand("start", 1200, 3000);
    }

    void requestDisconnectWindowsClient() {
        if (antiUavNodeRole() == "windows") {
            QTimer::singleShot(0, qApp, &QCoreApplication::quit);
            return;
        }
        sendWindowsAgentCommand("stop", 1200, 2500);
    }

    void requestToggleWindowsAutoReconnect() {
        const bool next = !g_windowsAutoReconnect.load();
        g_windowsAutoReconnect.store(next);
        setWindowsLaunchStatus(next ? "自动重连已开启" : "自动重连已关闭");
        if (next) {
            requestStartWindowsClient();
        }
    }

    void requestTouchMove(int pan, int tilt, int zoom) {
        touchPan = qBound(-1, pan, 1);
        touchTilt = qBound(-1, tilt, 1);
        touchZoom = qBound(-1, zoom, 1);
        touchMoveActive = touchPan != 0 || touchTilt != 0 || touchZoom != 0;
        if (!touchMoveActive) {
            requestTouchStop("touch_zero");
            return;
        }
        sendQtPtzCommand("move", touchPan, touchTilt, touchZoom,
                         manualPtzSpeed, manualZoomSpeed, "touch_press");
        if (!manualRepeatTimer.isActive()) manualRepeatTimer.start();
    }

    void requestTouchStop(const QString& reason = "touch_release") {
        touchMoveActive = false;
        touchPan = 0;
        touchTilt = 0;
        touchZoom = 0;
        sendQtManualStop(reason);
    }

    void setTouchSpeed(int speed) {
        manualPtzSpeed = qBound(10, speed, 100);
        manualZoomSpeed = qBound(30, speed * 2, QT_MANUAL_ZOOM_SPEED_MAX);
    }

    void requestAutofocus() {
        sendPtzAction("request_autofocus");
    }

    void requestSetAutoTracking(bool enabled) {
        QJsonObject extra;
        extra["enabled"] = enabled ? 1 : 0;
        sendPtzAction("set_auto_track", extra);
    }

    void requestSetControlParams(const QJsonObject& params) {
        sendPtzAction("set_control_params", params);
    }

    void requestSetAudioGuidance(bool enabled) {
        g_audioGuidanceRequested.store(enabled);
        QJsonObject extra;
        extra["enabled"] = enabled ? 1 : 0;
        sendPtzAction("set_audio_guidance", extra);
    }

    void requestMarkAudioCalibrationTarget() {
        sendPtzAction("mark_audio_calibration_target");
    }

    void requestResetAudioCalibration(bool resetRuntimeLearning = false) {
        QJsonObject extra;
        extra["runtime_learning"] = resetRuntimeLearning ? 1 : 0;
        sendPtzAction(resetRuntimeLearning ? "reset_audio_learning"
                                           : "reset_audio_calibration",
                      extra);
    }

    void requestSaveHome() {
        sendPtzAction("save_home");
    }

    void requestReturnInitial() {
        sendPtzAction("return_initial");
    }

    void requestReturnSavedHome() {
        sendPtzAction("return_home");
    }

    void requestGotoPreset(int preset) {
        QJsonObject extra;
        extra["preset"] = preset;
        sendPtzAction("goto_preset", extra);
    }

    void requestCenterOnPoint(double normalizedX, double normalizedY) {
        QJsonObject extra;
        extra["x"] = normalizedX;
        extra["y"] = normalizedY;
        sendPtzAction("click_center", extra);
    }

    void requestCameraDayMode() {
        sendCameraModeCommand("day");
    }

    void requestCameraNightMode() {
        sendCameraModeCommand("night");
    }

    void requestCameraAutoDayNight() {
        sendCameraModeCommand("auto");
    }

    void requestCameraProfile(const QString& profile) {
        sendCameraProfileCommand(profile);
    }

    QString manualPtzEndpointText() const {
        return QString("%1:%2").arg(qtManualControlHost).arg(qtManualControlPort);
    }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (handleKeyPress(event)) {
            event->accept();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void keyReleaseEvent(QKeyEvent* event) override {
        if (handleKeyRelease(event)) {
            event->accept();
            return;
        }
        QWidget::keyReleaseEvent(event);
    }

    bool eventFilter(QObject* watched, QEvent* event) override {
        Q_UNUSED(watched);
        if (event->type() == QEvent::ApplicationDeactivate ||
            event->type() == QEvent::WindowDeactivate ||
            event->type() == QEvent::TouchCancel ||
            event->type() == QEvent::UngrabMouse) {
            if (touchMoveActive || !activeKeys.isEmpty()) {
                activeKeys.clear();
                requestTouchStop("window_or_touch_cancel");
            }
        }
        if (!isActiveWindow()) {
            return false;
        }

        if (event->type() == QEvent::KeyPress) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            return handleKeyPress(keyEvent);
        }
        if (event->type() == QEvent::KeyRelease) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            return handleKeyRelease(keyEvent);
        }

        return false;
    }

private:
    static constexpr int QT_MANUAL_ZOOM_SPEED = 120;
    static constexpr int QT_MANUAL_ZOOM_SPEED_MAX = 180;
    static constexpr int MANUAL_REPEAT_MS = 80;

    QSet<int> activeKeys;
    QTimer manualRepeatTimer;
    QTimer windowsReconnectTimer;
    QUdpSocket controlSocket;
    bool touchMoveActive = false;
    int touchPan = 0;
    int touchTilt = 0;
    int touchZoom = 0;
    int manualPtzSpeed = envIntBounded("ANTI_UAV_QT_MANUAL_SPEED_DEFAULT", 60, 10, 100);
    int manualZoomSpeed = qBound(30, manualPtzSpeed * 2, QT_MANUAL_ZOOM_SPEED_MAX);
    QString controlHost = QString::fromStdString(
        envStringOrDefault("ANTI_UAV_RK_CONTROL_HOST", defaultBoardHostForThisNode()));
    QString qtManualControlHost = QString::fromStdString(
        envStringOrDefault("ANTI_UAV_QT_MANUAL_HOST", controlHost.toStdString()));
    int yoloControlPort = envIntBounded("ANTI_UAV_YOLO_CONTROL_PORT", 5007, 1, 65535);
    int audioControlPort = envIntBounded("ANTI_UAV_AUDIO_CONTROL_PORT", 5008, 1, 65535);
    int qtManualControlPort = envIntBounded("ANTI_UAV_QT_MANUAL_PORT", 5011, 1, 65535);
    QString windowsAgentHost = QString::fromStdString(
        envStringOrDefault("ANTI_UAV_WINDOWS_AGENT_HOST", "DEVICE_IP"));
    int windowsAgentPort = envIntBounded("ANTI_UAV_WINDOWS_AGENT_PORT", 5022, 1, 65535);

    bool sendWindowsAgentCommand(const QString& command,
                                 int connectTimeoutMs,
                                 int replyTimeoutMs) {
        QTcpSocket socket;
        socket.connectToHost(windowsAgentHost, static_cast<quint16>(windowsAgentPort));
        if (!socket.waitForConnected(connectTimeoutMs)) {
            setWindowsLaunchStatus(QString("Windows 代理不可达 %1:%2")
                                       .arg(windowsAgentHost)
                                       .arg(windowsAgentPort));
            return false;
        }

        QJsonObject request;
        request["type"] = "antiuav_windows_agent_request";
        request["command"] = command;
        request["source"] = "elf2_mipi";
        request["ts_ms"] = static_cast<double>(metricsNowMs());
        const QByteArray payload = QJsonDocument(request).toJson(QJsonDocument::Compact) + "\n";
        socket.write(payload);
        if (!socket.waitForBytesWritten(1000)) {
            setWindowsLaunchStatus("Windows 代理写入失败");
            return false;
        }
        if (!socket.waitForReadyRead(replyTimeoutMs)) {
            setWindowsLaunchStatus("Windows 代理无响应");
            return false;
        }
        const QByteArray reply = socket.readAll();
        const QJsonDocument doc = QJsonDocument::fromJson(reply.trimmed());
        if (!doc.isObject()) {
            setWindowsLaunchStatus("Windows 代理响应格式错误");
            return false;
        }
        const QJsonObject obj = doc.object();
        const bool ok = obj.value("ok").toInt(0) != 0;
        const QString message = obj.value("message").toString(ok ? "OK" : "FAIL");
        const int pid = obj.value("pid").toInt(0);
        if (ok && command == "start") {
            setWindowsLaunchStatus(pid > 0
                                       ? QString("Windows 界面已请求打开 PID %1").arg(pid)
                                       : "Windows 界面已请求打开");
        } else if (ok && (command == "stop" || command == "disconnect")) {
            setWindowsLaunchStatus("Windows 界面已断开");
        } else {
            setWindowsLaunchStatus(message);
        }
        return ok;
    }

    bool sendWindowsRecordingControl(const QString& action) {
        QString peerHost = windowsSyncHost();
        if (peerHost.isEmpty()) peerHost = windowsAgentHost;
        const QHostAddress peerAddress(peerHost);
        if (peerAddress.isNull()) {
            setWindowsLaunchStatus("Windows 同步地址无效");
            return false;
        }
        const QString sessionId = ensureCaptureSessionForRecording();
        QJsonObject obj;
        obj["type"] = "antiuav_sync_recording_control";
        obj["node"] = "elf2";
        obj["action"] = action;
        obj["session_id"] = sessionId;
        obj["data_disk"] = dataDiskStatusText();
        obj["ts_ms"] = static_cast<double>(metricsNowMs());
        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const quint16 syncPort = static_cast<quint16>(
            envIntBounded("ANTI_UAV_SYNC_PORT", 5012, 1, 65535));
        const qint64 sent = controlSocket.writeDatagram(payload, peerAddress, syncPort);
        if (sent < 0) {
            setWindowsLaunchStatus("Windows 录制控制发送失败: " + controlSocket.errorString());
            return false;
        }
        setWindowsLaunchStatus(QString("Windows录制 %1 -> %2:%3")
                                   .arg(action, peerHost)
                                   .arg(syncPort));
        return true;
    }

    bool handleKeyPress(QKeyEvent* event) {
        if (isExperimentEditorFocused()) {
            return false;
        }
        if (event->isAutoRepeat()) {
            return true;
        }

        const int key = event->key();
        if (key == Qt::Key_X) {
            toggleRecording();
            return true;
        }
        if (key == Qt::Key_B) {
            toggleRawRecording();
            return true;
        }
        if (key == Qt::Key_M) {
            toggleAudioRawRecording();
            return true;
        }
        if (key == Qt::Key_F) {
            toggleFusedRecording();
            return true;
        }
        if (key == Qt::Key_T) {
            toggleDetectionEnabled();
            return true;
        }
        if (key == Qt::Key_L) {
            toggleDisplayEcoMode();
            return true;
        }

        if (key == Qt::Key_Space || key == Qt::Key_Escape) {
            activeKeys.clear();
            sendQtManualStop("key_stop");
            return true;
        }

        if (isMotionKey(key)) {
            activeKeys.insert(key);
            updatePtzCommand();
            if (!manualRepeatTimer.isActive()) {
                manualRepeatTimer.start();
            }
            return true;
        }

        return false;
    }

    bool handleKeyRelease(QKeyEvent* event) {
        if (isExperimentEditorFocused()) {
            return false;
        }
        if (event->isAutoRepeat()) {
            return true;
        }

        const int key = event->key();
        if (isMotionKey(key)) {
            activeKeys.remove(key);
            if (activeKeys.isEmpty()) {
                sendQtManualStop("key_release_stop");
            } else {
                updatePtzCommand();
            }
            return true;
        }

        return false;
    }

    bool hasAnyKey(std::initializer_list<int> keys) const {
        for (int key : keys) {
            if (activeKeys.contains(key)) return true;
        }
        return false;
    }

    bool isMotionKey(int key) const {
        switch (key) {
        case Qt::Key_W:
        case Qt::Key_A:
        case Qt::Key_S:
        case Qt::Key_D:
        case Qt::Key_Up:
        case Qt::Key_Down:
        case Qt::Key_Left:
        case Qt::Key_Right:
        case Qt::Key_Plus:
        case Qt::Key_Equal:
        case Qt::Key_Minus:
        case Qt::Key_Underscore:
            return true;
        default:
            return false;
        }
    }

    bool isExperimentEditorFocused() const {
        QWidget* focus = qApp->focusWidget();
        return qobject_cast<QLineEdit*>(focus) != nullptr ||
               qobject_cast<QComboBox*>(focus) != nullptr;
    }

    void updatePtzCommand() {
        int pan = 0;
        int tilt = 0;
        int zoom = 0;

        if (hasAnyKey({Qt::Key_D, Qt::Key_Right})) pan += 1;
        if (hasAnyKey({Qt::Key_A, Qt::Key_Left})) pan -= 1;
        if (hasAnyKey({Qt::Key_W, Qt::Key_Up})) tilt += 1;
        if (hasAnyKey({Qt::Key_S, Qt::Key_Down})) tilt -= 1;
        if (hasAnyKey({Qt::Key_Plus, Qt::Key_Equal})) zoom += 1;
        if (hasAnyKey({Qt::Key_Minus, Qt::Key_Underscore})) zoom -= 1;

        pan = std::max(-1, std::min(1, pan));
        tilt = std::max(-1, std::min(1, tilt));
        zoom = std::max(-1, std::min(1, zoom));

        if (pan == 0 && tilt == 0 && zoom == 0) {
            sendQtManualStop("zero_direction_stop");
            return;
        }

        sendQtPtzCommand("move", pan, tilt, zoom, manualPtzSpeed, manualZoomSpeed);
    }

    void sendQtPtzCommand(const QString& cmd, int pan, int tilt, int zoom,
                          int speed, int zoomSpeed, const QString& reason = "manual_key") {
        QJsonObject obj;
        obj["type"] = "qt_ptz_cmd";
        obj["cmd"] = cmd;
        obj["pan"] = pan;
        obj["tilt"] = tilt;
        obj["zoom"] = zoom;
        obj["speed"] = speed;
        obj["zoom_speed"] = zoomSpeed;
        obj["manual"] = true;
        obj["source"] = antiUavNodeRole();
        obj["control_mode"] = cmd == "stop" ? "stop" : "manual";
        obj["reason"] = reason;
        const double commandTsMs = static_cast<double>(metricsNowMs());
        obj["seq"] = commandTsMs * 1000.0 + static_cast<double>((++g_qtControlSeq) % 1000);
        obj["timestamp_ms"] = commandTsMs;
        obj["ts_ms"] = commandTsMs;

        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const qint64 sent = controlSocket.writeDatagram(
            payload,
            QHostAddress(qtManualControlHost),
            static_cast<quint16>(qtManualControlPort)
        );
        const QString status = QString("%1 p=%2 t=%3 z=%4 -> %5:%6")
                                   .arg(cmd)
                                   .arg(pan)
                                   .arg(tilt)
                                   .arg(zoom)
                                   .arg(qtManualControlHost)
                                   .arg(qtManualControlPort);
        if (sent < 0) {
            qWarning() << "[QT_CTRL] manual UDP send failed"
                       << qtManualControlHost << qtManualControlPort
                       << controlSocket.errorString();
            setQtPtzStatusText("FAILED " + status);
        } else {
            setQtPtzStatusText(status);
        }
    }

    void sendQtManualStop(const QString& reason) {
        if (manualRepeatTimer.isActive()) {
            manualRepeatTimer.stop();
        }
        sendQtPtzCommand("stop", 0, 0, 0, 0, 0, reason);
    }

    void sendPtzAction(const QString& command, const QJsonObject& extra = QJsonObject()) {
        QJsonObject obj = extra;
        obj["type"] = "qt_ptz_cmd";
        obj["cmd"] = command;
        obj["manual"] = true;
        obj["source"] = antiUavNodeRole();
        obj["control_mode"] = command == "emergency_stop" ? "emergency" :
                              (command == "set_audio_guidance" ? "audio" :
                               (command == "set_auto_track" || command == "set_control_params" ||
                                command == "request_autofocus" ? "auto" : "manual"));
        const double commandTsMs = static_cast<double>(metricsNowMs());
        obj["seq"] = commandTsMs * 1000.0 + static_cast<double>((++g_qtControlSeq) % 1000);
        obj["timestamp_ms"] = commandTsMs;
        obj["ts_ms"] = commandTsMs;
        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const qint64 sent = controlSocket.writeDatagram(
            payload, QHostAddress(qtManualControlHost),
            static_cast<quint16>(qtManualControlPort));
        if (sent < 0) {
            setQtPtzStatusText("FAILED " + command + ": " + controlSocket.errorString());
        } else {
            setQtPtzStatusText("SENT " + command);
        }
    }

    void sendCameraModeCommand(const QString& mode) {
        QJsonObject obj;
        obj["type"] = "qt_camera_mode";
        obj["mode"] = mode;
        obj["source"] = antiUavNodeRole();
        obj["ts_ms"] = static_cast<double>(metricsNowMs());

        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const qint64 sent = controlSocket.writeDatagram(
            payload,
            QHostAddress(qtManualControlHost),
            static_cast<quint16>(qtManualControlPort)
        );
        const QString status = QString("%1 -> %2:%3")
                                   .arg(mode)
                                   .arg(qtManualControlHost)
                                   .arg(qtManualControlPort);
        if (sent < 0) {
            qWarning() << "[QT_CTRL] camera mode UDP send failed"
                       << qtManualControlHost << qtManualControlPort
                       << controlSocket.errorString();
            setQtCameraModeStatusText("FAILED " + status);
        } else {
            setQtCameraModeStatusText(status);
        }
    }

    void sendCameraProfileCommand(const QString& profile) {
        QJsonObject obj;
        obj["type"] = "qt_camera_profile";
        obj["profile"] = profile;
        obj["source"] = antiUavNodeRole();
        obj["ts_ms"] = static_cast<double>(metricsNowMs());

        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const qint64 sent = controlSocket.writeDatagram(
            payload,
            QHostAddress(qtManualControlHost),
            static_cast<quint16>(qtManualControlPort)
        );
        const QString status = QString("%1 -> %2:%3")
                                   .arg(profile)
                                   .arg(qtManualControlHost)
                                   .arg(qtManualControlPort);
        if (sent < 0) {
            qWarning() << "[QT_CTRL] camera profile UDP send failed"
                       << qtManualControlHost << qtManualControlPort
                       << controlSocket.errorString();
            setQtCameraModeStatusText("FAILED " + status);
        } else {
            setQtCameraModeStatusText(status);
        }
    }

    void toggleRecording() {
        const bool next = !g_recordingRequested.load();
        if (next && !recordingStorageAllowed("界面录屏", false)) {
            return;
        }
        if (next) {
            ensureCaptureSessionForRecording();
            g_recordingPaused.store(false);
        }
        g_recordingRequested.store(next);
        if (next) {
            setCurrentRecordingPath("");
            setWindowsRecordingStatus(antiUavNodeRole() == "windows" ? "录制中" : "录制中");
            qInfo() << "[REC] start requested";
        } else {
            g_recordingPaused.store(false);
            setWindowsRecordingStatus("正在保存");
            qInfo() << "[REC] stop requested";
        }
    }

    void toggleRawRecording() {
        const bool next = !g_rawRecordingRequested.load();
        if (next) {
            if (!recordingStorageAllowed("原始视频", false)) return;
            ensureCaptureSessionForRecording();
            startRawVideoRecording(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
            qInfo() << "[REC] raw camera start requested";
        } else {
            stopRawVideoRecording();
            qInfo() << "[REC] raw camera stop requested";
        }
    }

    void toggleAudioRawRecording() {
        const bool next = !g_audioRawRecordingRequested.load();
        if (next) {
            if (!recordingStorageAllowed("原始音频", false)) return;
            ensureCaptureSessionForRecording();
            startAudioRawRecording(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
            qInfo() << "[REC] audio raw start requested";
        } else {
            stopAudioRawRecording(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
            qInfo() << "[REC] audio raw stop requested";
        }
    }

    void toggleFusedRecording() {
        const bool next = !g_fusedRecordingRequested.load();
        if (next) {
            if (!recordingStorageAllowed("RK原始音视频同步采集", true)) return;
            const QString sessionId = ensureCaptureSessionForRecording();
            g_fusedRecordingRequested.store(true);
            g_fusedRecordingActive.store(true);
            appendRecordingTimelineEvent("sync_raw_capture_start", "qt");
            startRawVideoRecording(sessionId);
            startAudioRawRecording(sessionId);
            writeCaptureSessionManifest("recording");
            writeCaptureSessionSummary("recording");
            qInfo() << "[REC] RK raw audio/video sync capture started:" << sessionId;
        } else {
            const QString sessionId = currentCaptureSessionId();
            g_fusedRecordingRequested.store(false);
            appendRecordingTimelineEvent("sync_raw_capture_stop", "qt");
            stopRawVideoRecording();
            stopAudioRawRecording(sessionId);
            g_fusedRecordingActive.store(false);
            writeCaptureSessionManifest("stopped");
            writeCaptureSessionSummary("stopped");
            qInfo() << "[REC] RK raw audio/video sync capture stopped:" << sessionId;
        }
    }

    void toggleDetectionEnabled() {
        const bool next = !g_detectionEnabledRequested.load();
        g_detectionEnabledRequested.store(next);
        g_detectionEnabledReported.store(next);
        qInfo() << "[CTRL] detection enabled requested:" << next;

        QJsonObject obj;
        obj["type"] = "qt_control";
        obj["command"] = "set_detection_enabled";
        obj["detection_enabled"] = next ? 1 : 0;
        sendControlJson(obj, yoloControlPort);
    }

    void toggleDisplayEcoMode() {
        const bool next = !g_displayEcoMode.load();
        g_displayEcoMode.store(next);
        qInfo() << "[UI] light paint mode:" << next;
    }

    void startRawVideoRecording(const QString& recordId) {
        setCurrentRawRecordingPath("");
        setCurrentRawRecordingSessionId(recordId);
        g_rawRecordingRequested.store(true);
    }

    void stopRawVideoRecording() {
        g_rawRecordingRequested.store(false);
        setCurrentRawRecordingSessionId("");
    }

    void startAudioRawRecording(const QString& recordId) {
        setCurrentAudioRawRecordingPath("");
        setCurrentAudioRawRecordingSessionId(recordId);
        g_audioRawRecordingRequested.store(true);
        appendRecordingTimelineEvent("raw_audio_command_start", "qt_audio_control", currentAudioRawRecordingPath());
        sendAudioRawRecordingCommand(true, recordId);
    }

    void stopAudioRawRecording(const QString& recordId) {
        appendRecordingTimelineEvent("raw_audio_command_stop", "qt_audio_control", currentAudioRawRecordingPath());
        g_audioRawRecordingRequested.store(false);
        sendAudioRawRecordingCommand(false, recordId);
        setCurrentAudioRawRecordingSessionId("");
    }

    void sendAudioRawRecordingCommand(bool enabled, const QString& recordId) {
        QJsonObject obj;
        obj["type"] = "audio_control";
        obj["command"] = "set_raw_recording";
        obj["raw_recording"] = enabled ? 1 : 0;
        obj["record_id"] = recordId;
        obj["session_id"] = currentCaptureSessionId();
        const QString learnedAudioHost = latestAudioSenderHost();
        if (!learnedAudioHost.isEmpty() && learnedAudioHost != controlHost) {
            sendControlJsonToHost(obj, learnedAudioHost, audioControlPort);
        }
        sendControlJson(obj, audioControlPort);
    }

    void sendControlJson(QJsonObject obj, int port) {
        sendControlJsonToHost(obj, controlHost, port);
    }

    void sendControlJsonToHost(QJsonObject obj, const QString& host, int port) {
        obj["source"] = "qt_dashboard";
        obj["ts_ms"] = static_cast<double>(metricsNowMs());
        const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
        const qint64 sent = controlSocket.writeDatagram(payload, QHostAddress(host), port);
        if (sent < 0) {
            qWarning() << "[CTRL] UDP send failed" << host << port << controlSocket.errorString();
        } else {
            qInfo() << "[CTRL] UDP sent" << host << port << payload;
        }
    }

protected:
    void closeEvent(QCloseEvent* event) override {
        if (antiUavNodeRole() == "windows") {
            QWidget::closeEvent(event);
            return;
        }
        if (exitToDesktopRequested) {
            event->accept();
            return;
        }
        const QMessageBox::StandardButton choice = QMessageBox::question(
            this,
            "退出 AntiUAV",
            "是否停止系统并返回 App Launcher？",
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
        if (choice == QMessageBox::Yes) {
            event->ignore();
            requestExitToDesktop();
        } else {
            event->ignore();
        }
    }

private:
    bool exitToDesktopRequested = false;
};

static bool dashboardIsRecent(qint64 timestampMs, qint64 nowMs, qint64 maxAgeMs) {
    return timestampMs > 0 && nowMs >= timestampMs && (nowMs - timestampMs) <= maxAgeMs;
}

static QString dashboardAgeText(qint64 timestampMs, qint64 nowMs) {
    if (timestampMs <= 0 || nowMs < timestampMs) {
        return "未接收";
    }
    const double ageSec = static_cast<double>(nowMs - timestampMs) / 1000.0;
    if (ageSec < 0.8) {
        return "刚刚";
    }
    return QString("%1秒前").arg(ageSec, 0, 'f', 1);
}

static QString dashboardAngleText(double value) {
    if (!std::isfinite(value) || std::fabs(value) > 9000.0) {
        return "--";
    }
    return QString("%1°").arg(value, 0, 'f', 1);
}

static QString dashboardPercentText(double value) {
    if (!std::isfinite(value)) {
        value = 0.0;
    }
    const double percentValue = value > 1.5 ? value : value * 100.0;
    const int percent = qBound(0, static_cast<int>(std::round(percentValue)), 100);
    return QString("%1%").arg(percent);
}

enum class DashboardTone {
    Good,
    Warn,
    Bad,
    Idle
};

static QString dashboardPillStyle(DashboardTone tone) {
    QString fg = "#9AA7B5";
    QString bg = "rgba(90, 100, 112, 0.18)";
    QString border = "rgba(138, 148, 160, 0.45)";

    switch (tone) {
    case DashboardTone::Good:
        fg = "#6EE7B7";
        bg = "rgba(16, 185, 129, 0.16)";
        border = "rgba(110, 231, 183, 0.55)";
        break;
    case DashboardTone::Warn:
        fg = "#FBBF24";
        bg = "rgba(245, 158, 11, 0.16)";
        border = "rgba(251, 191, 36, 0.55)";
        break;
    case DashboardTone::Bad:
        fg = "#F87171";
        bg = "rgba(239, 68, 68, 0.18)";
        border = "rgba(248, 113, 113, 0.60)";
        break;
    case DashboardTone::Idle:
        break;
    }

    return QString(
        "QLabel { color: %1; background: %2; border: 1px solid %3; "
        "border-radius: 5px; padding: 6px 10px; font-weight: 800; font-size: 14px; }"
    ).arg(fg, bg, border);
}

static QString dashboardConfidenceText(const QString& label, double value) {
    const QString normalized = label.trimmed().toLower();
    if (normalized == "high") {
        return "高";
    }
    if (normalized == "medium" || normalized == "mid") {
        return "中";
    }
    if (normalized == "low") {
        return "低";
    }
    if (!label.trimmed().isEmpty()) {
        return label;
    }
    return value >= 0.75 ? "高" : (value >= 0.45 ? "中" : "低");
}

static QString dashboardFusionText(bool audioGuided,
                                   bool visualTracking,
                                   bool audioDetected,
                                   const QString& rawMode,
                                   double offsetConfidence) {
    if (audioGuided) {
        return "听觉引导视觉中";
    }
    if (visualTracking && audioDetected) {
        return "音视共同确认";
    }
    if (visualTracking) {
        return "视觉跟踪优先";
    }
    if (audioDetected) {
        return offsetConfidence >= 0.65 ? "听觉准备引导" : "听觉保守搜索";
    }

    const QString mode = rawMode.trimmed().toLower();
    if (mode == "track" || mode == "tracking") {
        return "视觉跟踪中";
    }
    if (mode == "search" || mode == "scan") {
        return "搜索中";
    }
    return "待机监听";
}

class HoldPtzButton : public QPushButton {
public:
    HoldPtzButton(const QString& text,
                  std::function<void()> pressAction,
                  std::function<void()> releaseAction,
                  QWidget* parent = nullptr)
        : QPushButton(text, parent),
          onPress(std::move(pressAction)),
          onRelease(std::move(releaseAction)) {
        setObjectName("PtzDirectionButton");
        setMinimumSize(50, 44);
        setAutoRepeat(false);
        setAttribute(Qt::WA_AcceptTouchEvents, true);
    }

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            active = true;
            if (onPress) onPress();
        }
        QPushButton::mousePressEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override {
        releaseNow();
        QPushButton::mouseReleaseEvent(event);
    }

    void leaveEvent(QEvent* event) override {
        releaseNow();
        QPushButton::leaveEvent(event);
    }

    bool event(QEvent* event) override {
        if (event->type() == QEvent::TouchCancel ||
            event->type() == QEvent::UngrabMouse ||
            event->type() == QEvent::Hide) {
            releaseNow();
        }
        return QPushButton::event(event);
    }

private:
    void releaseNow() {
        if (!active) return;
        active = false;
        if (onRelease) onRelease();
    }

    bool active = false;
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

class PtzTouchPanel : public QFrame {
public:
    explicit PtzTouchPanel(DashboardWindow* controller, QWidget* parent = nullptr)
        : QFrame(parent), dashboard(controller) {
        setObjectName("PtzTouchPanel");
        QVBoxLayout* root = new QVBoxLayout(this);
        root->setContentsMargins(8, 5, 8, 6);
        root->setSpacing(5);

        QHBoxLayout* titleRow = new QHBoxLayout();
        QLabel* title = new QLabel("触摸云台");
        title->setObjectName("SectionTitle");
        ptzState = new QLabel("P --  T --  Z --");
        ptzState->setObjectName("PtzState");
        ptzState->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        titleRow->addWidget(title);
        titleRow->addStretch(1);
        titleRow->addWidget(ptzState);
        root->addLayout(titleRow);

        QHBoxLayout* body = new QHBoxLayout();
        body->setSpacing(8);
        QGridLayout* directionGrid = new QGridLayout();
        directionGrid->setSpacing(4);
        addHold(directionGrid, "↖", 0, 0, -1, 1, 0);
        addHold(directionGrid, "▲", 0, 1, 0, 1, 0);
        addHold(directionGrid, "↗", 0, 2, 1, 1, 0);
        addHold(directionGrid, "◀", 1, 0, -1, 0, 0);
        QPushButton* stop = new QPushButton("■");
        stop->setObjectName("PtzStopButton");
        stop->setMinimumSize(50, 44);
        connect(stop, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestTouchStop("touch_center_stop");
        });
        directionGrid->addWidget(stop, 1, 1);
        addHold(directionGrid, "▶", 1, 2, 1, 0, 0);
        addHold(directionGrid, "↙", 2, 0, -1, -1, 0);
        addHold(directionGrid, "▼", 2, 1, 0, -1, 0);
        addHold(directionGrid, "↘", 2, 2, 1, -1, 0);
        body->addLayout(directionGrid);

        QVBoxLayout* zoomColumn = new QVBoxLayout();
        zoomColumn->setSpacing(5);
        zoomColumn->addWidget(makeHold("＋", 0, 0, 1));
        zoomColumn->addWidget(makeHold("－", 0, 0, -1));
        QPushButton* autofocus = actionButton("自动对焦");
        connect(autofocus, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestAutofocus();
        });
        zoomColumn->addWidget(autofocus);
        body->addLayout(zoomColumn);

        QGridLayout* actionGrid = new QGridLayout();
        actionGrid->setSpacing(5);
        autoTrackButton = actionButton("自动跟踪：开");
        autoTrackButton->setCheckable(true);
        autoTrackButton->setChecked(true);
        centerModeButton = actionButton("点按画面居中");
        centerModeButton->setCheckable(true);
        QPushButton* saveHome = actionButton("保存位置");
        QPushButton* returnInitial = actionButton("返回初始");
        QPushButton* returnSaved = actionButton("返回保存");
        QPushButton* gotoPreset = actionButton("调用预置点");
        emergencyButton = new QPushButton("紧急停止");
        emergencyButton->setObjectName("EmergencyStopButton");
        emergencyButton->setMinimumHeight(38);

        actionGrid->addWidget(autoTrackButton, 0, 0);
        actionGrid->addWidget(centerModeButton, 0, 1);
        actionGrid->addWidget(saveHome, 1, 0);
        actionGrid->addWidget(returnSaved, 1, 1);
        actionGrid->addWidget(returnInitial, 2, 0);
        actionGrid->addWidget(gotoPreset, 2, 1);
        presetSpin = new QSpinBox();
        presetSpin->setRange(1, 300);
        presetSpin->setValue(envIntBounded("PTZ_HOME_PRESET", 1, 1, 300));
        presetSpin->setPrefix("预置点 ");
        presetSpin->setMinimumHeight(32);
        actionGrid->addWidget(presetSpin, 3, 0, 1, 2);
        actionGrid->addWidget(emergencyButton, 4, 0, 1, 2);
        body->addLayout(actionGrid, 1);
        root->addLayout(body);

        QHBoxLayout* speedRow = new QHBoxLayout();
        const int defaultManualSpeed = envIntBounded("ANTI_UAV_QT_MANUAL_SPEED_DEFAULT", 60, 10, 100);
        if (dashboard) dashboard->setTouchSpeed(defaultManualSpeed);
        speedValue = new QLabel(QString::number(defaultManualSpeed));
        speedSlider = new QSlider(Qt::Horizontal);
        speedSlider->setRange(10, 100);
        speedSlider->setValue(defaultManualSpeed);
        speedRow->addWidget(new QLabel("速度"));
        speedRow->addWidget(speedSlider, 1);
        speedRow->addWidget(speedValue);
        root->addLayout(speedRow);

        commandState = new QLabel("按住移动，松手立即停止");
        commandState->setObjectName("PanelHint");
        root->addWidget(commandState);

        connect(speedSlider, &QSlider::valueChanged, this, [this](int value) {
            speedValue->setText(QString::number(value));
            if (dashboard) dashboard->setTouchSpeed(value);
        });
        connect(autoTrackButton, &QPushButton::toggled, this, [this](bool enabled) {
            if (dashboard) dashboard->requestSetAutoTracking(enabled);
        });
        connect(saveHome, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestSaveHome();
        });
        connect(returnInitial, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestReturnInitial();
        });
        connect(returnSaved, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestReturnSavedHome();
        });
        connect(gotoPreset, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestGotoPreset(presetSpin->value());
        });
        connect(emergencyButton, &QPushButton::clicked, this, [this]() {
            autoTrackButton->setChecked(false);
            if (dashboard) dashboard->requestEmergencyStop();
        });

        statusTimer.setInterval(250);
        connect(&statusTimer, &QTimer::timeout, this, [this]() { updateStatus(); });
        statusTimer.start();
        updateStatus();
    }

    void handleVideoClick(double nx, double ny) {
        if (!centerModeButton->isChecked()) return;
        centerModeButton->setChecked(false);
        if (dashboard) dashboard->requestCenterOnPoint(nx, ny);
    }

private:
    QPushButton* actionButton(const QString& text) {
        QPushButton* button = new QPushButton(text);
        button->setObjectName("PtzActionButton");
        button->setMinimumHeight(34);
        return button;
    }

    HoldPtzButton* makeHold(const QString& text, int pan, int tilt, int zoom) {
        return new HoldPtzButton(
            text,
            [this, pan, tilt, zoom]() {
                if (dashboard) dashboard->requestTouchMove(pan, tilt, zoom);
            },
            [this]() {
                if (dashboard) dashboard->requestTouchStop();
            },
            this);
    }

    void addHold(QGridLayout* grid, const QString& text,
                 int row, int column, int pan, int tilt, int zoom) {
        grid->addWidget(makeHold(text, pan, tilt, zoom), row, column);
    }

    void updateStatus() {
        const double panDeg = g_latestCameraAzimuthUnit.load() / 10.0;
        const double tiltDeg = g_latestCameraElevationUnit.load() / 10.0;
        const double zoomRatio = g_latestCameraAbsoluteZoom.load() / 10.0;
        ptzState->setText(QString("P %1°  T %2°  Z %3×")
                              .arg(panDeg, 0, 'f', 1)
                              .arg(tiltDeg, 0, 'f', 1)
                              .arg(zoomRatio, 0, 'f', 1));
        const bool autoEnabled = g_latestAutoTrackingEnabled.load();
        if (autoTrackButton->isChecked() != autoEnabled) {
            autoTrackButton->blockSignals(true);
            autoTrackButton->setChecked(autoEnabled);
            autoTrackButton->blockSignals(false);
        }
        autoTrackButton->setText(autoEnabled ? "自动跟踪：开" : "自动跟踪：关");
        commandState->setText(qtPtzStatusText());
    }

    DashboardWindow* dashboard = nullptr;
    QLabel* ptzState = nullptr;
    QLabel* speedValue = nullptr;
    QLabel* commandState = nullptr;
    QSlider* speedSlider = nullptr;
    QSpinBox* presetSpin = nullptr;
    QPushButton* autoTrackButton = nullptr;
    QPushButton* centerModeButton = nullptr;
    QPushButton* emergencyButton = nullptr;
    QTimer statusTimer;
};

class DashboardSidePanel : public QWidget {
public:
    explicit DashboardSidePanel(DashboardWindow* controller, QWidget* parent = nullptr)
        : QWidget(parent), dashboard(controller) {
        setObjectName("DashboardSidePanel");
        setMinimumWidth(500);
        setMaximumWidth(580);
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

        QVBoxLayout* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(12);

        QFrame* header = new QFrame();
        header->setObjectName("PanelHeader");
        QVBoxLayout* headerLayout = new QVBoxLayout(header);
        headerLayout->setContentsMargins(18, 14, 18, 14);
        headerLayout->setSpacing(6);

        QLabel* title = new QLabel("反无人机融合控制台");
        title->setObjectName("PanelTitle");
        QLabel* subtitle = new QLabel("视觉锁定  |  听觉引导  |  在线校准");
        subtitle->setObjectName("PanelSubtitle");
        clockLabel = new QLabel();
        clockLabel->setObjectName("PanelClock");

        headerLayout->addWidget(title);
        headerLayout->addWidget(subtitle);
        headerLayout->addWidget(clockLabel);
        root->addWidget(header);

        QVBoxLayout* statusBody = nullptr;
        QFrame* statusSection = createSection("系统状态", statusBody);
        QGridLayout* statusGrid = new QGridLayout();
        statusGrid->setContentsMargins(0, 0, 0, 0);
        statusGrid->setHorizontalSpacing(12);
        statusGrid->setVerticalSpacing(10);
        addStatusRow(statusGrid, 0, "视频流", videoPill);
        addStatusRow(statusGrid, 1, "视觉数据", visionPill);
        addStatusRow(statusGrid, 2, "音频数据", audioPill);
        addStatusRow(statusGrid, 3, "融合状态", fusionPill);
        addStatusRow(statusGrid, 4, "检测开关", detectionPill);
        addStatusRow(statusGrid, 5, "显示模式", displayModePill);
        addValueRow(statusGrid, 6, "模型", modelNameValue);
        statusBody->addLayout(statusGrid);
        root->addWidget(statusSection);

        QVBoxLayout* captureBody = nullptr;
        QFrame* captureSection = createSection("数据采集", captureBody);
        QGridLayout* captureGrid = new QGridLayout();
        captureGrid->setContentsMargins(0, 0, 0, 0);
        captureGrid->setHorizontalSpacing(12);
        captureGrid->setVerticalSpacing(10);
        addStatusRow(captureGrid, 0, "RK同步采集", fusionRecordPill);
        addStatusRow(captureGrid, 1, "Qt界面录屏", recordPill);
        addStatusRow(captureGrid, 2, "RK原始视频", rawRecordPill);
        addStatusRow(captureGrid, 3, "RK原始音频", audioRawRecordPill);
        addValueRow(captureGrid, 4, "会话ID", sessionIdValue);
        addValueRow(captureGrid, 5, "数据盘", dataDiskValue);
        addValueRow(captureGrid, 6, "保存路径", recordingPathValue);
        captureBody->addLayout(captureGrid);

        QGridLayout* captureButtons = new QGridLayout();
        captureButtons->setContentsMargins(0, 0, 0, 0);
        captureButtons->setHorizontalSpacing(10);
        captureButtons->setVerticalSpacing(10);
        fusedRecordButton = new QPushButton("同步开始原始音视频采集  F");
        fusedRecordButton->setObjectName("RecordButton");
        recordButton = new QPushButton("Qt界面录屏  X");
        recordButton->setObjectName("RecordButton");
        rawRecordButton = new QPushButton("开始原始视频录像  B");
        rawRecordButton->setObjectName("RecordButton");
        audioRawRecordButton = new QPushButton("开始原始音频录音  M");
        audioRawRecordButton->setObjectName("RecordButton");
        captureButtons->addWidget(fusedRecordButton, 0, 0, 1, 2);
        captureButtons->addWidget(rawRecordButton, 1, 0);
        captureButtons->addWidget(audioRawRecordButton, 1, 1);
        captureButtons->addWidget(recordButton, 2, 0, 1, 2);
        captureBody->addLayout(captureButtons);
        root->addWidget(captureSection);
        captureSection->hide();

        QVBoxLayout* visionBody = nullptr;
        QFrame* visionSection = createSection("视觉跟踪", visionBody);
        QGridLayout* visionGrid = new QGridLayout();
        visionGrid->setContentsMargins(0, 0, 0, 0);
        visionGrid->setHorizontalSpacing(12);
        visionGrid->setVerticalSpacing(9);
        addValueRow(visionGrid, 0, "目标状态", visualLockValue);
        addValueRow(visionGrid, 1, "相机方位", cameraAzimuthValue);
        addValueRow(visionGrid, 2, "视觉方位", visualAzimuthValue);
        addValueRow(visionGrid, 3, "融合阶段", fusionStateValue);
        addValueRow(visionGrid, 4, "控制来源", controlSourceValue);
        addValueRow(visionGrid, 5, "搜索状态", searchStateValue);
        visionBody->addLayout(visionGrid);
        root->addWidget(visionSection);

        QVBoxLayout* audioBody = nullptr;
        QFrame* audioSection = createSection("听觉定位", audioBody);
        QGridLayout* audioGrid = new QGridLayout();
        audioGrid->setContentsMargins(0, 0, 0, 0);
        audioGrid->setHorizontalSpacing(12);
        audioGrid->setVerticalSpacing(9);
        addValueRow(audioGrid, 0, "原始方位", rawDoaValue);
        addValueRow(audioGrid, 1, "映射方位", audioWorldValue);
        addValueRow(audioGrid, 2, "置信/声压", audioScoreValue);
        audioBody->addLayout(audioGrid);
        doaBar = createProgressBar("DOA稳定度");
        audioBody->addWidget(doaBar);
        root->addWidget(audioSection);

        QVBoxLayout* fusionBody = nullptr;
        QFrame* fusionSection = createSection("在线校准", fusionBody);
        QGridLayout* fusionGrid = new QGridLayout();
        fusionGrid->setContentsMargins(0, 0, 0, 0);
        fusionGrid->setHorizontalSpacing(12);
        fusionGrid->setVerticalSpacing(9);
        addValueRow(fusionGrid, 0, "角度偏移", offsetValue);
        addValueRow(fusionGrid, 1, "音视误差", avErrorValue);
        addValueRow(fusionGrid, 2, "样本数量", samplesValue);
        fusionBody->addLayout(fusionGrid);
        offsetBar = createProgressBar("校准置信度");
        fusionBody->addWidget(offsetBar);
        root->addWidget(fusionSection);

        QVBoxLayout* experimentBody = nullptr;
        QFrame* experimentSection = createSection("实验标注", experimentBody);
        QGridLayout* experimentGrid = new QGridLayout();
        experimentGrid->setContentsMargins(0, 0, 0, 0);
        experimentGrid->setHorizontalSpacing(10);
        experimentGrid->setVerticalSpacing(8);

        experimentIdEdit = createLineEdit("run");
        trialIdEdit = createLineEdit("trial");
        sceneCombo = createCombo();
        sceneCombo->addItem("未选择", "");
        sceneCombo->addItem("室内近距离", "indoor_near");
        sceneCombo->addItem("室外空旷", "outdoor_open");
        sceneCombo->addItem("室外风噪", "outdoor_wind");
        sceneCombo->addItem("人声/车辆干扰", "complex_noise");
        distanceCombo = createCombo();
        distanceCombo->addItem("未选择", "");
        distanceCombo->addItem("5 m", "5");
        distanceCombo->addItem("20 m", "20");
        distanceCombo->addItem("30 m", "30");
        distanceCombo->addItem("50 m", "50");
        noiseCombo = createCombo();
        noiseCombo->addItem("未选择", "");
        noiseCombo->addItem("低噪声", "low_noise");
        noiseCombo->addItem("中噪声", "medium_noise");
        noiseCombo->addItem("风噪", "wind_noise");
        noiseCombo->addItem("复杂噪声", "complex_noise");
        targetCombo = createCombo();
        targetCombo->addItem("未标注", "");
        targetCombo->addItem("有无人机", "1");
        targetCombo->addItem("无无人机", "0");
        quickPresetCombo = createCombo();
        quickPresetCombo->addItem("手动选择", "");
        quickPresetCombo->addItem("室内近距离 / 5m / 低噪声 / 有目标", "indoor_near|5|low_noise|1");
        quickPresetCombo->addItem("室外空旷 / 20m / 中噪声 / 有目标", "outdoor_open|20|medium_noise|1");
        quickPresetCombo->addItem("室外风噪 / 30m / 风噪 / 有目标", "outdoor_wind|30|wind_noise|1");
        quickPresetCombo->addItem("人声车辆干扰 / 20m / 复杂噪声 / 有目标", "complex_noise|20|complex_noise|1");
        quickPresetCombo->addItem("无无人机背景 / 20m / 当前噪声 / 无目标", "outdoor_open|20|medium_noise|0");
        addExperimentRow(experimentGrid, 0, "预设", quickPresetCombo);
        addExperimentRow(experimentGrid, 1, "实验ID", experimentIdEdit);
        addExperimentRow(experimentGrid, 2, "场景", sceneCombo);
        addExperimentRow(experimentGrid, 3, "距离", distanceCombo);
        addExperimentRow(experimentGrid, 4, "噪声", noiseCombo);
        addExperimentRow(experimentGrid, 5, "目标", targetCombo);
        addExperimentRow(experimentGrid, 6, "试次", trialIdEdit);
        experimentBody->addLayout(experimentGrid);

        QHBoxLayout* experimentButtons = new QHBoxLayout();
        experimentButtons->setContentsMargins(0, 0, 0, 0);
        experimentButtons->setSpacing(10);
        applyPresetButton = new QPushButton("套用预设");
        applyPresetButton->setObjectName("ApplyExperimentButton");
        newTrialButton = new QPushButton("新试次");
        newTrialButton->setObjectName("ApplyExperimentButton");
        applyExperimentButton = new QPushButton("应用实验标注");
        applyExperimentButton->setObjectName("ApplyExperimentButton");
        experimentButtons->addWidget(applyPresetButton);
        experimentButtons->addWidget(newTrialButton);
        experimentBody->addLayout(experimentButtons);
        experimentBody->addWidget(applyExperimentButton);
        experimentHint = new QLabel("方法/模块由同一轮日志自动统计");
        experimentHint->setObjectName("PanelHint");
        experimentBody->addWidget(experimentHint);
        root->addWidget(experimentSection);
        loadExperimentSelectionToUi(currentExperimentSelection());

        QVBoxLayout* controlBody = nullptr;
        QFrame* controlSection = createSection("控制", controlBody);
        detectionToggleButton = new QPushButton("关闭无人机检测  T");
        detectionToggleButton->setObjectName("ControlButton");
        ecoModeButton = new QPushButton("轻量绘制  L");
        ecoModeButton->setObjectName("ControlButton");
        stopButton = new QPushButton("云台停止  Space");
        stopButton->setObjectName("StopButton");
        dayModeButton = new QPushButton("白天彩色");
        dayModeButton->setObjectName("ControlButton");
        nightModeButton = new QPushButton("夜晚红外");
        nightModeButton->setObjectName("ControlButton");
        autoDayNightButton = new QPushButton("自动日夜");
        autoDayNightButton->setObjectName("ControlButton");
        profileNormalButton = new QPushButton("普通白天");
        profileNormalButton->setObjectName("ControlButton");
        profileCloudyButton = new QPushButton("阴天无人机");
        profileCloudyButton->setObjectName("ControlButton");
        profileFastButton = new QPushButton("高速跟踪");
        profileFastButton->setObjectName("ControlButton");
        profileLowLightButton = new QPushButton("低照度彩色");
        profileLowLightButton->setObjectName("ControlButton");
        profileNightIrButton = new QPushButton("夜晚IR测试");
        profileNightIrButton->setObjectName("ControlButton");
        profileAutoButton = new QPushButton("自动图像模式");
        profileAutoButton->setObjectName("ControlButton");

        QGridLayout* controlStatusGrid = new QGridLayout();
        controlStatusGrid->setContentsMargins(0, 0, 0, 0);
        controlStatusGrid->setHorizontalSpacing(12);
        controlStatusGrid->setVerticalSpacing(8);
        addValueRow(controlStatusGrid, 0, "Manual UDP", manualUdpValue);
        addValueRow(controlStatusGrid, 1, "Last PTZ cmd", lastPtzCmdValue);
        addValueRow(controlStatusGrid, 2, "Last camera mode", lastCameraModeValue);
        addValueRow(controlStatusGrid, 3, "Camera profile", cameraProfileValue);
        addValueRow(controlStatusGrid, 4, "画质亮度/对比", imageBrightnessValue);
        addValueRow(controlStatusGrid, 5, "饱和/模糊", imageColorBlurValue);
        addValueRow(controlStatusGrid, 6, "灰度/告警", imageWarningValue);
        addValueRow(controlStatusGrid, 7, "Profile reason", profileReasonValue);
        controlBody->addLayout(controlStatusGrid);

        QGridLayout* controlButtons = new QGridLayout();
        controlButtons->setContentsMargins(0, 0, 0, 0);
        controlButtons->setHorizontalSpacing(10);
        controlButtons->setVerticalSpacing(10);
        controlButtons->addWidget(detectionToggleButton, 0, 0);
        controlButtons->addWidget(ecoModeButton, 0, 1);
        controlButtons->addWidget(stopButton, 1, 0, 1, 2);
        controlButtons->addWidget(dayModeButton, 2, 0);
        controlButtons->addWidget(nightModeButton, 2, 1);
        controlButtons->addWidget(autoDayNightButton, 3, 0, 1, 2);
        controlButtons->addWidget(profileNormalButton, 4, 0);
        controlButtons->addWidget(profileCloudyButton, 4, 1);
        controlButtons->addWidget(profileFastButton, 5, 0);
        controlButtons->addWidget(profileLowLightButton, 5, 1);
        controlButtons->addWidget(profileNightIrButton, 6, 0);
        controlButtons->addWidget(profileAutoButton, 6, 1);
        controlBody->addLayout(controlButtons);

        keyHint = new QLabel("W/A/S/D 或方向键控制云台，+/- 变焦，Space/Esc 停止，Qt 仅发 UDP 5011");
        keyHint->setObjectName("PanelHint");
        controlBody->addWidget(keyHint);
        root->addWidget(controlSection);
        root->addStretch(1);

        connect(recordButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleRecording();
            }
        });
        connect(rawRecordButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleRawRecording();
            }
        });
        connect(audioRawRecordButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleAudioRawRecording();
            }
        });
        connect(fusedRecordButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleFusedRecording();
            }
        });
        connect(detectionToggleButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleDetectionEnabled();
            }
        });
        connect(ecoModeButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestToggleDisplayEcoMode();
            }
        });
        connect(stopButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestEmergencyStop();
            }
        });
        connect(dayModeButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestCameraDayMode();
            }
        });
        connect(nightModeButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestCameraNightMode();
            }
        });
        connect(autoDayNightButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) {
                dashboard->requestCameraAutoDayNight();
            }
        });
        connect(profileNormalButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("DAY_NORMAL");
        });
        connect(profileCloudyButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("DAY_CLOUDY_UAV");
        });
        connect(profileFastButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("FAST_TRACKING");
        });
        connect(profileLowLightButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("LOW_LIGHT_KEEP_RGB");
        });
        connect(profileNightIrButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("NIGHT_IR_TEST");
        });
        connect(profileAutoButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestCameraProfile("AUTO_PROFILE");
        });
        connect(applyPresetButton, &QPushButton::clicked, this, [this]() {
            applyQuickPreset();
        });
        connect(newTrialButton, &QPushButton::clicked, this, [this]() {
            createNewTrial();
        });
        connect(applyExperimentButton, &QPushButton::clicked, this, [this]() {
            applyExperimentSelectionFromUi();
        });

        timer.setInterval(250);
        connect(&timer, &QTimer::timeout, this, [this]() { updatePanel(); });
        timer.start();
        updatePanel();
    }

private:
    DashboardWindow* dashboard = nullptr;
    QTimer timer;

    QLabel* clockLabel = nullptr;
    QLabel* videoPill = nullptr;
    QLabel* visionPill = nullptr;
    QLabel* audioPill = nullptr;
    QLabel* fusionPill = nullptr;
    QLabel* detectionPill = nullptr;
    QLabel* displayModePill = nullptr;
    QLabel* modelNameValue = nullptr;
    QLabel* fusionRecordPill = nullptr;
    QLabel* recordPill = nullptr;
    QLabel* rawRecordPill = nullptr;
    QLabel* audioRawRecordPill = nullptr;
    QLabel* sessionIdValue = nullptr;
    QLabel* dataDiskValue = nullptr;
    QLabel* recordingPathValue = nullptr;

    QLabel* visualLockValue = nullptr;
    QLabel* cameraAzimuthValue = nullptr;
    QLabel* visualAzimuthValue = nullptr;
    QLabel* fusionStateValue = nullptr;
    QLabel* controlSourceValue = nullptr;
    QLabel* searchStateValue = nullptr;
    QLabel* rawDoaValue = nullptr;
    QLabel* audioWorldValue = nullptr;
    QLabel* audioScoreValue = nullptr;
    QLabel* offsetValue = nullptr;
    QLabel* avErrorValue = nullptr;
    QLabel* samplesValue = nullptr;
    QLabel* keyHint = nullptr;
    QLabel* experimentHint = nullptr;
    QLabel* manualUdpValue = nullptr;
    QLabel* lastPtzCmdValue = nullptr;
    QLabel* lastCameraModeValue = nullptr;
    QLabel* cameraProfileValue = nullptr;
    QLabel* imageBrightnessValue = nullptr;
    QLabel* imageColorBlurValue = nullptr;
    QLabel* imageWarningValue = nullptr;
    QLabel* profileReasonValue = nullptr;

    QProgressBar* doaBar = nullptr;
    QProgressBar* offsetBar = nullptr;
    QPushButton* recordButton = nullptr;
    QPushButton* rawRecordButton = nullptr;
    QPushButton* audioRawRecordButton = nullptr;
    QPushButton* fusedRecordButton = nullptr;
    QPushButton* detectionToggleButton = nullptr;
    QPushButton* ecoModeButton = nullptr;
    QPushButton* stopButton = nullptr;
    QPushButton* dayModeButton = nullptr;
    QPushButton* nightModeButton = nullptr;
    QPushButton* autoDayNightButton = nullptr;
    QPushButton* profileNormalButton = nullptr;
    QPushButton* profileCloudyButton = nullptr;
    QPushButton* profileFastButton = nullptr;
    QPushButton* profileLowLightButton = nullptr;
    QPushButton* profileNightIrButton = nullptr;
    QPushButton* profileAutoButton = nullptr;
    QPushButton* applyExperimentButton = nullptr;
    QPushButton* applyPresetButton = nullptr;
    QPushButton* newTrialButton = nullptr;
    QLineEdit* experimentIdEdit = nullptr;
    QLineEdit* trialIdEdit = nullptr;
    QComboBox* quickPresetCombo = nullptr;
    QComboBox* sceneCombo = nullptr;
    QComboBox* distanceCombo = nullptr;
    QComboBox* noiseCombo = nullptr;
    QComboBox* targetCombo = nullptr;

    QFrame* createSection(const QString& title, QVBoxLayout*& bodyLayout) {
        QFrame* frame = new QFrame();
        frame->setObjectName("PanelSection");
        bodyLayout = new QVBoxLayout(frame);
        bodyLayout->setContentsMargins(16, 13, 16, 15);
        bodyLayout->setSpacing(10);

        QLabel* titleLabel = new QLabel(title);
        titleLabel->setObjectName("SectionTitle");
        bodyLayout->addWidget(titleLabel);
        return frame;
    }

    QLabel* createNameLabel(const QString& text) {
        QLabel* label = new QLabel(text);
        label->setObjectName("MetricName");
        return label;
    }

    QLabel* createValueLabel() {
        QLabel* label = new QLabel("--");
        label->setObjectName("MetricValue");
        label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        label->setMinimumWidth(170);
        return label;
    }

    QLabel* createPillLabel() {
        QLabel* label = new QLabel("--");
        label->setAlignment(Qt::AlignCenter);
        label->setMinimumWidth(150);
        label->setStyleSheet(dashboardPillStyle(DashboardTone::Idle));
        return label;
    }

    QProgressBar* createProgressBar(const QString& format) {
        QProgressBar* bar = new QProgressBar();
        bar->setRange(0, 100);
        bar->setValue(0);
        bar->setTextVisible(true);
        bar->setFormat(format + "  %p%");
        bar->setFixedHeight(24);
        return bar;
    }

    QComboBox* createCombo() {
        QComboBox* combo = new QComboBox();
        combo->setMinimumHeight(30);
        combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        return combo;
    }

    QLineEdit* createLineEdit(const QString& placeholder) {
        QLineEdit* edit = new QLineEdit();
        edit->setPlaceholderText(placeholder);
        edit->setMinimumHeight(30);
        edit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        return edit;
    }

    void addStatusRow(QGridLayout* grid, int row, const QString& name, QLabel*& pill) {
        grid->addWidget(createNameLabel(name), row, 0);
        pill = createPillLabel();
        grid->addWidget(pill, row, 1);
    }

    void addValueRow(QGridLayout* grid, int row, const QString& name, QLabel*& value) {
        grid->addWidget(createNameLabel(name), row, 0);
        value = createValueLabel();
        grid->addWidget(value, row, 1);
    }

    void addExperimentRow(QGridLayout* grid, int row, const QString& name, QWidget* editor) {
        grid->addWidget(createNameLabel(name), row, 0);
        grid->addWidget(editor, row, 1);
    }

    static void setComboByData(QComboBox* combo, const QString& value) {
        if (!combo) return;
        const int idx = combo->findData(value);
        if (idx >= 0) {
            combo->setCurrentIndex(idx);
        }
    }

    static QString comboData(QComboBox* combo) {
        return combo ? combo->currentData().toString() : "";
    }

    void loadExperimentSelectionToUi(const ExperimentSelection& selection) {
        if (experimentIdEdit) experimentIdEdit->setText(selection.experimentId);
        if (trialIdEdit) trialIdEdit->setText(selection.trialId);
        setComboByData(sceneCombo, selection.sceneType);
        setComboByData(distanceCombo, selection.distanceM);
        setComboByData(noiseCombo, selection.noiseType);
        setComboByData(targetCombo, selection.targetPresent);
    }

    void applyQuickPreset() {
        if (!quickPresetCombo) return;
        const QString data = quickPresetCombo->currentData().toString();
        if (data.isEmpty()) return;
        const QStringList parts = data.split('|');
        if (parts.size() >= 4) {
            setComboByData(sceneCombo, parts.at(0));
            setComboByData(distanceCombo, parts.at(1));
            setComboByData(noiseCombo, parts.at(2));
            setComboByData(targetCombo, parts.at(3));
        }
        if (experimentIdEdit && experimentIdEdit->text().trimmed().isEmpty()) {
            experimentIdEdit->setText("run_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
        }
        if (trialIdEdit && trialIdEdit->text().trimmed().isEmpty()) {
            trialIdEdit->setText("trial_001");
        }
        applyExperimentSelectionFromUi();
    }

    void createNewTrial() {
        if (experimentIdEdit && experimentIdEdit->text().trimmed().isEmpty()) {
            experimentIdEdit->setText("run_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
        }
        if (trialIdEdit) {
            const QString current = trialIdEdit->text().trimmed();
            bool ok = false;
            int n = current.mid(current.lastIndexOf('_') + 1).toInt(&ok);
            if (!ok) n = 0;
            trialIdEdit->setText(QString("trial_%1").arg(n + 1, 3, 10, QLatin1Char('0')));
        }
        applyExperimentSelectionFromUi();
    }

    void applyExperimentSelectionFromUi() {
        ExperimentSelection selection = currentExperimentSelection();
        selection.experimentId = experimentIdEdit ? experimentIdEdit->text().trimmed() : "";
        selection.sceneType = comboData(sceneCombo);
        selection.distanceM = comboData(distanceCombo);
        selection.noiseType = comboData(noiseCombo);
        selection.targetPresent = comboData(targetCombo);
        selection.methodLabel = "all_methods_auto";
        selection.moduleLabel = "auto_multimodal";
        selection.trialId = trialIdEdit ? trialIdEdit->text().trimmed() : "";
        setCurrentExperimentSelection(selection);
        MetricsLogger::instance().updateExperimentSelection(selection);
        if (experimentHint) {
            experimentHint->setText("已应用：" + QDateTime::currentDateTime().toString("HH:mm:ss"));
        }
    }

    void setPill(QLabel* label, const QString& text, DashboardTone tone) {
        if (!label) {
            return;
        }
        label->setText(text);
        label->setStyleSheet(dashboardPillStyle(tone));
    }

    void setProgress(QProgressBar* bar, double value) {
        if (!bar) {
            return;
        }
        if (!std::isfinite(value)) {
            value = 0.0;
        }
        bar->setValue(qBound(0, static_cast<int>(std::round(value * 100.0)), 100));
    }

    void updatePanel() {
        const qint64 now = metricsNowMs();
        const bool videoLive = g_hasVideoSignal.load() &&
                               dashboardIsRecent(g_lastVideoFrameMs.load(), now, 2200);
        const bool visionLive = dashboardIsRecent(g_lastVisionRecvMs.load(), now, 2600);
        const bool audioLive = dashboardIsRecent(g_lastAudioRecvMs.load(), now, 2600) ||
                               g_hasAudioSignal.load();
        const bool recording = g_recordingActive.load();
        const bool recordingPending = g_recordingRequested.load() && !recording;
        const bool rawRecording = g_rawRecordingActive.load();
        const bool rawRecordingPending = g_rawRecordingRequested.load() && !rawRecording;
        const bool detectionRequested = g_detectionEnabledRequested.load();
        const bool detectionReported = g_detectionEnabledReported.load();
        const bool audioRawRecording = g_audioRawRecordingActive.load();
        const bool audioRawRecordingPending = g_audioRawRecordingRequested.load() && !audioRawRecording;
        const bool fusedRecording = g_fusedRecordingActive.load();
        const bool fusedPending = g_fusedRecordingRequested.load() && (!rawRecording || !audioRawRecording);
        const bool ecoMode = g_displayEcoMode.load();

        clockLabel->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd  HH:mm:ss"));

        setPill(videoPill,
                videoLive ? "视频正常" : "等待视频",
                videoLive ? DashboardTone::Good : DashboardTone::Bad);
        setPill(visionPill,
                visionLive ? ("接收中 " + dashboardAgeText(g_lastVisionRecvMs.load(), now)) : "等待视觉",
                visionLive ? DashboardTone::Good : DashboardTone::Idle);
        setPill(audioPill,
                audioLive ? ("接收中 " + dashboardAgeText(g_lastAudioRecvMs.load(), now)) : "监听中",
                audioLive ? DashboardTone::Good : DashboardTone::Idle);
        const QString fusionText = dashboardFusionText(g_latestAudioGuided.load(),
                                                       g_latestVisionTracking.load(),
                                                       g_latestAudioDetected.load(),
                                                       dashboardText(g_latestFusionMode),
                                                       g_latestOffsetConfidence.load());
        setPill(fusionPill,
                fusionText,
                g_latestAudioGuided.load() ? DashboardTone::Warn : DashboardTone::Idle);
        setPill(detectionPill,
                detectionRequested ? (detectionReported ? "已开启" : "等待回包") : "已关闭",
                detectionRequested ? (detectionReported ? DashboardTone::Good : DashboardTone::Warn) : DashboardTone::Idle);
        setPill(displayModePill,
                ecoMode ? "轻绘制 30fps" : "标准 30fps",
                ecoMode ? DashboardTone::Warn : DashboardTone::Good);
        modelNameValue->setText(antiUavYoloModelName());
        setPill(fusionRecordPill,
                fusedRecording ? (fusedPending ? "同步中" : "会话录制中") : "未开启",
                fusedRecording ? (fusedPending ? DashboardTone::Warn : DashboardTone::Bad) : DashboardTone::Idle);
        setPill(recordPill,
                recording ? "录像中" : (recordingPending ? "准备中" : "未录像"),
                recording ? DashboardTone::Bad : (recordingPending ? DashboardTone::Warn : DashboardTone::Idle));
        setPill(rawRecordPill,
                rawRecording ? "录制中" : (rawRecordingPending ? "准备中" : "未录制"),
                rawRecording ? DashboardTone::Bad : (rawRecordingPending ? DashboardTone::Warn : DashboardTone::Idle));
        setPill(audioRawRecordPill,
                audioRawRecording ? "录音中" : (audioRawRecordingPending ? "等待音频端" : "未录音"),
                audioRawRecording ? DashboardTone::Bad : (audioRawRecordingPending ? DashboardTone::Warn : DashboardTone::Idle));
        const QString sessionId = currentCaptureSessionId();
        sessionIdValue->setText(sessionId.isEmpty() ? "--" : sessionId);
        dataDiskValue->setText(dataDiskStatusText());
        QString recordPath = currentRecordingPath();
        if (recordPath.isEmpty()) recordPath = currentRawRecordingPath();
        if (recordPath.isEmpty()) recordPath = currentAudioRawRecordingPath();
        if (recordPath.isEmpty()) recordPath = lastRecordingPath();
        recordingPathValue->setText(recordPath.isEmpty() ? windowsRecordingStatus() : recordPath);

        const bool visualDetected = g_latestVisionDetected.load();
        const bool visualTracking = g_latestVisionTracking.load();
        visualLockValue->setText(QString("%1 / 置信 %2")
                                     .arg(visualTracking ? "跟踪中" : (visualDetected ? "已发现" : "等待目标"))
                                     .arg(g_latestVisionConf.load(), 0, 'f', 2));
        cameraAzimuthValue->setText(dashboardAngleText(g_latestCameraAzimuthDeg.load()));
        visualAzimuthValue->setText(dashboardAngleText(g_latestVisualAzimuthDeg.load()));
        fusionStateValue->setText(dashboardText(g_latestFusionState));
        controlSourceValue->setText(dashboardText(g_latestControlSource));
        searchStateValue->setText(QString("%1 / Δ%2°")
                                      .arg(dashboardText(g_latestSearchState))
                                      .arg(g_latestVerticalSweepDeltaDeg.load(), 0, 'f', 1));

        rawDoaValue->setText(dashboardAngleText(g_latestAudioAngle.load()));
        audioWorldValue->setText(dashboardAngleText(g_latestAudioWorldAzimuthDeg.load()));
        audioScoreValue->setText(QString("%1 / %2 dB")
                                     .arg(dashboardPercentText(g_latestAudioScoreEma.load()))
                                     .arg(g_latestAudioRmsDbfs.load(), 0, 'f', 1));
        setProgress(doaBar, g_latestAudioDoaStability.load());

        offsetValue->setText(QString("%1  置信:%2")
                                 .arg(dashboardAngleText(g_latestMicOffsetDeg.load()))
                                 .arg(dashboardConfidenceText(dashboardText(g_latestOffsetConfidenceLabel),
                                                              g_latestOffsetConfidence.load())));
        avErrorValue->setText(dashboardAngleText(g_latestAudioVisualErrDeg.load()));
        samplesValue->setText(QString::number(g_latestCalibrationSamples.load()));
        setProgress(offsetBar, g_latestOffsetConfidence.load());

        manualUdpValue->setText(dashboard ? dashboard->manualPtzEndpointText() : "127.0.0.1:5011");
        lastPtzCmdValue->setText(qtPtzStatusText());
        lastCameraModeValue->setText(qtCameraModeStatusText());
        cameraProfileValue->setText(dashboardText(g_latestCameraProfile));
        imageBrightnessValue->setText(QString("Y %1 / C %2")
                                          .arg(g_latestBrightnessMean.load(), 0, 'f', 1)
                                          .arg(g_latestContrastStd.load(), 0, 'f', 1));
        imageColorBlurValue->setText(QString("S %1 / B %2")
                                         .arg(g_latestSaturationMean.load(), 0, 'f', 1)
                                         .arg(g_latestBlurLaplacianVar.load(), 0, 'f', 0));
        imageWarningValue->setText(QString("%1 / %2")
                                       .arg(g_latestGrayscaleLike.load() ? "近灰度" : "彩色")
                                       .arg(dashboardText(g_latestImageQualityWarning)));
        profileReasonValue->setText(dashboardText(g_latestProfileSwitchReason));

        recordButton->setText(g_recordingRequested.load() ? "停止Qt界面录屏  X" : "Qt界面录屏  X");
        rawRecordButton->setText(g_rawRecordingRequested.load() ? "停止原始视频录像  B" : "开始原始视频录像  B");
        audioRawRecordButton->setText(g_audioRawRecordingRequested.load() ? "停止原始音频录音  M" : "开始原始音频录音  M");
        fusedRecordButton->setText(g_fusedRecordingRequested.load() ? "停止并保存原始音视频  F" : "同步开始原始音视频采集  F");
        detectionToggleButton->setText(g_detectionEnabledRequested.load() ? "关闭无人机检测  T" : "开启无人机检测  T");
        ecoModeButton->setText(g_displayEcoMode.load() ? "标准绘制  L" : "轻量绘制  L");
    }
};

class PersistentCaptureBar : public QFrame {
public:
    explicit PersistentCaptureBar(DashboardWindow* controller, QWidget* parent = nullptr)
        : QFrame(parent), dashboard(controller) {
        setObjectName("PersistentCaptureBar");
        QVBoxLayout* root = new QVBoxLayout(this);
        root->setContentsMargins(6, 4, 6, 5);
        root->setSpacing(4);

        QHBoxLayout* titleRow = new QHBoxLayout();
        QLabel* title = new QLabel("录像");
        title->setObjectName("SectionTitle");
        stateLabel = new QLabel("未录制");
        stateLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        stateLabel->setStyleSheet(dashboardPillStyle(DashboardTone::Idle));
        emergencyButton = new QPushButton("紧急停止");
        emergencyButton->setObjectName("EmergencyStopButton");
        emergencyButton->setMinimumHeight(32);
        emergencyButton->setMaximumWidth(130);
        titleRow->addWidget(title);
        titleRow->addWidget(stateLabel, 1);
        titleRow->addWidget(emergencyButton);
        root->addLayout(titleRow);

        QHBoxLayout* buttons = new QHBoxLayout();
        buttons->setSpacing(5);
        fusedButton = makeButton("RK同步采集");
        dashboardButton = makeButton("Windows本地录像");
        videoButton = makeButton("暂停采集");
        audioButton = makeButton("停止并保存");
        buttons->addWidget(fusedButton, 1);
        buttons->addWidget(dashboardButton, 1);
        buttons->addWidget(videoButton, 1);
        buttons->addWidget(audioButton, 1);
        root->addLayout(buttons);

        connect(fusedButton, &QPushButton::clicked, this, [this]() {
            if (!dashboard) return;
            if (antiUavNodeRole() == "windows") dashboard->requestToggleWindowsRecording();
            else dashboard->requestToggleFusedRecording();
        });
        connect(dashboardButton, &QPushButton::clicked, this, [this]() {
            if (!dashboard) return;
            if (antiUavNodeRole() == "windows") {
                if (g_recordingPaused.load()) dashboard->requestResumeWindowsRecording();
                else dashboard->requestPauseWindowsRecording();
            } else {
                dashboard->requestToggleWindowsRecording();
            }
        });
        connect(videoButton, &QPushButton::clicked, this, [this]() {
            if (!dashboard) return;
            if (antiUavNodeRole() == "windows") dashboard->requestStopAllRecordings();
            else dashboard->requestToggleRkCapturePause();
        });
        connect(audioButton, &QPushButton::clicked, this, [this]() {
            if (!dashboard) return;
            if (antiUavNodeRole() == "windows") dashboard->requestOpenRecordingDirectory();
            else dashboard->requestStopAllRecordings();
        });
        connect(emergencyButton, &QPushButton::clicked, this, [this]() {
            if (dashboard) dashboard->requestEmergencyStop();
        });

        timer.setInterval(250);
        connect(&timer, &QTimer::timeout, this, [this]() { updateState(); });
        timer.start();
        updateState();
    }

private:
    QPushButton* makeButton(const QString& text) {
        QPushButton* button = new QPushButton(text);
        button->setObjectName("PersistentRecordButton");
        button->setMinimumHeight(28);
        return button;
    }

    void updateState() {
        const bool any = g_fusedRecordingRequested.load() ||
                         g_recordingRequested.load() ||
                         g_windowsRemoteRecordingRequested.load() ||
                         g_rawRecordingRequested.load() ||
                         g_audioRawRecordingRequested.load();
        stateLabel->setText(any ? (g_recordingPaused.load() ? "暂停" : "录制中") : "未录制");
        stateLabel->setStyleSheet(
            dashboardPillStyle(any ? DashboardTone::Bad : DashboardTone::Idle));
        if (antiUavNodeRole() == "windows") {
            fusedButton->setText(g_recordingRequested.load() ? "停止Windows录像" : "开始Windows录像");
            dashboardButton->setText(g_recordingPaused.load() ? "恢复" : "暂停");
            videoButton->setText("停止保存");
            audioButton->setText("打开目录");
        } else {
            fusedButton->setText(g_fusedRecordingRequested.load()
                                     ? "停止并保存RK同步采集"
                                     : "同步开始原始音视频采集");
            dashboardButton->setText(g_windowsRemoteRecordingRequested.load()
                                         ? "停止Windows本地录像"
                                         : "开始Windows本地录像");
            videoButton->setText(g_fusedRecordingRequested.load()
                                     ? (g_recordingPaused.load() ? "继续采集" : "暂停采集")
                                     : "暂停采集");
            audioButton->setText("停止并保存");
        }
    }

    DashboardWindow* dashboard = nullptr;
    QLabel* stateLabel = nullptr;
    QPushButton* fusedButton = nullptr;
    QPushButton* dashboardButton = nullptr;
    QPushButton* videoButton = nullptr;
    QPushButton* audioButton = nullptr;
    QPushButton* emergencyButton = nullptr;
    QTimer timer;
};

class TopStatusStrip : public QFrame {
public:
    explicit TopStatusStrip(QWidget* parent = nullptr) : QFrame(parent) {
        setObjectName("TopStatusStrip");
        QHBoxLayout* root = new QHBoxLayout(this);
        root->setContentsMargins(3, 2, 3, 2);
        root->setSpacing(3);
        for (const QString& name : {"网络", "球机", "NPU", "音频", "RID"}) {
            QLabel* label = new QLabel(name);
            label->setAlignment(Qt::AlignCenter);
            label->setMinimumHeight(24);
            label->setStyleSheet(dashboardPillStyle(DashboardTone::Idle));
            root->addWidget(label, 1);
            labels.append(label);
        }
        timer.setInterval(500);
        connect(&timer, &QTimer::timeout, this, [this]() { updateState(); });
        timer.start();
        updateState();
    }

private:
    void setStatus(int index, const QString& text, DashboardTone tone) {
        labels[index]->setText(text);
        labels[index]->setStyleSheet(dashboardPillStyle(tone));
    }

    void updateState() {
        const qint64 now = metricsNowMs();
        const bool visionLive = dashboardIsRecent(g_lastVisionRecvMs.load(), now, 3000);
        const bool videoLive = g_hasVideoSignal.load() &&
                               dashboardIsRecent(g_lastVideoFrameMs.load(), now, 3000);
        const bool npuLive = g_latestVisionFps.load() > 1.0;
        const bool audioLive = dashboardIsRecent(g_lastAudioRecvMs.load(), now, 3000);
        const bool ridConnected = g_ridConnected.load() &&
                                  dashboardIsRecent(g_lastRidRecvMs.load(), now, 5000);
        const bool ridDataLive = dashboardIsRecent(g_lastRidDataMs.load(), now, 5000);
        setStatus(0, visionLive ? "网络 OK" : "网络 --",
                  visionLive ? DashboardTone::Good : DashboardTone::Idle);
        setStatus(1, videoLive ? "球机 OK" : "球机 --",
                  videoLive ? DashboardTone::Good : DashboardTone::Idle);
        setStatus(2, npuLive ? "NPU OK" : "NPU --",
                  npuLive ? DashboardTone::Good : DashboardTone::Idle);
        setStatus(3, audioLive ? "音频 OK" : "音频 --",
                  audioLive ? DashboardTone::Good : DashboardTone::Idle);
        setStatus(4,
                  ridDataLive ? "RID OK" : (ridConnected ? "RID 已连" : "RID --"),
                  ridDataLive ? DashboardTone::Good
                              : (ridConnected ? DashboardTone::Warn : DashboardTone::Idle));
    }

    QList<QLabel*> labels;
    QTimer timer;
};

class OverviewCardsPage : public QWidget {
public:
    explicit OverviewCardsPage(std::function<void(int)> openPage,
                               QWidget* parent = nullptr)
        : QWidget(parent), openPage(std::move(openPage)) {
        QGridLayout* grid = new QGridLayout(this);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(5);
        grid->setVerticalSpacing(5);
        const QStringList names = {
            "摄像头", "YOLO 检测", "自动跟踪", "PTZ",
            "ReSpeaker", "YAMNet", "RID900", "Windows 同步",
            "录像", "系统资源"
        };
        const QList<int> pages = {1, 1, 2, 2, 3, 3, 4, 6, 5, 6};
        for (int i = 0; i < names.size(); ++i) {
            QPushButton* card = new QPushButton(names.at(i));
            card->setObjectName("OverviewCard");
            card->setMinimumHeight(47);
            card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            connect(card, &QPushButton::clicked, this, [this, pages, i]() {
                if (this->openPage) this->openPage(pages.at(i));
            });
            grid->addWidget(card, i / 2, i % 2);
            cards.append(card);
        }
        timer.setInterval(500);
        connect(&timer, &QTimer::timeout, this, [this]() { updateCards(); });
        timer.start();
        updateCards();
    }

private:
    void setCard(int index, const QString& metric, DashboardTone tone) {
        const QString name = cards.at(index)->property("moduleName").toString();
        const QString base = name.isEmpty() ? cards.at(index)->text().section('\n', 0, 0) : name;
        cards.at(index)->setProperty("moduleName", base);
        cards.at(index)->setText(base + "\n" + metric);
        QString color = "#94A3B8";
        if (tone == DashboardTone::Good) color = "#34D399";
        else if (tone == DashboardTone::Warn) color = "#FBBF24";
        else if (tone == DashboardTone::Bad) color = "#F87171";
        cards.at(index)->setStyleSheet(
            QString("QPushButton#OverviewCard { border-left: 4px solid %1; }").arg(color));
    }

    void updateCards() {
        const qint64 now = metricsNowMs();
        const bool videoLive = g_hasVideoSignal.load() &&
                               dashboardIsRecent(g_lastVideoFrameMs.load(), now, 3000);
        const bool visionLive = dashboardIsRecent(g_lastVisionRecvMs.load(), now, 3000);
        const bool audioLive = dashboardIsRecent(g_lastAudioRecvMs.load(), now, 3000);
        const bool ridConnected = g_ridConnected.load() &&
                                  dashboardIsRecent(g_lastRidRecvMs.load(), now, 5000);
        const bool ridDataLive = dashboardIsRecent(g_lastRidDataMs.load(), now, 5000);
        const double temp = g_latestMaxTemperatureC.load();
        setCard(0, videoLive ? "1920x1080 / 在线" : "等待视频",
                videoLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(1, visionLive
                       ? QString("%1 FPS / %2 ms")
                             .arg(g_latestVisionFps.load(), 0, 'f', 1)
                             .arg(g_latestVisionInferMs.load(), 0, 'f', 1)
                       : "等待推理",
                visionLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(2, g_latestAutoTrackingEnabled.load() ? "已开启" : "已关闭",
                g_latestAutoTrackingEnabled.load() ? DashboardTone::Warn : DashboardTone::Idle);
        setCard(3, QString("P%1 T%2 Z%3")
                       .arg(g_latestCameraAzimuthUnit.load())
                       .arg(g_latestCameraElevationUnit.load())
                       .arg(g_latestCameraAbsoluteZoom.load()),
                videoLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(4, audioLive ? "6 通道在线" : "等待音频",
                audioLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(5, audioLive
                       ? QString("置信 %1").arg(dashboardPercentText(g_latestAudioScoreEma.load()))
                       : "等待推理",
                audioLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(6,
                ridDataLive ? "心跳正常"
                            : (ridConnected ? "已连接 / 等待报文" : "等待模块"),
                ridDataLive ? DashboardTone::Good
                            : (ridConnected ? DashboardTone::Warn : DashboardTone::Idle));
        const bool windowsLive =
            dashboardIsRecent(g_lastWindowsSyncMs.load(), now, 4000);
        setCard(7, windowsLive ? "客户端在线" : "等待 Windows 客户端",
                windowsLive ? DashboardTone::Good : DashboardTone::Idle);
        setCard(8, g_fusedRecordingRequested.load() ? "RK同步采集中" : "未录制",
                g_fusedRecordingRequested.load() ? DashboardTone::Bad : DashboardTone::Idle);
        setCard(9, temp >= 85.0
                       ? QString("%1°C / 请散热").arg(temp, 0, 'f', 1)
                       : QString("CPU %1% / %2°C")
                             .arg(g_latestSystemCpuPercent.load(), 0, 'f', 0)
                             .arg(temp, 0, 'f', 1),
                temp >= 85.0 ? DashboardTone::Bad : DashboardTone::Good);
    }

    std::function<void(int)> openPage;
    QList<QPushButton*> cards;
    QTimer timer;
};

enum class DetailPageKind {
    Video,
    Audio,
    Recording,
    System
};

class ModuleDetailPage : public QFrame {
public:
    ModuleDetailPage(DetailPageKind kind,
                     DashboardWindow* controller,
                     QWidget* parent = nullptr)
        : QFrame(parent), kind(kind), dashboard(controller) {
        setObjectName("ModuleDetailPage");
        QVBoxLayout* root = new QVBoxLayout(this);
        root->setContentsMargins(10, 6, 10, 7);
        root->setSpacing(6);
        title = new QLabel();
        title->setObjectName("PanelTitle");
        root->addWidget(title);

        QGridLayout* grid = new QGridLayout();
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(8);
        for (int row = 0; row < 10; ++row) {
            names[row] = new QLabel();
            names[row]->setObjectName("MetricName");
            values[row] = new QLabel("--");
            values[row]->setObjectName("MetricValue");
            values[row]->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            names[row]->setVisible(false);
            values[row]->setVisible(false);
            grid->addWidget(names[row], row, 0);
            grid->addWidget(values[row], row, 1);
        }
        root->addLayout(grid);

        if (kind == DetailPageKind::Video) {
            title->setText("视频与跟踪");
            detectionButton = new QPushButton("切换检测开关");
            detectionButton->setObjectName("ControlButton");
            detectionButton->setMinimumHeight(40);
            root->addWidget(detectionButton);
            connect(detectionButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestToggleDetectionEnabled();
            });

            modeCombo = new QComboBox();
            modeCombo->addItem("高召回搜索", 0);
            modeCombo->addItem("均衡搜索", 1);
            modeCombo->addItem("低误判确认", 2);
            modeCombo->addItem("保守报警", 3);
            modeCombo->addItem("自定义", 4);
            modeCombo->setMinimumHeight(32);
            root->addWidget(modeCombo);

            QGridLayout* params = new QGridLayout();
            params->setHorizontalSpacing(8);
            params->setVerticalSpacing(5);
            addParamSlider(params, 0, "conf", confSlider, confValue, 20, 90, 35, 100.0);
            addParamSlider(params, 1, "搜索框", searchMinBoxSlider, searchMinBoxValue, 1, 30, 4, 100.0);
            addParamSlider(params, 2, "跟踪框", trackMinBoxSlider, trackMinBoxValue, 2, 35, 6, 100.0);
            addParamSlider(params, 3, "报警框", alarmMinBoxSlider, alarmMinBoxValue, 3, 40, 8, 100.0);
            addParamSlider(params, 4, "确认帧", confirmFramesSlider, confirmFramesValue, 1, 8, 3, 1.0);
            addParamSlider(params, 5, "理想下限", idealLowSlider, idealLowValue, 4, 24, 12, 100.0);
            addParamSlider(params, 6, "理想上限", idealHighSlider, idealHighValue, 12, 35, 22, 100.0);
            addParamSlider(params, 7, "最大速度", maxSpeedSlider, maxSpeedValue, 12, 110, 90, 1.0);
            addParamSlider(params, 8, "pan增益", panGainSlider, panGainValue, 30, 220, 125, 100.0);
            addParamSlider(params, 9, "变焦步长", zoomStepSlider, zoomStepValue, 5, 180, 35, 1.0);
            addParamSlider(params, 10, "冷却ms", zoomCooldownSlider, zoomCooldownValue, 80, 3000, 120, 1.0);
            addParamSlider(params, 11, "丢失缩小", lostZoomSlider, lostZoomValue, 10, 220, 160, 1.0);
            addParamSlider(params, 12, "最小zoom", minZoomSlider, minZoomValue, 10, 200, 10, 10.0);
            addParamSlider(params, 13, "最大zoom", maxZoomSlider, maxZoomValue, 10, 370, 250, 10.0);
            addParamSlider(params, 14, "搜索zoom", lostSearchZoomSlider, lostSearchZoomValue, 10, 200, 12, 10.0);
            root->addLayout(params);

            QGridLayout* toggles = new QGridLayout();
            toggles->setSpacing(6);
            autoTrackToggle = makeCheckButton("自动跟踪");
            autoZoomToggle = makeCheckButton("自动变焦");
            autoFocusToggle = makeCheckButton("自动对焦");
            autoTrackToggle->setChecked(true);
            autoZoomToggle->setChecked(true);
            autoFocusToggle->setChecked(true);
            trackingModeCombo = new QComboBox();
            trackingModeCombo->addItem("稳定", 0);
            trackingModeCombo->addItem("均衡", 1);
            trackingModeCombo->addItem("快速", 2);
            trackingModeCombo->addItem("自适应", 3);
            trackingModeCombo->setMinimumHeight(32);
            QPushButton* restoreDefaults = new QPushButton("恢复默认参数");
            restoreDefaults->setObjectName("ControlButton");
            restoreDefaults->setMinimumHeight(32);
            QPushButton* resetAdaptive = new QPushButton("重置本次自适应学习");
            resetAdaptive->setObjectName("ControlButton");
            resetAdaptive->setMinimumHeight(32);
            QPushButton* focusNowButton = new QPushButton("立即对焦");
            focusNowButton->setObjectName("ControlButton");
            focusNowButton->setMinimumHeight(32);
            toggles->addWidget(autoTrackToggle, 0, 0);
            toggles->addWidget(autoZoomToggle, 0, 1);
            toggles->addWidget(autoFocusToggle, 0, 2);
            toggles->addWidget(trackingModeCombo, 1, 0);
            toggles->addWidget(restoreDefaults, 1, 1);
            toggles->addWidget(resetAdaptive, 1, 2);
            toggles->addWidget(focusNowButton, 2, 0, 1, 3);
            root->addLayout(toggles);

            auto sendParams = [this]() { sendVideoControlParams(); };
            connect(modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, sendParams);
            connect(trackingModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, sendParams);
            for (QSlider* slider : {confSlider, searchMinBoxSlider, trackMinBoxSlider,
                                    alarmMinBoxSlider, confirmFramesSlider, idealLowSlider,
                                    idealHighSlider, maxSpeedSlider, panGainSlider,
                                    zoomStepSlider, zoomCooldownSlider, lostZoomSlider,
                                    minZoomSlider, maxZoomSlider, lostSearchZoomSlider}) {
                connect(slider, &QSlider::valueChanged, this, sendParams);
            }
            connect(autoTrackToggle, &QPushButton::toggled, this, sendParams);
            connect(autoZoomToggle, &QPushButton::toggled, this, sendParams);
            connect(autoFocusToggle, &QPushButton::toggled, this, sendParams);
            connect(restoreDefaults, &QPushButton::clicked, this, [this]() {
                modeCombo->setCurrentIndex(1);
                trackingModeCombo->setCurrentIndex(1);
                confSlider->setValue(35);
                searchMinBoxSlider->setValue(4);
                trackMinBoxSlider->setValue(6);
                alarmMinBoxSlider->setValue(8);
                confirmFramesSlider->setValue(3);
                idealLowSlider->setValue(12);
                idealHighSlider->setValue(22);
                maxSpeedSlider->setValue(90);
                panGainSlider->setValue(125);
                zoomStepSlider->setValue(35);
                zoomCooldownSlider->setValue(120);
                lostZoomSlider->setValue(160);
                minZoomSlider->setValue(10);
                maxZoomSlider->setValue(250);
                lostSearchZoomSlider->setValue(12);
                autoTrackToggle->setChecked(true);
                autoZoomToggle->setChecked(true);
                autoFocusToggle->setChecked(true);
                sendVideoControlParams();
            });
            connect(resetAdaptive, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestResetAudioCalibration(true);
            });
            connect(focusNowButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestAutofocus();
            });
            sendVideoControlParams();
        } else if (kind == DetailPageKind::Audio) {
            title->setText("音频 / DOA");
            audioGuidanceButton = new QPushButton("声学引导：关");
            audioGuidanceButton->setObjectName("ControlButton");
            audioGuidanceButton->setMinimumHeight(36);
            root->addWidget(audioGuidanceButton);
            connect(audioGuidanceButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) {
                    dashboard->requestSetAudioGuidance(!g_audioGuidanceRequested.load());
                }
            });

            QHBoxLayout* calibrationRow = new QHBoxLayout();
            calibrationRow->setSpacing(6);
            markCalibrationButton = new QPushButton("设置当前无人机方向");
            markCalibrationButton->setObjectName("ControlButton");
            markCalibrationButton->setMinimumHeight(36);
            calibrationRow->addWidget(markCalibrationButton);
            connect(markCalibrationButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestMarkAudioCalibrationTarget();
            });

            resetCalibrationButton = new QPushButton("重新标定");
            resetCalibrationButton->setObjectName("ControlButton");
            resetCalibrationButton->setMinimumHeight(36);
            calibrationRow->addWidget(resetCalibrationButton);
            connect(resetCalibrationButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestResetAudioCalibration(false);
            });
            root->addLayout(calibrationRow);

            resetLearningButton = new QPushButton("清空本轮学习");
            resetLearningButton->setObjectName("ControlButton");
            resetLearningButton->setMinimumHeight(36);
            root->addWidget(resetLearningButton);
            connect(resetLearningButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestResetAudioCalibration(true);
            });
        } else if (kind == DetailPageKind::Recording) {
            title->setText("录像服务");
        } else {
            title->setText("系统设置与资源");
            if (antiUavNodeRole() != "windows") {
                startWindowsButton = new QPushButton("启动并连接 Windows 界面");
                startWindowsButton->setObjectName("ControlButton");
                startWindowsButton->setMinimumHeight(36);
                root->addWidget(startWindowsButton);
                connect(startWindowsButton, &QPushButton::clicked, this, [this]() {
                    if (dashboard) dashboard->requestStartWindowsClient();
                });

                disconnectWindowsButton = new QPushButton("断开 Windows");
                disconnectWindowsButton->setObjectName("ControlButton");
                disconnectWindowsButton->setMinimumHeight(36);
                root->addWidget(disconnectWindowsButton);
                connect(disconnectWindowsButton, &QPushButton::clicked, this, [this]() {
                    if (dashboard) dashboard->requestDisconnectWindowsClient();
                });

                autoReconnectWindowsButton = new QPushButton("自动重连：关");
                autoReconnectWindowsButton->setObjectName("ControlButton");
                autoReconnectWindowsButton->setMinimumHeight(36);
                root->addWidget(autoReconnectWindowsButton);
                connect(autoReconnectWindowsButton, &QPushButton::clicked, this, [this]() {
                    if (dashboard) dashboard->requestToggleWindowsAutoReconnect();
                });
            }
            QPushButton* exitButton = new QPushButton(
                antiUavNodeRole() == "windows" ? "退出程序" : "退出到桌面");
            exitButton->setObjectName("ControlButton");
            exitButton->setMinimumHeight(36);
            root->addWidget(exitButton);
            connect(exitButton, &QPushButton::clicked, this, [this]() {
                if (dashboard) dashboard->requestExitToDesktop();
            });
            if (antiUavNodeRole() != "windows") {
                stopSystemButton = new QPushButton("停止系统");
                stopSystemButton->setObjectName("StopButton");
                stopSystemButton->setMinimumHeight(36);
                root->addWidget(stopSystemButton);
                connect(stopSystemButton, &QPushButton::clicked, this, [this]() {
                    stopSystemRequestedMs = metricsNowMs();
                    if (stopSystemButton) {
                        stopSystemButton->setText("停止中...");
                        stopSystemButton->setEnabled(false);
                    }
                    if (dashboard) {
                        dashboard->requestStopSystem();
                    }
                });

                restoreSystemButton = new QPushButton("恢复系统");
                restoreSystemButton->setObjectName("RestoreButton");
                restoreSystemButton->setMinimumHeight(36);
                root->addWidget(restoreSystemButton);
                connect(restoreSystemButton, &QPushButton::clicked, this, [this]() {
                    restoreSystemRequestedMs = metricsNowMs();
                    if (restoreSystemButton) {
                        restoreSystemButton->setText("恢复中...");
                        restoreSystemButton->setEnabled(false);
                    }
                    if (dashboard) {
                        dashboard->requestRestoreSystem();
                    }
                });
            }
        }
        root->addStretch(1);

        timer.setInterval(500);
        connect(&timer, &QTimer::timeout, this, [this]() { updateState(); });
        timer.start();
        updateState();
    }

private:
    QPushButton* makeCheckButton(const QString& text) {
    QPushButton* button = new QPushButton(text + "：关");
        button->setObjectName("ControlButton");
        button->setCheckable(true);
        button->setMinimumHeight(32);
        return button;
    }

    void addParamSlider(QGridLayout* grid,
                        int row,
                        const QString& name,
                        QSlider*& slider,
                        QLabel*& valueLabel,
                        int minValue,
                        int maxValue,
                        int defaultValue,
                        double scale) {
        QLabel* label = new QLabel(name);
        label->setObjectName("MetricName");
        slider = new QSlider(Qt::Horizontal);
        slider->setRange(minValue, maxValue);
        slider->setValue(defaultValue);
        slider->setMinimumWidth(120);
        valueLabel = new QLabel();
        valueLabel->setObjectName("MetricValue");
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        grid->addWidget(label, row, 0);
        grid->addWidget(slider, row, 1);
        grid->addWidget(valueLabel, row, 2);
        connect(slider, &QSlider::valueChanged, this, [valueLabel, scale](int value) {
            valueLabel->setText(scale == 1.0
                                    ? QString::number(value)
                                    : QString::number(value / scale, 'f', 2));
        });
        valueLabel->setText(scale == 1.0
                                ? QString::number(defaultValue)
                                : QString::number(defaultValue / scale, 'f', 2));
    }

    double sliderScaled(QSlider* slider, double scale) const {
        return slider ? slider->value() / scale : 0.0;
    }

    void sendVideoControlParams() {
        if (kind != DetailPageKind::Video || dashboard == nullptr || modeCombo == nullptr) return;
        if (idealHighSlider && idealLowSlider && idealHighSlider->value() < idealLowSlider->value()) {
            idealHighSlider->blockSignals(true);
            idealHighSlider->setValue(idealLowSlider->value());
            idealHighSlider->blockSignals(false);
        }
        QJsonObject params;
        params["detection_mode"] = modeCombo->currentData().toInt();
        const double conf = sliderScaled(confSlider, 100.0);
        params["search_conf"] = conf;
        params["track_conf"] = conf;
        params["alarm_conf"] = conf;
        params["search_min_box"] = sliderScaled(searchMinBoxSlider, 100.0);
        params["track_min_box"] = sliderScaled(trackMinBoxSlider, 100.0);
        params["alarm_min_box"] = sliderScaled(alarmMinBoxSlider, 100.0);
        params["confirm_frames"] = static_cast<int>(sliderScaled(confirmFramesSlider, 1.0));
        params["target_box_min"] = 0.10;
        params["target_box_ideal_low"] = sliderScaled(idealLowSlider, 100.0);
        params["target_box_ideal_high"] = sliderScaled(idealHighSlider, 100.0);
        params["target_box_max"] = 0.25;
        params["min_zoom"] = sliderScaled(minZoomSlider, 10.0);
        params["max_zoom"] = sliderScaled(maxZoomSlider, 10.0);
        params["lost_search_zoom"] = sliderScaled(lostSearchZoomSlider, 10.0);
        params["auto_track_enabled"] = autoTrackToggle && autoTrackToggle->isChecked() ? 1 : 0;
        params["auto_zoom_enabled"] = autoZoomToggle && autoZoomToggle->isChecked() ? 1 : 0;
        params["auto_focus_enabled"] = autoFocusToggle && autoFocusToggle->isChecked() ? 1 : 0;
        params["tracking_mode"] = trackingModeCombo ? trackingModeCombo->currentData().toInt() : 1;
        params["pan_gain"] = sliderScaled(panGainSlider, 100.0);
        params["tilt_gain"] = sliderScaled(panGainSlider, 100.0);
        params["max_speed"] = static_cast<int>(sliderScaled(maxSpeedSlider, 1.0));
        params["high_zoom_max_speed"] = qBound(30, static_cast<int>(sliderScaled(maxSpeedSlider, 1.0) * 0.70), 75);
        params["center_dead_zone"] = 0.040;
        params["feedforward_gain"] = 0.10;
        params["zoom_step_in"] = static_cast<int>(sliderScaled(zoomStepSlider, 1.0));
        params["zoom_step_out"] = qBound(5, static_cast<int>(sliderScaled(zoomStepSlider, 1.0) * 2.0), 180);
        params["zoom_cooldown_ms"] = static_cast<int>(sliderScaled(zoomCooldownSlider, 1.0));
        params["lost_zoom_out_step"] = static_cast<int>(sliderScaled(lostZoomSlider, 1.0));
        params["focus_cooldown_ms"] = 5000;
        dashboard->requestSetControlParams(params);
    }

    void setRow(int row, const QString& name, const QString& value) {
        if (row < 0 || row >= 10) return;
        names[row]->setText(name);
        values[row]->setText(value);
        names[row]->setVisible(true);
        values[row]->setVisible(true);
    }

    void updateState() {
        const qint64 now = metricsNowMs();
        for (int row = 0; row < 10; ++row) {
            names[row]->setVisible(false);
            values[row]->setVisible(false);
        }
        if (kind == DetailPageKind::Video) {
            setRow(0, "摄像头", g_hasVideoSignal.load() ? "在线 / 完整画幅" : "等待视频");
            setRow(1, "模型", antiUavYoloModelName());
            setRow(2, "FPS / 推理", QString("%1 / %2 ms")
                                          .arg(g_latestVisionFps.load(), 0, 'f', 1)
                                          .arg(g_latestVisionInferMs.load(), 0, 'f', 1));
            setRow(3, "检测 / 跟踪", QString("%1 / %2")
                                           .arg(g_latestVisionDetected.load() ? "有目标" : "无目标")
                                           .arg(g_latestVisionTracking.load() ? "跟踪中" : "未跟踪"));
            setRow(4, "置信度", QString::number(g_latestVisionConf.load(), 'f', 2));
            setRow(5, "变焦状态", QString("Z%1 / ratio %2 / speed %3")
                                      .arg(g_latestCameraAbsoluteZoom.load())
                                      .arg(g_latestZoomRatio.load(), 0, 'f', 2)
                                      .arg(g_latestZoomSpeed.load(), 0, 'f', 0));
            setRow(6, "对焦状态", QString("%1 / 冷却 %2ms")
                                      .arg(dashboardText(g_latestFocusState))
                                      .arg(g_latestFocusCooldownRemainingMs.load()));
            setRow(7, "最近对焦", g_latestFocusRequestMs.load() > 0
                                      ? QString::number(g_latestFocusRequestMs.load())
                                      : "--");
            setRow(8, "融合控制", QString("%1 / %2")
                                      .arg(dashboardText(g_latestFusionState))
                                      .arg(dashboardText(g_latestControlSource)));
            setRow(9, "搜索状态", dashboardText(g_latestSearchState));
            if (detectionButton) {
                detectionButton->setText(
                    g_detectionEnabledRequested.load() ? "关闭无人机检测" : "开启无人机检测");
            }
            if (autoTrackToggle && autoTrackToggle->isChecked() != g_latestAutoTrackingEnabled.load()) {
                autoTrackToggle->blockSignals(true);
                autoTrackToggle->setChecked(g_latestAutoTrackingEnabled.load());
                autoTrackToggle->blockSignals(false);
            }
            if (autoZoomToggle && autoZoomToggle->isChecked() != g_latestAutoZoomEnabled.load()) {
                autoZoomToggle->blockSignals(true);
                autoZoomToggle->setChecked(g_latestAutoZoomEnabled.load());
                autoZoomToggle->blockSignals(false);
            }
            if (autoTrackToggle) autoTrackToggle->setText(autoTrackToggle->isChecked() ? "自动跟踪：开" : "自动跟踪：关");
            if (autoZoomToggle) autoZoomToggle->setText(autoZoomToggle->isChecked() ? "自动变焦：开" : "自动变焦：关");
            if (autoFocusToggle) autoFocusToggle->setText(autoFocusToggle->isChecked() ? "自动对焦：开" : "自动对焦：关");
        } else if (kind == DetailPageKind::Audio) {
            const bool live = dashboardIsRecent(g_lastAudioRecvMs.load(), now, 3000);
            const bool requested = g_audioGuidanceRequested.load();
            const bool activelyGuided = g_latestAudioGuided.load();
            QString guidanceState = dashboardText(g_latestAudioGuidanceState);
            if (activelyGuided) guidanceState = "guiding / " + guidanceState;
            if (!requested) guidanceState = "ui_off / " + guidanceState;
            setRow(0, "ReSpeaker", live ? "6 通道在线" : "等待设备数据");
            setRow(1, "声学引导", guidanceState);
            setRow(2, "YAMNet", live ? dashboardText(g_latestAudioState) : "等待推理");
            setRow(3, "置信 / RMS", QString("%1 / %2 dB")
                                        .arg(dashboardPercentText(g_latestAudioScoreEma.load()))
                                        .arg(g_latestAudioRmsDbfs.load(), 0, 'f', 1));
            setRow(4, "DOA", QString("%1° / 稳定 %2%")
                                   .arg(g_latestAudioAngle.load(), 0, 'f', 1)
                                   .arg(g_latestAudioDoaStability.load() * 100.0, 0, 'f', 0));
            setRow(5, "当前 pan", dashboardAngleText(g_latestCameraAzimuthDeg.load()));
            setRow(6, "映射目标 pan", dashboardAngleText(g_latestAudioWorldAzimuthDeg.load()));
            setRow(7, "本次偏置", QString("%1 / 置信 %2")
                                   .arg(dashboardAngleText(g_latestMicOffsetDeg.load()))
                                   .arg(dashboardConfidenceText(dashboardText(g_latestOffsetConfidenceLabel),
                                                               g_latestOffsetConfidence.load())));
            setRow(8, "校准状态", QString("%1 / 样本 %2")
                                      .arg(dashboardText(g_latestAudioCalibrationState))
                                      .arg(g_latestCalibrationSamples.load()));
            setRow(9, "拒绝/状态", QString("%1 / %2")
                                  .arg(dashboardText(g_latestAudioRejectReason))
                                  .arg(dashboardText(g_latestSearchState)));
            if (audioGuidanceButton) {
                audioGuidanceButton->setText(
                    requested
                        ? (activelyGuided ? "声学引导：开 / 接管中" : "声学引导：开")
                        : "声学引导：关");
            }
        } else if (kind == DetailPageKind::Recording) {
            setRow(0, "融合会话", g_fusedRecordingRequested.load() ? "录制中" : "未录制");
            setRow(1, "界面录屏", g_recordingRequested.load() ? "录制中" : "未录制");
            setRow(2, "原始视频", g_rawRecordingRequested.load() ? "录制中" : "未录制");
            setRow(3, "原始音频", g_audioRawRecordingRequested.load() ? "录制中" : "未录制");
            setRow(4, "会话 ID", currentCaptureSessionId().isEmpty()
                                      ? "--" : currentCaptureSessionId());
        } else {
            const double temp = g_latestMaxTemperatureC.load();
            const bool backendLive =
                dashboardIsRecent(g_lastVisionRecvMs.load(), now, 3500) ||
                dashboardIsRecent(g_lastAudioRecvMs.load(), now, 3500) ||
                (g_ridConnected.load() && dashboardIsRecent(g_lastRidRecvMs.load(), now, 5000));
            const bool stopRecent =
                stopSystemRequestedMs > 0 && (now - stopSystemRequestedMs) < 600000;
            const bool stopPending =
                stopSystemRequestedMs > 0 && (now - stopSystemRequestedMs) < 6000 && backendLive;
            const bool restorePending =
                restoreSystemRequestedMs > 0 && (now - restoreSystemRequestedMs) < 12000 && !backendLive;
            if (backendLive && restoreSystemRequestedMs > 0) {
                restoreSystemRequestedMs = -1;
                stopSystemRequestedMs = -1;
            }
            setRow(0, "CPU", QString("%1%").arg(g_latestSystemCpuPercent.load(), 0, 'f', 1));
            setRow(1, "内存", QString("%1%").arg(g_latestSystemMemPercent.load(), 0, 'f', 1));
            setRow(2, "NPU", QString("%1% / %2 MHz")
                                 .arg(g_latestNpuLoadPercent.load(), 0, 'f', 0)
                                 .arg(g_latestNpuFreqMhz.load(), 0, 'f', 0));
            setRow(3, "温度", temp >= 85.0
                                  ? QString("%1°C / 请主动散热").arg(temp, 0, 'f', 1)
                                  : QString("%1°C").arg(temp, 0, 'f', 1));
            const bool windowsLive = dashboardIsRecent(g_lastWindowsSyncMs.load(), now, 5000);
            setRow(4, "Windows", windowsLive ? "界面在线 / 已同步" : windowsLaunchStatus());
            setRow(5, "后端", backendLive
                                  ? "运行中"
                                  : (restorePending ? "恢复中" : "已停止 / 可恢复"));
            if (autoReconnectWindowsButton) {
                autoReconnectWindowsButton->setText(
                    g_windowsAutoReconnect.load() ? "自动重连：开" : "自动重连：关");
            }
            if (stopSystemButton) {
                stopSystemButton->setText(stopPending ? "停止中..." : "停止系统");
                stopSystemButton->setEnabled(!stopPending && !restorePending);
            }
            if (restoreSystemButton) {
                const bool restoreReady = !restorePending &&
                    (!backendLive || (stopRecent && (now - stopSystemRequestedMs) >= 1500));
                restoreSystemButton->setText(restorePending ? "恢复中..." : "恢复系统");
                restoreSystemButton->setEnabled(restoreReady);
            }
        }
    }

    DetailPageKind kind;
    DashboardWindow* dashboard = nullptr;
    QLabel* title = nullptr;
    QLabel* names[10]{};
    QLabel* values[10]{};
    QPushButton* detectionButton = nullptr;
    QComboBox* modeCombo = nullptr;
    QComboBox* trackingModeCombo = nullptr;
    QSlider* confSlider = nullptr;
    QSlider* searchMinBoxSlider = nullptr;
    QSlider* trackMinBoxSlider = nullptr;
    QSlider* alarmMinBoxSlider = nullptr;
    QSlider* confirmFramesSlider = nullptr;
    QSlider* idealLowSlider = nullptr;
    QSlider* idealHighSlider = nullptr;
    QSlider* maxSpeedSlider = nullptr;
    QSlider* panGainSlider = nullptr;
    QSlider* zoomStepSlider = nullptr;
    QSlider* zoomCooldownSlider = nullptr;
    QSlider* lostZoomSlider = nullptr;
    QSlider* minZoomSlider = nullptr;
    QSlider* maxZoomSlider = nullptr;
    QSlider* lostSearchZoomSlider = nullptr;
    QLabel* confValue = nullptr;
    QLabel* searchMinBoxValue = nullptr;
    QLabel* trackMinBoxValue = nullptr;
    QLabel* alarmMinBoxValue = nullptr;
    QLabel* confirmFramesValue = nullptr;
    QLabel* idealLowValue = nullptr;
    QLabel* idealHighValue = nullptr;
    QLabel* maxSpeedValue = nullptr;
    QLabel* panGainValue = nullptr;
    QLabel* zoomStepValue = nullptr;
    QLabel* zoomCooldownValue = nullptr;
    QLabel* lostZoomValue = nullptr;
    QLabel* minZoomValue = nullptr;
    QLabel* maxZoomValue = nullptr;
    QLabel* lostSearchZoomValue = nullptr;
    QPushButton* autoTrackToggle = nullptr;
    QPushButton* autoZoomToggle = nullptr;
    QPushButton* autoFocusToggle = nullptr;
    QPushButton* startWindowsButton = nullptr;
    QPushButton* disconnectWindowsButton = nullptr;
    QPushButton* autoReconnectWindowsButton = nullptr;
    QPushButton* stopSystemButton = nullptr;
    QPushButton* restoreSystemButton = nullptr;
    QPushButton* audioGuidanceButton = nullptr;
    QPushButton* markCalibrationButton = nullptr;
    QPushButton* resetCalibrationButton = nullptr;
    QPushButton* resetLearningButton = nullptr;
    qint64 stopSystemRequestedMs = -1;
    qint64 restoreSystemRequestedMs = -1;
    QTimer timer;
};

class RemoteIdPanel : public QFrame {
public:
    explicit RemoteIdPanel(QWidget* parent = nullptr) : QFrame(parent) {
        setObjectName("RemoteIdPanel");
        QVBoxLayout* root = new QVBoxLayout(this);
        root->setContentsMargins(10, 6, 10, 7);
        root->setSpacing(6);

        QLabel* title = new QLabel("RID900 / Remote ID");
        title->setObjectName("PanelTitle");
        root->addWidget(title);

        QGridLayout* summary = new QGridLayout();
        summary->setHorizontalSpacing(10);
        summary->setVerticalSpacing(7);
        summary->addWidget(nameLabel("连接"), 0, 0);
        statusValue = valueLabel();
        summary->addWidget(statusValue, 0, 1);
        summary->addWidget(nameLabel("串口"), 1, 0);
        sourceValue = valueLabel();
        summary->addWidget(sourceValue, 1, 1);
        summary->addWidget(nameLabel("设备"), 2, 0);
        deviceValue = valueLabel();
        summary->addWidget(deviceValue, 2, 1);
        summary->addWidget(nameLabel("模块位置"), 3, 0);
        modulePositionValue = valueLabel();
        summary->addWidget(modulePositionValue, 3, 1);
        summary->addWidget(nameLabel("目标/更新"), 4, 0);
        targetCountValue = valueLabel();
        summary->addWidget(targetCountValue, 4, 1);
        root->addLayout(summary);

        table = new QTableWidget(0, 4);
        table->setHorizontalHeaderLabels({"目标", "信号", "位置", "飞行"});
        table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        table->verticalHeader()->setVisible(false);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionMode(QAbstractItemView::NoSelection);
        table->setWordWrap(true);
        table->setAlternatingRowColors(true);
        root->addWidget(table, 1);

        QLabel* hint = new QLabel("USB Type-C 虚拟串口，115200 8N1；未连接模块时不影响视觉系统。");
        hint->setObjectName("PanelHint");
        hint->setWordWrap(true);
        root->addWidget(hint);

        timer.setInterval(500);
        connect(&timer, &QTimer::timeout, this, [this]() { updateState(); });
        timer.start();
        updateState();
    }

private:
    QLabel* nameLabel(const QString& text) {
        QLabel* label = new QLabel(text);
        label->setObjectName("MetricName");
        return label;
    }

    QLabel* valueLabel() {
        QLabel* label = new QLabel("--");
        label->setObjectName("MetricValue");
        label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        return label;
    }

    void setCell(int row, int column, const QString& text) {
        QTableWidgetItem* item = table->item(row, column);
        if (!item) {
            item = new QTableWidgetItem();
            table->setItem(row, column, item);
        }
        item->setText(text);
        item->setTextAlignment(Qt::AlignCenter);
    }

    void updateState() {
        const qint64 now = metricsNowMs();
        QList<RidTargetState> targets;
        QString status;
        QString source;
        QString device;
        QString modulePosition;
        {
            QMutexLocker locker(&g_ridMutex);
            targets = g_ridTargets.values();
            status = g_ridStatusText;
            source = g_ridSource;
            device = g_ridDeviceId;
            modulePosition = g_ridModulePosition;
        }
        std::sort(targets.begin(), targets.end(),
                  [](const RidTargetState& a, const RidTargetState& b) {
                      return a.lastSeenMs > b.lastSeenMs;
                  });

        const bool connected = g_ridConnected.load() &&
                               dashboardIsRecent(g_lastRidRecvMs.load(), now, 5000);
        const bool dataLive = dashboardIsRecent(g_lastRidDataMs.load(), now, 5000);
        statusValue->setText(
            dataLive ? status : (connected ? "已连接 / 等待报文" : "等待模块"));
        statusValue->setStyleSheet(
            dashboardPillStyle(dataLive ? DashboardTone::Good
                                        : (connected ? DashboardTone::Warn
                                                     : DashboardTone::Idle)));
        sourceValue->setText(source);
        deviceValue->setText(device);
        modulePositionValue->setText(modulePosition);
        targetCountValue->setText(
            QString("%1 / %2")
                .arg(targets.size())
                .arg(dashboardAgeText(g_lastRidDataMs.load(), now)));

        table->setRowCount(targets.size());
        for (int row = 0; row < targets.size(); ++row) {
            const RidTargetState& target = targets.at(row);
            setCell(row, 0, QString("%1\n%2 %3")
                                .arg(target.serialNumber, target.vendor, target.productType));
            setCell(row, 1, QString("%1 dBm\nSNR %2")
                                .arg(target.rssiDbm)
                                .arg(target.snrDb));
            setCell(row, 2, QString("%1\n%2")
                                .arg(target.longitude, 0, 'f', 6)
                                .arg(target.latitude, 0, 'f', 6));
            setCell(row, 3, QString("H %1 m\nV %2 m/s / %3°")
                                .arg(target.heightM, 0, 'f', 1)
                                .arg(target.speedMps, 0, 'f', 1)
                                .arg(target.yawDeg));
            table->setRowHeight(row, 58);
        }
    }

    QLabel* statusValue = nullptr;
    QLabel* sourceValue = nullptr;
    QLabel* deviceValue = nullptr;
    QLabel* modulePositionValue = nullptr;
    QLabel* targetCountValue = nullptr;
    QTableWidget* table = nullptr;
    QTimer timer;
};

// ============================================================
// 7. Qt 窗口录制：录制完整界面而不是只录制 RTSP 原始帧
// ============================================================

class WindowRecorder : public QObject {
public:
    explicit WindowRecorder(QWidget* targetWidget, QObject* parent = nullptr)
        : QObject(parent), target(targetWidget) {
        timer.setInterval(40); // 25 FPS，兼顾画面流畅度和本机写盘压力。
        connect(&timer, &QTimer::timeout, this, [this]() { tick(); });
        timer.start();
    }

    ~WindowRecorder() override {
        closeWriter("abnormal_end");
    }

private:
    QWidget* target = nullptr;
    QTimer timer;
    cv::VideoWriter writer;
    cv::Size writerSize;
    QString activePath;
    QString activeRecordId;
    QString activeSessionId;
    QString activeSource;
    QString activeCodec;
    QDateTime activeStartTime;
    qint64 startedMs = 0;
    int framesWritten = 0;

    void tick() {
        if (!target) {
            closeWriter("abnormal_no_target");
            return;
        }

        if (!g_recordingRequested.load()) {
            closeWriter("manual_stop");
            return;
        }

        if (g_recordingPaused.load()) {
            setWindowsRecordingStatus("暂停");
            return;
        }

        QPixmap pixmap = target->grab();
        if (pixmap.isNull()) {
            return;
        }

        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
        if (image.isNull() || image.width() <= 0 || image.height() <= 0) {
            return;
        }

        cv::Mat rgb(image.height(),
                    image.width(),
                    CV_8UC3,
                    const_cast<uchar*>(image.constBits()),
                    static_cast<size_t>(image.bytesPerLine()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        if (!writer.isOpened()) {
            openWriter(bgr.size());
        }

        if (!writer.isOpened()) {
            return;
        }

        if (bgr.size() != writerSize) {
            cv::resize(bgr, bgr, writerSize);
        }
        writer.write(bgr);
        ++framesWritten;
    }

    void openWriter(const cv::Size& size) {
        if (!recordingStorageAllowed("界面录屏", false)) {
            g_recordingRequested.store(false);
            g_recordingActive.store(false);
            setCurrentRecordingPath("");
            return;
        }
#ifdef Q_OS_WIN
        WindowsRecordingFilePlan plan = makeWindowsRecordingFilePlan("mp4");
        QString path = plan.path;
#else
        const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
        const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        QString path = makeUniquePath(recordingDirPath(date), stamp + "_mipi_dashboard", "mp4", 1);
#endif

        writerSize = size;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(path.toStdString(), fourcc, 25.0, writerSize, true);
        QString codec = "mp4v";

        if (!writer.isOpened()) {
#ifdef Q_OS_WIN
            plan = makeWindowsRecordingFilePlan("avi", &plan);
            path = plan.path;
#else
            path = makeUniquePath(recordingDirPath(date), stamp + "_mipi_dashboard", "avi", 1);
#endif
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open(path.toStdString(), fourcc, 25.0, writerSize, true);
            codec = "MJPG";
        }

        if (writer.isOpened()) {
            activePath = path;
#ifdef Q_OS_WIN
            activeRecordId = plan.recordId;
            activeSessionId = plan.sessionId;
            activeSource = plan.source;
            activeStartTime = plan.startTime;
#else
            activeRecordId.clear();
            activeSessionId = currentCaptureSessionId();
            activeSource = "local";
            activeStartTime = QDateTime::currentDateTime();
#endif
            activeCodec = codec;
            startedMs = metricsNowMs();
            framesWritten = 0;
            setCurrentRecordingPath(path);
            g_recordingActive.store(true);
            setWindowsRecordingStatus("录制中");
            appendWindowsRecordingEvent("start", path);
#ifdef Q_OS_WIN
            appendWindowsRecordingIndexEvent("start",
                                             activeRecordId,
                                             activeSessionId,
                                             activeSource,
                                             activeStartTime,
                                             QDateTime(),
                                             path,
                                             activeCodec,
                                             writerSize,
                                             0,
                                             "");
#endif
            qInfo() << "[REC] dashboard recording started:" << path
                    << "size=" << writerSize.width << "x" << writerSize.height;
        } else {
            g_recordingRequested.store(false);
            g_recordingActive.store(false);
            setWindowsRecordingStatus("保存失败");
            setCurrentRecordingPath("");
            qWarning() << "[REC] failed to open dashboard recorder";
        }
    }

    void closeWriter(const QString& reason) {
        if (writer.isOpened()) {
            writer.release();
            const QString path = activePath.isEmpty() ? currentRecordingPath() : activePath;
            const bool normalStop = (reason == "manual_stop" || reason == "stop");
            const QString event = normalStop ? "end" : "abnormal_end";
            appendWindowsRecordingEvent(normalStop ? "stop" : "abnormal_end", path);
            writeWindowsRecordingSummary(path, startedMs, framesWritten, "保存完成");
            setLastRecordingPath(path);
#ifdef Q_OS_WIN
            appendWindowsRecordingIndexEvent(event,
                                             activeRecordId,
                                             activeSessionId,
                                             activeSource,
                                             activeStartTime,
                                             QDateTime::currentDateTime(),
                                             path,
                                             activeCodec,
                                             writerSize,
                                             framesWritten,
                                             reason);
#endif
            setWindowsRecordingStatus("保存完成");
            qInfo() << "[REC] dashboard recording stopped:" << path;
        }
        g_recordingActive.store(false);
        activePath.clear();
        activeRecordId.clear();
        activeSessionId.clear();
        activeSource.clear();
        activeCodec.clear();
        activeStartTime = QDateTime();
        startedMs = 0;
        framesWritten = 0;
        setCurrentRecordingPath("");
    }
};

// ============================================================
// 7.1 原始摄像头录制：直接保存无叠加视频帧，便于后续人工标注数据集
// ============================================================

class RawCameraRecorder {
public:
    RawCameraRecorder()
        : decoupleWrites(envIntBounded("ANTI_UAV_QT_RECORD_DECOUPLE", 1, 0, 1) != 0),
          queueMax(envIntBounded("ANTI_UAV_QT_RECORD_QUEUE_MAX", 8, 1, 120)) {
    }

    ~RawCameraRecorder() {
        stopWorker();
        closeWriter();
    }

    void writeFrame(const cv::Mat& frameBgr, int frameId = -1) {
        if (!g_rawRecordingRequested.load()) {
            if (decoupleWrites) {
                recordCv.notify_one();
            } else {
                closeWriter();
            }
            return;
        }
        if (frameBgr.empty()) {
            return;
        }

        if (!decoupleWrites) {
            writeFrameSync(frameBgr, frameId);
            return;
        }

        ensureWorker();
        {
            std::lock_guard<std::mutex> lock(recordMutex);
            if (recordQueue.size() >= static_cast<size_t>(queueMax)) {
                recordQueue.pop_front();
                g_frameStreamRecordingDroppedFrames.fetch_add(1);
            }
            recordQueue.push_back({frameBgr.clone(), frameId});
            g_frameStreamRecordingQueueSize.store(static_cast<int>(recordQueue.size()));
        }
        recordCv.notify_one();
    }

private:
    struct PendingRecordFrame {
        cv::Mat frame;
        int frameId = -1;
    };

    cv::VideoWriter writer;
    cv::Size writerSize;
    QString activePath;
    QString activeRecordId;
    QString activeSessionId;
    QString activeSource;
    QString activeCodec;
    QDateTime activeStartTime;
    int framesWritten = 0;
    const bool decoupleWrites;
    const int queueMax;
    std::deque<PendingRecordFrame> recordQueue;
    std::mutex recordMutex;
    std::condition_variable recordCv;
    std::thread recordThread;
    bool stopRequested = false;

    void ensureWorker() {
        if (recordThread.joinable()) {
            return;
        }
        stopRequested = false;
        recordThread = std::thread([this]() { recordLoop(); });
    }

    void stopWorker() {
        {
            std::lock_guard<std::mutex> lock(recordMutex);
            stopRequested = true;
        }
        recordCv.notify_one();
        if (recordThread.joinable()) {
            recordThread.join();
        }
        g_frameStreamRecordingQueueSize.store(0);
    }

    void recordLoop() {
        while (true) {
            PendingRecordFrame item;
            bool hasFrame = false;
            {
                std::unique_lock<std::mutex> lock(recordMutex);
                recordCv.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                    return stopRequested || !recordQueue.empty() || !g_rawRecordingRequested.load();
                });
                if (stopRequested && recordQueue.empty()) {
                    break;
                }
                if (!recordQueue.empty()) {
                    item = std::move(recordQueue.front());
                    recordQueue.pop_front();
                    g_frameStreamRecordingQueueSize.store(static_cast<int>(recordQueue.size()));
                    hasFrame = true;
                }
            }

            if (!g_rawRecordingRequested.load()) {
                closeWriter();
                continue;
            }
            if (hasFrame) {
                writeFrameSync(item.frame, item.frameId);
            }
        }
        closeWriter();
    }

    void writeFrameSync(const cv::Mat& frameBgr, int frameId) {
        if (!g_rawRecordingRequested.load()) {
            closeWriter();
            return;
        }
        if (frameBgr.empty()) {
            return;
        }

        const qint64 writeStartMs = qtMonotonicMs();
        if (!writer.isOpened()) {
            openWriter(frameBgr.size());
        } else if (frameBgr.size() != writerSize) {
            closeWriter();
            openWriter(frameBgr.size());
        }

        if (writer.isOpened()) {
            writer.write(frameBgr);
            ++framesWritten;
            g_frameStreamLastRecordFrameId.store(frameId);
            g_frameStreamRecordingWriteMs.store(static_cast<int>(qtMonotonicMs() - writeStartMs));
        }
    }

    void openWriter(const cv::Size& size) {
        const QString date = QDateTime::currentDateTime().toString("yyyyMMdd");
#ifdef Q_OS_WIN
        WindowsRecordingFilePlan plan = makeWindowsRecordingFilePlan("mp4");
        QString path = plan.path;
#else
        QString stamp = currentRawRecordingSessionId();
        if (stamp.isEmpty()) {
            stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        }
        QString path = makeUniquePath(rawVideoDirPath(date), stamp + "_camera_raw", "mp4", 1);
#endif
        const double fps = static_cast<double>(envIntBounded("ANTI_UAV_RAW_RECORD_FPS", 25, 1, 60));

        writerSize = size;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!path.isEmpty()) {
            writer.open(path.toStdString(), fourcc, fps, writerSize, true);
        }

        if (!writer.isOpened()) {
#ifdef Q_OS_WIN
            plan = makeWindowsRecordingFilePlan("avi", &plan);
            path = plan.path;
#else
            path = makeUniquePath(rawVideoDirPath(date), stamp + "_camera_raw", "avi", 1);
#endif
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            if (!path.isEmpty()) {
                writer.open(path.toStdString(), fourcc, fps, writerSize, true);
            }
        }

        if (writer.isOpened()) {
            activePath = path;
#ifdef Q_OS_WIN
            activeRecordId = plan.recordId;
            activeSessionId = plan.sessionId;
            activeSource = plan.source;
            activeStartTime = plan.startTime;
            activeCodec = (fourcc == cv::VideoWriter::fourcc('m', 'p', '4', 'v')) ? "mp4v" : "MJPG";
#else
            activeRecordId.clear();
            activeSessionId = currentCaptureSessionId();
            activeSource = "local";
            activeStartTime = QDateTime::currentDateTime();
            activeCodec = (fourcc == cv::VideoWriter::fourcc('m', 'p', '4', 'v')) ? "mp4v" : "MJPG";
#endif
            framesWritten = 0;
            setCurrentRawRecordingPath(path);
            g_rawRecordingActive.store(true);
            QJsonObject extra;
            extra["width"] = writerSize.width;
            extra["height"] = writerSize.height;
            extra["fps"] = fps;
            appendRecordingTimelineEvent("raw_video_first_frame", "qt_raw_video", path, extra);
#ifdef Q_OS_WIN
            appendWindowsRecordingIndexEvent("start",
                                             activeRecordId,
                                             activeSessionId,
                                             activeSource,
                                             activeStartTime,
                                             QDateTime(),
                                             path,
                                             activeCodec,
                                             writerSize,
                                             0,
                                             "");
#endif
            if (g_fusedRecordingRequested.load()) {
                writeCaptureSessionManifest("recording");
                writeCaptureSessionSummary("recording");
            }
            qInfo() << "[REC] raw camera recording started:" << path
                    << "size=" << writerSize.width << "x" << writerSize.height
                    << "fps=" << fps;
        } else {
            g_rawRecordingRequested.store(false);
            g_rawRecordingActive.store(false);
            setCurrentRawRecordingPath("");
            qWarning() << "[REC] failed to open raw camera recorder";
        }
    }

    void closeWriter() {
        if (writer.isOpened()) {
            const QString path = activePath.isEmpty() ? currentRawRecordingPath() : activePath;
            writer.release();
            appendRecordingTimelineEvent("raw_video_stop", "qt_raw_video", path);
#ifdef Q_OS_WIN
            appendWindowsRecordingIndexEvent("end",
                                             activeRecordId,
                                             activeSessionId,
                                             activeSource,
                                             activeStartTime,
                                             QDateTime::currentDateTime(),
                                             path,
                                             activeCodec,
                                             writerSize,
                                             framesWritten,
                                             "manual_stop");
#endif
            qInfo() << "[REC] raw camera recording stopped:" << path;
            if (!currentCaptureSessionId().isEmpty()) {
                const QString state = g_fusedRecordingRequested.load() ? "recording" : "stopped";
                writeCaptureSessionManifest(state);
                writeCaptureSessionSummary(state);
            }
        }
        g_rawRecordingActive.store(false);
        activePath.clear();
        activeRecordId.clear();
        activeSessionId.clear();
        activeSource.clear();
        activeCodec.clear();
        activeStartTime = QDateTime();
        framesWritten = 0;
    }
};

// ============================================================
// 8. RK 同帧视频流读取线程：显示 RK YOLO 实际处理的同一帧
// ============================================================

class RkFrameStreamReader {
public:
    bool run(QThread* owner) {
        const QString host = QString::fromStdString(
            envStringOrDefault("ANTI_UAV_RK_FRAME_STREAM_HOST", defaultBoardHostForThisNode()));
        const quint16 port = static_cast<quint16>(
            envIntBounded("ANTI_UAV_FRAME_STREAM_PORT", 5010, 1, 65535));
        const bool latestOnly = envIntBounded("ANTI_UAV_QT_LATEST_ONLY", 1, 0, 1) != 0;
        const int maxDrainFrames = envIntBounded("ANTI_UAV_QT_DRAIN_MAX_FRAMES", 16, 1, 120);

        qInfo() << "[RK_STREAM] connecting" << host << port
                << "latest_only=" << latestOnly
                << "record_decouple=" << (envIntBounded("ANTI_UAV_QT_RECORD_DECOUPLE", 1, 0, 1) != 0);
        MetricsLogger::instance().logFrameStreamEvent("connect_attempt", host, port,
                                                       -1, 0, 0, 0, 0, "pending", "");

        QTcpSocket socket;
        socket.setProxy(QNetworkProxy::NoProxy);
        socket.setSocketOption(QAbstractSocket::LowDelayOption, 1);
        socket.setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption, 262144);
        socket.setReadBufferSize(524288);
        socket.connectToHost(host, port);
        if (!socket.waitForConnected(1500)) {
            g_hasVideoSignal.store(false);
            g_videoUsingRkStream.store(false);
            qWarning() << "[RK_STREAM] connect failed:" << socket.errorString();
            MetricsLogger::instance().logFrameStreamEvent("connect_failed", host, port,
                                                           -1, 0, 0, 0, 0, "failed", socket.errorString());
            return false;
        }

        g_videoUsingRkStream.store(true);
        g_preferFrameStreamMeta.store(true);
        qInfo() << "[RK_STREAM] connected";
        MetricsLogger::instance().logFrameStreamEvent("connected", host, port,
                                                       -1, 0, 0, 0, 0, "ok", "");

        RawCameraRecorder rawRecorder;
        qint64 lastDisplayPublishMs = 0;
        qint64 lastFrameStreamLogMs = 0;
        while (!owner->isInterruptionRequested() && socket.state() == QAbstractSocket::ConnectedState) {
            QByteArray header;
            if (!readHeader(socket, header, owner)) {
                break;
            }

            int jsonLen = 0;
            int jpgLen = 0;
            QString headerError;
            if (!parseFrameHeader(header, jsonLen, jpgLen, headerError)) {
                qWarning() << "[RK_STREAM] invalid header:" << header.left(80);
                MetricsLogger::instance().logFrameStreamEvent("invalid_header", host, port,
                                                               -1, 0, 0, 0, 0, "bad_header",
                                                               headerError);
                break;
            }

            QByteArray jsonBytes;
            QByteArray jpgBytes;
            if (!readExact(socket, jsonBytes, jsonLen, owner) ||
                !readExact(socket, jpgBytes, jpgLen, owner)) {
                MetricsLogger::instance().logFrameStreamEvent("read_failed", host, port,
                                                               -1, jsonLen, jpgLen, 0, 0, "read_failed",
                                                               socket.errorString());
                break;
            }

            int drainedFrames = 0;
            if (latestOnly) {
                QByteArray nextHeader;
                QByteArray nextJsonBytes;
                QByteArray nextJpgBytes;
                int nextJsonLen = 0;
                int nextJpgLen = 0;
                while (drainedFrames < maxDrainFrames &&
                       tryReadBufferedPacket(socket, nextHeader, nextJsonBytes, nextJpgBytes,
                                             nextJsonLen, nextJpgLen)) {
                    header = std::move(nextHeader);
                    jsonBytes = std::move(nextJsonBytes);
                    jpgBytes = std::move(nextJpgBytes);
                    jsonLen = nextJsonLen;
                    jpgLen = nextJpgLen;
                    ++drainedFrames;
                }
                if (drainedFrames > 0) {
                    g_frameStreamDroppedOldFrames.fetch_add(drainedFrames);
                }
            }
            const qint64 qtReceiveDoneMonoMs = qtMonotonicMs();

            QJsonParseError jsonErr;
            QJsonDocument doc = QJsonDocument::fromJson(jsonBytes, &jsonErr);
            if (!doc.isObject()) {
                qWarning() << "[RK_STREAM] invalid json:" << jsonErr.errorString();
                MetricsLogger::instance().logFrameStreamEvent("invalid_json", host, port,
                                                               -1, jsonLen, jpgLen, 0, 0, "bad_json",
                                                               jsonErr.errorString());
                continue;
            }

            const qint64 recvMs = metricsNowMs();
            const qint64 pendingBytes = socket.bytesAvailable();
            const int backlogBytes = static_cast<int>(std::min<qint64>(pendingBytes, 2147483647));
            g_frameStreamBacklogBytes.store(backlogBytes);
            QJsonObject obj = doc.object();

            const qint64 qtDecodeStartMonoMs = qtMonotonicMs();
            std::vector<uchar> jpgData(static_cast<size_t>(jpgBytes.size()));
            std::memcpy(jpgData.data(), jpgBytes.constData(), static_cast<size_t>(jpgBytes.size()));
            cv::Mat frameBgr = cv::imdecode(jpgData, cv::IMREAD_COLOR);
            if (frameBgr.empty()) {
                qWarning() << "[RK_STREAM] JPEG decode failed";
                MetricsLogger::instance().logFrameStreamEvent("jpeg_decode_failed", host, port,
                                                               -1, jsonLen, jpgLen, 0, 0, "bad_jpeg", "");
                continue;
            }
            const qint64 qtDecodeEndMonoMs = qtMonotonicMs();

            if (ROTATE_IMAGE_180) {
                cv::rotate(frameBgr, frameBgr, cv::ROTATE_180);
            }
            const int frameId = metricsJsonIntAny(obj, {"frame_id", "frame", "id"}, -1);
            rawRecorder.writeFrame(frameBgr, frameId);
            g_frameStreamDecodedFrames.fetch_add(1);

            updateVisionDashboardState(obj, recvMs);
            MetricsLogger::instance().logSystemFromJson(obj, recvMs);
            if (recvMs - lastFrameStreamLogMs >= 1000) {
                lastFrameStreamLogMs = recvMs;
                MetricsLogger::instance().logFrameStreamEvent("frame", host, port,
                                                               metricsJsonIntAny(obj, {"frame_id", "frame", "id"}, -1),
                                                               jsonLen, jpgLen, frameBgr.cols, frameBgr.rows, "ok", "");
            }

            g_videoWidth.store(frameBgr.cols);
            g_videoHeight.store(frameBgr.rows);
            g_hasVideoSignal.store(true);
            g_lastVideoFrameMs.store(recvMs);
            if (shouldPublishDisplayFrame(recvMs, lastDisplayPublishMs)) {
                auto rgbFrame = std::make_shared<cv::Mat>();
                cv::cvtColor(frameBgr, *rgbFrame, cv::COLOR_BGR2RGB);
                const qint64 qtPublishMonoMs = qtMonotonicMs();
                auto timing = std::make_shared<VideoFrameTiming>();
                timing->frame_id = frameId;
                timing->camera_estimated_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"camera_estimated_mono_ms"}, -1.0));
                timing->rtsp_grab_start_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"rtsp_grab_start_mono_ms"}, -1.0));
                timing->rtsp_grab_done_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"rtsp_grab_done_mono_ms"}, -1.0));
                timing->decode_done_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"decode_done_mono_ms"}, -1.0));
                timing->infer_start_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"infer_start_mono_ms"}, -1.0));
                timing->infer_end_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"infer_end_mono_ms"}, -1.0));
                timing->postprocess_done_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"postprocess_done_mono_ms"}, -1.0));
                timing->stream_enqueue_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"stream_enqueue_mono_ms"}, -1.0));
                timing->jpeg_start_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"jpeg_start_mono_ms"}, -1.0));
                timing->jpeg_end_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"jpeg_end_mono_ms"}, -1.0));
                timing->tcp_send_start_mono_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"tcp_send_start_mono_ms"}, -1.0));
                timing->yolo_stream_send_ts_ms = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"yolo_stream_send_ts_ms", "send_ts_ms", "tcp_send_start_mono_ms"}, -1.0));
                timing->payload_size = static_cast<qint64>(
                    metricsJsonDoubleAny(obj, {"payload_size", "jpeg_payload_size"}, jpgBytes.size()));
                timing->qt_receive_done_mono_ms = qtReceiveDoneMonoMs;
                timing->qt_decode_start_mono_ms = qtDecodeStartMonoMs;
                timing->qt_decode_end_mono_ms = qtDecodeEndMonoMs;
                timing->qt_publish_mono_ms = qtPublishMonoMs;
                timing->display_frame_id = frameId;
                timing->record_frame_id = g_frameStreamLastRecordFrameId.load();
                timing->display_dropped_old_frames = g_frameStreamDroppedOldFrames.load();
                timing->recording_queue_size = g_frameStreamRecordingQueueSize.load();
                timing->recording_dropped_frames = g_frameStreamRecordingDroppedFrames.load();
                timing->recording_write_ms = g_frameStreamRecordingWriteMs.load();
                std::atomic_store(&g_latestVideoFrame, rgbFrame);
                std::atomic_store(&g_latestVideoTiming, timing);
                g_frameStreamLastDisplayFrameId.store(frameId);
                g_frameStreamPublishedFrames.fetch_add(1);
            }
            publishVisionMetaFromJson(obj, recvMs);
        }

        g_hasVideoSignal.store(false);
        g_videoUsingRkStream.store(false);
        qWarning() << "[RK_STREAM] disconnected";
        MetricsLogger::instance().logFrameStreamEvent("disconnected", host, port,
                                                       -1, 0, 0, 0, 0, "closed", socket.errorString());
        return false;
    }

private:
    bool parseFrameHeader(const QByteArray& header, int& jsonLen, int& jpgLen, QString& errorText) const {
        const QList<QByteArray> parts = header.trimmed().split(' ');
        if (parts.size() != 3 || parts.at(0) != "AUVF") {
            errorText = QString::fromUtf8(header.left(80));
            return false;
        }

        bool okJson = false;
        bool okJpg = false;
        jsonLen = parts.at(1).toInt(&okJson);
        jpgLen = parts.at(2).toInt(&okJpg);
        if (!okJson || !okJpg || jsonLen <= 0 || jsonLen > 65536 ||
            jpgLen <= 0 || jpgLen > 4 * 1024 * 1024) {
            errorText = QString("json=%1 jpg=%2").arg(jsonLen).arg(jpgLen);
            return false;
        }
        return true;
    }

    bool tryReadBufferedPacket(QTcpSocket& socket,
                               QByteArray& header,
                               QByteArray& jsonBytes,
                               QByteArray& jpgBytes,
                               int& jsonLen,
                               int& jpgLen) const {
        const qint64 available = socket.bytesAvailable();
        if (available <= 0) {
            return false;
        }
        const QByteArray buffered = socket.peek(available);
        const int headerEnd = buffered.indexOf('\n');
        if (headerEnd < 0) {
            return false;
        }

        QString errorText;
        const QByteArray nextHeader = buffered.left(headerEnd + 1);
        int nextJsonLen = 0;
        int nextJpgLen = 0;
        if (!parseFrameHeader(nextHeader, nextJsonLen, nextJpgLen, errorText)) {
            return false;
        }

        const qint64 packetSize = static_cast<qint64>(headerEnd + 1) +
                                  static_cast<qint64>(nextJsonLen) +
                                  static_cast<qint64>(nextJpgLen);
        if (available < packetSize) {
            return false;
        }

        header = socket.read(headerEnd + 1);
        jsonBytes = socket.read(nextJsonLen);
        jpgBytes = socket.read(nextJpgLen);
        if (header.size() != headerEnd + 1 ||
            jsonBytes.size() != nextJsonLen ||
            jpgBytes.size() != nextJpgLen) {
            return false;
        }
        jsonLen = nextJsonLen;
        jpgLen = nextJpgLen;
        return true;
    }

    bool readHeader(QTcpSocket& socket, QByteArray& out, QThread* owner) const {
        while (!owner->isInterruptionRequested()) {
            if (socket.canReadLine()) {
                out = socket.readLine();
                return !out.isEmpty();
            }
            if (!socket.waitForReadyRead(1000)) {
                if (socket.state() != QAbstractSocket::ConnectedState) return false;
            }
        }
        return false;
    }

    bool readExact(QTcpSocket& socket, QByteArray& out, int size, QThread* owner) const {
        out.clear();
        out.reserve(size);
        while (!owner->isInterruptionRequested() && out.size() < size) {
            if (socket.bytesAvailable() <= 0 && !socket.waitForReadyRead(1000)) {
                if (socket.state() != QAbstractSocket::ConnectedState) return false;
                continue;
            }

            const qint64 need = size - out.size();
            QByteArray chunk = socket.read(need);
            if (chunk.isEmpty()) {
                if (socket.state() != QAbstractSocket::ConnectedState) return false;
                continue;
            }
            out.append(chunk);
        }
        return out.size() == size;
    }
};

// ============================================================
// 9. 视频读取线程：默认 RK 同帧流，可显式回退海康 RTSP
// ============================================================

class VideoReaderWorker : public QThread {
protected:
    void run() override {
        QString source = QString::fromUtf8(qgetenv("ANTI_UAV_QT_VIDEO_SOURCE")).trimmed().toLower();
        if (source.isEmpty()) source = defaultVideoSourceForThisNode();
        const bool autoSource = source.isEmpty() || source == "auto";
        const bool useRkStream = autoSource || source == "rk_stream" || source == "rk" || source == "frame_stream";
        g_expectRkFrameStream.store(useRkStream);
        g_preferFrameStreamMeta.store(useRkStream);

        if (useRkStream) {
            RkFrameStreamReader reader;
            int streamFailCount = 0;
            const int autoFallbackFails = envIntBounded("ANTI_UAV_RK_STREAM_AUTO_FAILS", 10, 1, 120);
            while (!isInterruptionRequested()) {
                const bool streamEnded = reader.run(this);
                Q_UNUSED(streamEnded);
                streamFailCount++;
                if (autoSource && streamFailCount >= autoFallbackFails) {
                    qWarning() << "[RK_STREAM] auto fallback to RTSP after failures:" << streamFailCount;
                    g_expectRkFrameStream.store(false);
                    g_preferFrameStreamMeta.store(false);
                    g_videoUsingRkStream.store(false);
                    break;
                }
                if (!isInterruptionRequested()) {
                    QThread::msleep(500);
                }
            }

            if (!autoSource || isInterruptionRequested()) {
                g_recordingRequested.store(false);
                g_recordingActive.store(false);
                setCurrentRecordingPath("");
                g_rawRecordingRequested.store(false);
                g_rawRecordingActive.store(false);
                setCurrentRawRecordingPath("");
                g_audioRawRecordingRequested.store(false);
                g_audioRawRecordingActive.store(false);
                setCurrentAudioRawRecordingPath("");
                g_fusedRecordingRequested.store(false);
                g_fusedRecordingActive.store(false);
                setCurrentCaptureSession("", "");
                g_hasVideoSignal.store(false);
                g_videoUsingRkStream.store(false);
                return;
            }
        }

        const std::string rtspUrl = envStringOrDefault(
            "ANTI_UAV_QT_RTSP_URL",
            envStringOrDefault("ANTI_UAV_RTSP_URL", HIKVISION_RTSP_URL));
        QByteArray primaryTransport = qgetenv("ANTI_UAV_QT_RTSP_TRANSPORT").trimmed().toLower();
        if (primaryTransport.isEmpty()) {
            primaryTransport = "udp";
        }
        if (rtspUrl.empty()) {
            qWarning() << "[VIDEO] RTSP URL is empty; set ANTI_UAV_QT_RTSP_URL or ANTI_UAV_RTSP_URL";
            while (!isInterruptionRequested()) {
                g_hasVideoSignal.store(false);
                QThread::sleep(1);
            }
            return;
        }

        RawCameraRecorder rawRecorder;
        qint64 lastDisplayPublishMs = 0;
        while (!isInterruptionRequested()) {
            qInfo() << "[VIDEO] opening Hikvision RTSP:"
                    << QString::fromStdString(rtspUrl)
                    << "transport=" << primaryTransport;

            cv::VideoCapture cap;
            QByteArray activeTransport;
            std::vector<QByteArray> transports{primaryTransport};
            if (primaryTransport != "tcp") {
                transports.push_back("tcp");
            }
            if (primaryTransport != "udp") {
                transports.push_back("udp");
            }

            for (const QByteArray& transport : transports) {
                if (openRtspCapture(cap, rtspUrl, transport)) {
                    activeTransport = transport;
                    break;
                }
            }

            if (!cap.isOpened()) {
                g_hasVideoSignal.store(false);
                qWarning() << "[VIDEO] RTSP open failed, retry after 1s";
                QThread::sleep(1);
                continue;
            }

            qInfo() << "[VIDEO] RTSP opened, transport=" << activeTransport;

            while (!isInterruptionRequested()) {
                cv::Mat frameBgr;
                bool ok = cap.read(frameBgr);
                if (!ok || frameBgr.empty()) {
                    g_hasVideoSignal.store(false);
                    qWarning() << "[VIDEO] frame read failed, reconnecting";
                    if (activeTransport == "udp") {
                        primaryTransport = "tcp";
                        qWarning() << "[VIDEO] UDP opened but produced no frame, fallback to TCP on next reconnect";
                    }
                    break;
                }

                if (ROTATE_IMAGE_180) {
                    cv::rotate(frameBgr, frameBgr, cv::ROTATE_180);
                }
                rawRecorder.writeFrame(frameBgr);

                g_videoWidth.store(frameBgr.cols);
                g_videoHeight.store(frameBgr.rows);
                g_hasVideoSignal.store(true);
                const qint64 recvMs = metricsNowMs();
                g_lastVideoFrameMs.store(recvMs);

                if (shouldPublishDisplayFrame(recvMs, lastDisplayPublishMs)) {
                    auto rgbFrame = std::make_shared<cv::Mat>();
                    cv::cvtColor(frameBgr, *rgbFrame, cv::COLOR_BGR2RGB);
                    std::atomic_store(&g_latestVideoFrame, rgbFrame);
                }

                // 不主动 sleep，跟随 RTSP 原始帧率。
            }

            cap.release();
            QThread::msleep(300);
        }

        g_recordingRequested.store(false);
        g_recordingActive.store(false);
        setCurrentRecordingPath("");
        g_rawRecordingRequested.store(false);
        g_rawRecordingActive.store(false);
        setCurrentRawRecordingPath("");
        g_audioRawRecordingRequested.store(false);
        g_audioRawRecordingActive.store(false);
        setCurrentAudioRawRecordingPath("");
        g_fusedRecordingRequested.store(false);
        g_fusedRecordingActive.store(false);
        setCurrentCaptureSession("", "");
        g_hasVideoSignal.store(false);
    }

private:
    QByteArray ffmpegOptionsForTransport(const QByteArray& transport) const {
        QByteArray options;
        if (transport == "tcp") {
            options = "rtsp_transport;tcp|rtsp_flags;prefer_tcp";
        } else {
            options = "rtsp_transport;udp";
        }
        options += "|fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32768|max_delay;0|reorder_queue_size;0";
        return options;
    }

    bool openRtspCapture(cv::VideoCapture& cap,
                         const std::string& rtspUrl,
                         const QByteArray& transport) const {
        qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", ffmpegOptionsForTransport(transport));

        cap.release();
        cap = cv::VideoCapture();
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        qInfo() << "[VIDEO] trying RTSP transport=" << transport
                << "options=" << qgetenv("OPENCV_FFMPEG_CAPTURE_OPTIONS");
        const bool ok = cap.open(rtspUrl, cv::CAP_FFMPEG);
        if (ok) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
        return ok;
    }
};

// ============================================================
// 7. 主程序
// ============================================================

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
#ifndef Q_OS_WIN
    QFont::insertSubstitution("Microsoft YaHei UI", "Noto Sans SC");
    QFont::insertSubstitution("Segoe UI", "Noto Sans SC");
#endif
    QNetworkProxyFactory::setUseSystemConfiguration(false);
    QNetworkProxy::setApplicationProxy(QNetworkProxy::NoProxy);
    g_displayEcoMode.store(envIntBounded("ANTI_UAV_QT_ECO_MODE", 1, 0, 1) != 0);
    g_visionOverlayMaxAgeMs.store(envIntBounded("ANTI_UAV_QT_BOX_MAX_AGE_MS", 250, 80, 1000));
    const QByteArray videoSource = qgetenv("ANTI_UAV_QT_VIDEO_SOURCE").trimmed().isEmpty()
        ? defaultVideoSourceForThisNode().toUtf8()
        : qgetenv("ANTI_UAV_QT_VIDEO_SOURCE").trimmed();
    qInfo() << "[SYNC] realtime overlay max age ms=" << g_visionOverlayMaxAgeMs.load()
            << "video_source=" << videoSource
            << "rk_stream=" << qgetenv("ANTI_UAV_RK_FRAME_STREAM_HOST").trimmed()
            << ":" << qgetenv("ANTI_UAV_FRAME_STREAM_PORT").trimmed()
            << "rtsp_transport="
            << (qgetenv("ANTI_UAV_QT_RTSP_TRANSPORT").trimmed().isEmpty()
                    ? QByteArray("udp")
                    : qgetenv("ANTI_UAV_QT_RTSP_TRANSPORT").trimmed());
    qInfo() << "[EFFECTIVE_CONFIG] ANTI_UAV_QT_ECO_MODE env="
            << (qgetenv("ANTI_UAV_QT_ECO_MODE").trimmed().isEmpty()
                    ? QByteArray("<unset; code_default=1>")
                    : qgetenv("ANTI_UAV_QT_ECO_MODE").trimmed())
            << "final=" << (g_displayEcoMode.load() ? 1 : 0)
            << "ANTI_UAV_QT_DISPLAY_INTERVAL_MS env="
            << (qgetenv("ANTI_UAV_QT_DISPLAY_INTERVAL_MS").trimmed().isEmpty()
                    ? QByteArray("<unset; code_default=40>")
                    : qgetenv("ANTI_UAV_QT_DISPLAY_INTERVAL_MS").trimmed())
            << "effective_interval_ms=" << displayFrameIntervalMs()
            << "ANTI_UAV_QT_LATEST_ONLY env="
            << (qgetenv("ANTI_UAV_QT_LATEST_ONLY").trimmed().isEmpty()
                    ? QByteArray("<unset; code_default=1>")
                    : qgetenv("ANTI_UAV_QT_LATEST_ONLY").trimmed())
            << "ANTI_UAV_QT_RECORD_DECOUPLE env="
            << (qgetenv("ANTI_UAV_QT_RECORD_DECOUPLE").trimmed().isEmpty()
                    ? QByteArray("<unset; code_default=1>")
                    : qgetenv("ANTI_UAV_QT_RECORD_DECOUPLE").trimmed())
            << "ANTI_UAV_QT_LATENCY_OVERLAY env="
            << (qgetenv("ANTI_UAV_QT_LATENCY_OVERLAY").trimmed().isEmpty()
                    ? QByteArray("<unset; code_default=1>")
                    : qgetenv("ANTI_UAV_QT_LATENCY_OVERLAY").trimmed());

    DashboardWindow mainWindow;
    mainWindow.setWindowTitle("AntiUAV 无人机检测系统");
    mainWindow.resize(1500, 900);
    mainWindow.setStyleSheet(R"(
        QWidget {
            background-color: #080B0F;
            color: #E5E7EB;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            letter-spacing: 0px;
        }
        #PanelHeader, #PanelSection, #PersistentCaptureBar, #RemoteIdPanel,
        #TopStatusStrip, #ModuleDetailPage {
            background-color: #111820;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
        }
        #PersistentCaptureBar {
            border-left: 3px solid #EF4444;
        }
        #PersistentRecordButton {
            min-height: 28px;
            padding: 3px 7px;
            font-size: 12px;
            border-color: rgba(248, 113, 113, 0.55);
        }
        #PageNavButton {
            min-height: 32px;
            padding: 3px 6px;
            font-size: 12px;
            border-color: rgba(103, 232, 249, 0.35);
        }
        #PageNavButton:checked {
            color: #071014;
            background-color: #67E8F9;
            border-color: #A5F3FC;
        }
        #OverviewCard {
            min-height: 47px;
            padding: 3px 7px;
            font-size: 12px;
            text-align: left;
            background-color: #151F29;
        }
        #OverviewCard:pressed {
            background-color: #0E7490;
        }
        #PtzTouchPanel {
            background-color: #101820;
            border: 1px solid rgba(34, 211, 238, 0.45);
            border-radius: 6px;
        }
        #PtzState {
            color: #67E8F9;
            font-size: 13px;
            font-weight: 800;
        }
        #PtzDirectionButton, #PtzStopButton {
            min-width: 50px;
            min-height: 44px;
            padding: 2px;
            font-size: 22px;
            background-color: #18232D;
        }
        #PtzDirectionButton:pressed {
            background-color: #0E7490;
            border-color: #67E8F9;
        }
        #PtzStopButton {
            color: #FBBF24;
            border-color: rgba(251, 191, 36, 0.75);
        }
        #PtzActionButton {
            min-height: 34px;
            padding: 3px 7px;
            font-size: 12px;
        }
        #PtzActionButton:checked {
            color: #071014;
            background-color: #67E8F9;
            border-color: #A5F3FC;
        }
        #EmergencyStopButton {
            min-height: 38px;
            color: white;
            background-color: #B91C1C;
            border: 2px solid #FCA5A5;
            font-size: 15px;
            font-weight: 900;
        }
        #EmergencyStopButton:pressed {
            background-color: #7F1D1D;
        }
        #PanelHeader {
            border-left: 3px solid #22D3EE;
        }
        #PanelTitle {
            color: #F8FAFC;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        }
        #PanelSubtitle {
            color: #94A3B8;
            font-size: 13px;
            background: transparent;
        }
        #PanelClock {
            color: #67E8F9;
            font-size: 13px;
            background: transparent;
        }
        #SectionTitle {
            color: #CBD5E1;
            font-size: 13px;
            font-weight: 800;
            background: transparent;
        }
        #MetricName {
            color: #94A3B8;
            font-size: 13px;
            background: transparent;
        }
        #MetricValue {
            color: #F8FAFC;
            font-size: 14px;
            font-weight: 700;
            background: transparent;
        }
        #PanelHint {
            color: #94A3B8;
            font-size: 12px;
            background: transparent;
        }
        QPushButton {
            background-color: #1F2933;
            color: #F8FAFC;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 6px;
            padding: 6px 9px;
            font-weight: 800;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #2B3947;
            border-color: rgba(103, 232, 249, 0.70);
        }
        QComboBox, QLineEdit {
            background-color: #0F1720;
            color: #F8FAFC;
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 5px;
            padding: 5px 8px;
            font-size: 13px;
        }
        QComboBox:hover, QLineEdit:focus {
            border-color: rgba(103, 232, 249, 0.68);
        }
        QScrollArea {
            background: transparent;
            border: none;
        }
        QScrollArea > QWidget > QWidget {
            background: transparent;
        }
        QTableWidget {
            background-color: #0B1118;
            alternate-background-color: #121B24;
            border: 1px solid rgba(148, 163, 184, 0.24);
            gridline-color: rgba(148, 163, 184, 0.18);
            font-size: 11px;
        }
        QHeaderView::section {
            background-color: #18232D;
            color: #CBD5E1;
            border: none;
            border-right: 1px solid rgba(148, 163, 184, 0.20);
            padding: 6px;
            font-weight: 800;
        }
        #RecordButton {
            border-color: rgba(248, 113, 113, 0.60);
        }
        #ControlButton {
            border-color: rgba(103, 232, 249, 0.55);
        }
        #RestoreButton {
            color: #D1FAE5;
            border-color: rgba(52, 211, 153, 0.70);
            background-color: rgba(6, 78, 59, 0.30);
        }
        #StopButton {
            border-color: rgba(251, 191, 36, 0.55);
        }
        QProgressBar {
            background-color: rgba(15, 23, 32, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 4px;
            color: #CBD5E1;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
        }
        QProgressBar::chunk {
            background-color: #22D3EE;
            border-radius: 3px;
        }
    )");

    QHBoxLayout* layout = new QHBoxLayout(&mainWindow);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(8);

    const QSize screenSize = QGuiApplication::primaryScreen()
        ? QGuiApplication::primaryScreen()->size()
        : QSize(1500, 900);
    const bool compactMipi = screenSize.width() <= 1100 || screenSize.height() <= 700;

    VideoWidget* videoWidget = new VideoWidget();
    videoWidget->setMinimumSize(compactMipi ? QSize(520, 320) : QSize(860, 540));
    videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QWidget* rightColumn = new QWidget();
    rightColumn->setObjectName("RightColumn");
    rightColumn->setMinimumWidth(compactMipi ? 340 : 430);
    rightColumn->setMaximumWidth(compactMipi ? 370 : 490);
    rightColumn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    QVBoxLayout* rightLayout = new QVBoxLayout(rightColumn);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(5);

    QScrollArea* sideScroll = new QScrollArea();
    DashboardSidePanel* sidePanel = new DashboardSidePanel(&mainWindow, sideScroll);
    PtzTouchPanel* ptzTouchPanel = new PtzTouchPanel(&mainWindow, rightColumn);
    RemoteIdPanel* remoteIdPanel = new RemoteIdPanel(rightColumn);
    PersistentCaptureBar* captureBar = new PersistentCaptureBar(&mainWindow, rightColumn);
    TopStatusStrip* topStatusStrip = new TopStatusStrip(rightColumn);
    sideScroll->setWidgetResizable(true);
    sideScroll->setFrameShape(QFrame::NoFrame);
    sideScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    sideScroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sideScroll->setWidget(sidePanel);

    RadarWidget* radarWidget = new RadarWidget();
    radarWidget->setMinimumHeight(330);

    QFrame* navFrame = new QFrame();
    QGridLayout* navLayout = new QGridLayout(navFrame);
    navLayout->setContentsMargins(0, 0, 0, 0);
    navLayout->setSpacing(4);
    const QStringList navNames = {
        "概览", "视频与跟踪", "手动控制", "音频",
        "Remote ID", "录像", "系统设置"
    };
    QList<QPushButton*> navButtons;
    for (int i = 0; i < navNames.size(); ++i) {
        QPushButton* button = new QPushButton(navNames.at(i));
        button->setObjectName("PageNavButton");
        button->setCheckable(true);
        button->setAutoExclusive(true);
        button->setMinimumHeight(34);
        navLayout->addWidget(button, i / 4, i % 4);
        navButtons.append(button);
    }

    QStackedWidget* pageStack = new QStackedWidget();
    OverviewCardsPage* overviewPage = new OverviewCardsPage(
        [pageStack, navButtons](int index) {
            if (index < 0 || index >= navButtons.size()) return;
            pageStack->setCurrentIndex(index);
            navButtons.at(index)->setChecked(true);
        },
        rightColumn);
    ModuleDetailPage* videoPage =
        new ModuleDetailPage(DetailPageKind::Video, &mainWindow, rightColumn);
    ModuleDetailPage* audioPage =
        new ModuleDetailPage(DetailPageKind::Audio, &mainWindow, rightColumn);
    ModuleDetailPage* recordingPage =
        new ModuleDetailPage(DetailPageKind::Recording, &mainWindow, rightColumn);
    ModuleDetailPage* systemPage =
        new ModuleDetailPage(DetailPageKind::System, &mainWindow, rightColumn);
    pageStack->addWidget(overviewPage);
    pageStack->addWidget(videoPage);
    pageStack->addWidget(ptzTouchPanel);
    pageStack->addWidget(audioPage);
    pageStack->addWidget(remoteIdPanel);
    pageStack->addWidget(recordingPage);
    pageStack->addWidget(systemPage);
    navButtons.at(0)->setChecked(true);
    for (int i = 0; i < navButtons.size(); ++i) {
        QObject::connect(navButtons.at(i), &QPushButton::clicked,
                         pageStack, [pageStack, i]() { pageStack->setCurrentIndex(i); });
    }
    QObject::connect(pageStack, &QStackedWidget::currentChanged,
                     &mainWindow, [&mainWindow](int index) {
                         if (index != 2) {
                             mainWindow.requestTouchStop("page_switch");
                         }
                     });
    const QString startPage =
        QString::fromUtf8(qgetenv("ANTI_UAV_QT_START_PAGE")).trimmed().toLower();
    int startIndex = 0;
    if (startPage == "video" || startPage == "tracking") startIndex = 1;
    else if (startPage == "manual" || startPage == "ptz") startIndex = 2;
    else if (startPage == "audio") startIndex = 3;
    else if (startPage == "remoteid" || startPage == "rid900") startIndex = 4;
    else if (startPage == "recording") startIndex = 5;
    else if (startPage == "settings" || startPage == "system") startIndex = 6;
    if (startIndex > 0) {
        navButtons.at(startIndex)->setChecked(true);
        pageStack->setCurrentIndex(startIndex);
    }

    sidePanel->hide();
    sideScroll->hide();
    radarWidget->hide();
    rightLayout->addWidget(topStatusStrip, 0);
    rightLayout->addWidget(captureBar, 0);
    rightLayout->addWidget(pageStack, 1);
    rightLayout->addWidget(navFrame, 0);

    layout->addWidget(videoWidget, 1);
    layout->addWidget(rightColumn, 0);
    QObject::connect(videoWidget, &VideoWidget::videoPointClicked,
                     ptzTouchPanel, &PtzTouchPanel::handleVideoClick);

    QThread* networkThread = new QThread(&mainWindow);
    UdpReceiverWorker* udpWorker = new UdpReceiverWorker();
    udpWorker->moveToThread(networkThread);
    QObject::connect(networkThread, &QThread::started,
                     udpWorker, &UdpReceiverWorker::start);
    QObject::connect(networkThread, &QThread::finished,
                     udpWorker, &QObject::deleteLater);
    networkThread->start(QThread::HighPriority);

    QThread* audioNetworkThread = new QThread(&mainWindow);
    AudioReceiverWorker* audioWorker = new AudioReceiverWorker();
    audioWorker->moveToThread(audioNetworkThread);
    QObject::connect(audioNetworkThread, &QThread::started,
                     audioWorker, &AudioReceiverWorker::start);
    QObject::connect(audioNetworkThread, &QThread::finished,
                     audioWorker, &QObject::deleteLater);
    audioNetworkThread->start(QThread::HighPriority);

    QThread* fusionNetworkThread = new QThread(&mainWindow);
    FusionReceiverWorker* fusionWorker = new FusionReceiverWorker();
    fusionWorker->moveToThread(fusionNetworkThread);
    QObject::connect(fusionNetworkThread, &QThread::started,
                     fusionWorker, &FusionReceiverWorker::start);
    QObject::connect(fusionNetworkThread, &QThread::finished,
                     fusionWorker, &QObject::deleteLater);
    fusionNetworkThread->start(QThread::HighPriority);

    QThread* ridNetworkThread = new QThread(&mainWindow);
    RidReceiverWorker* ridWorker = new RidReceiverWorker();
    ridWorker->moveToThread(ridNetworkThread);
    QObject::connect(ridNetworkThread, &QThread::started,
                     ridWorker, &RidReceiverWorker::start);
    QObject::connect(ridNetworkThread, &QThread::finished,
                     ridWorker, &QObject::deleteLater);
    ridNetworkThread->start(QThread::HighPriority);

    QThread* syncNetworkThread = new QThread(&mainWindow);
    SyncWorker* syncWorker = new SyncWorker();
    syncWorker->moveToThread(syncNetworkThread);
    QObject::connect(syncNetworkThread, &QThread::started,
                     syncWorker, &SyncWorker::start);
    QObject::connect(syncNetworkThread, &QThread::finished,
                     syncWorker, &QObject::deleteLater);
    syncNetworkThread->start(QThread::HighPriority);

    VideoReaderWorker* videoThread = new VideoReaderWorker();
    videoThread->start(QThread::TimeCriticalPriority);

    SystemMetricsWorker* metricsThread = new SystemMetricsWorker();
    metricsThread->start();

    WindowRecorder* windowRecorder = new WindowRecorder(&mainWindow, &mainWindow);

    const QString validationWindowSize =
        QString::fromUtf8(qgetenv("ANTI_UAV_QT_WINDOW_SIZE")).trimmed().toLower();
    const QStringList validationSizeParts = validationWindowSize.split('x');
    if (validationSizeParts.size() == 2) {
        bool widthOk = false;
        bool heightOk = false;
        const int validationWidth = validationSizeParts.at(0).toInt(&widthOk);
        const int validationHeight = validationSizeParts.at(1).toInt(&heightOk);
        if (widthOk && heightOk && validationWidth >= 640 && validationHeight >= 480) {
            mainWindow.resize(validationWidth, validationHeight);
            mainWindow.show();
        } else {
            mainWindow.showMaximized();
        }
    } else {
        mainWindow.showMaximized();
    }
    mainWindow.raise();
    mainWindow.activateWindow();
    mainWindow.setFocus();
    const QString screenshotPath =
        QString::fromUtf8(qgetenv("ANTI_UAV_QT_SCREENSHOT_PATH")).trimmed();
    if (!screenshotPath.isEmpty()) {
        QTimer::singleShot(1800, &mainWindow, [&mainWindow, screenshotPath]() {
            const bool ok = mainWindow.grab().save(screenshotPath);
            qInfo() << "[UI_SCREENSHOT]" << (ok ? "saved" : "failed") << screenshotPath;
            if (qEnvironmentVariableIntValue("ANTI_UAV_QT_EXIT_AFTER_SCREENSHOT") == 1) {
                QTimer::singleShot(200, QCoreApplication::instance(), &QCoreApplication::quit);
            }
        });
    }
    int ret = a.exec();

    delete windowRecorder;

    metricsThread->requestInterruption();
    metricsThread->wait();
    delete metricsThread;

    videoThread->requestInterruption();
    videoThread->wait();
    delete videoThread;

    networkThread->quit();
    networkThread->wait();

    audioNetworkThread->quit();
    audioNetworkThread->wait();

    fusionNetworkThread->quit();
    fusionNetworkThread->wait();

    ridNetworkThread->quit();
    ridNetworkThread->wait();

    syncNetworkThread->quit();
    syncNetworkThread->wait();

    MetricsLogger::instance().finalize();

    return ret;
}

// main.cpp 内包含 QObject 子类，必须保留。
#include "main.moc"
