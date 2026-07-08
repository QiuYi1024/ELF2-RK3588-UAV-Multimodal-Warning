#pragma once

#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>
#include <QtGlobal>

#define META_BUFFER_SIZE 30

// 视觉检测元数据。
// bbox 坐标默认对应 coord_w x coord_h 坐标系。
// 如果 UDP 未提供 coord_w/coord_h，则 UI 会自动使用当前视频帧尺寸。
struct SensorMeta {
    int frame_id = -1;
    long long recv_ms = 0;
    long long source_ts_ms = 0;
    double infer_ms = -1.0;

    int bbox_x = 0;
    int bbox_y = 0;
    int bbox_w = 0;
    int bbox_h = 0;

    int coord_w = 0;
    int coord_h = 0;

    float conf = 0.0f;
    bool has_detection = false;
    bool tracking = false;
};

struct VideoFrameTiming {
    int frame_id = -1;
    qint64 camera_estimated_mono_ms = -1;
    qint64 rtsp_grab_start_mono_ms = -1;
    qint64 rtsp_grab_done_mono_ms = -1;
    qint64 decode_done_mono_ms = -1;
    qint64 infer_start_mono_ms = -1;
    qint64 infer_end_mono_ms = -1;
    qint64 postprocess_done_mono_ms = -1;
    qint64 stream_enqueue_mono_ms = -1;
    qint64 jpeg_start_mono_ms = -1;
    qint64 jpeg_end_mono_ms = -1;
    qint64 tcp_send_start_mono_ms = -1;
    qint64 yolo_stream_send_ts_ms = -1;
    qint64 payload_size = -1;
    qint64 qt_receive_done_mono_ms = -1;
    qint64 qt_decode_start_mono_ms = -1;
    qint64 qt_decode_end_mono_ms = -1;
    qint64 qt_publish_mono_ms = -1;
    int display_frame_id = -1;
    int record_frame_id = -1;
    int display_dropped_old_frames = 0;
    int recording_queue_size = 0;
    int recording_dropped_frames = 0;
    qint64 recording_write_ms = -1;
};

// 保留旧环形缓冲，兼容旧程序；新 UI 主要使用 g_latestSensorMeta。
extern SensorMeta g_metaRingBuffer[META_BUFFER_SIZE];
extern std::atomic<int> g_yolo_max_frame;

extern std::shared_ptr<cv::Mat> g_latestVideoFrame;
extern std::shared_ptr<SensorMeta> g_latestSensorMeta;
extern std::shared_ptr<VideoFrameTiming> g_latestVideoTiming;
extern std::atomic<int> g_visionOverlayMaxAgeMs;

extern std::atomic<bool> g_hasVideoSignal;
extern std::atomic<int> g_videoWidth;
extern std::atomic<int> g_videoHeight;
extern std::atomic<bool> g_expectRkFrameStream;
extern std::atomic<bool> g_videoUsingRkStream;

// 音频雷达部分保留
extern std::atomic<float> g_latestAudioAngle;
extern std::atomic<bool> g_hasAudioSignal;
extern std::atomic<bool> g_latestAudioDetected;
extern std::atomic<bool> g_latestAudioDoaValid;
extern std::atomic<bool> g_latestAudioStableDoa;
extern std::atomic<double> g_latestAudioScoreEma;
extern std::atomic<double> g_latestAudioDoaStability;
extern std::atomic<double> g_latestAudioDoaConfidence;
extern std::atomic<double> g_latestAudioPanErrDeg;
extern std::atomic<bool> g_latestVisionTracking;
extern std::atomic<qint64> g_lastAudioRecvMs;

// Qt 端录像状态。按 X 请求开始/停止，视频线程在下一帧实际打开或关闭文件。
extern std::atomic<bool> g_recordingRequested;
extern std::atomic<bool> g_recordingActive;
extern std::atomic<bool> g_rawRecordingRequested;
extern std::atomic<bool> g_rawRecordingActive;
extern std::atomic<bool> g_audioRawRecordingRequested;
extern std::atomic<bool> g_audioRawRecordingActive;
extern std::atomic<bool> g_fusedRecordingRequested;
extern std::atomic<bool> g_fusedRecordingActive;
extern std::atomic<bool> g_displayEcoMode;

void logVideoDisplayTiming(const VideoFrameTiming& timing,
                           qint64 paintStartMonoMs,
                           qint64 paintEndMonoMs);
