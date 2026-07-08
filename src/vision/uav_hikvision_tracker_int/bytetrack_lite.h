#ifndef BYTETRACK_LITE_H
#define BYTETRACK_LITE_H

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct ByteTrackDetection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = 0;
};

struct ByteTrackPtzState {
    bool valid = false;
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
    double zoom_ratio = 1.0;
    int frame_width = 0;
    int frame_height = 0;
    double hfov_deg = 0.0;
    double vfov_deg = 0.0;
};

enum class ByteTrackState {
    Tracked = 0,
    Lost = 1,
    Removed = 2
};

struct ByteTrackOutput {
    int track_id = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float score = 0.0f;
    std::string state = "Lost";
    int lost_frames = 0;
    bool confirmed = false;
};

struct ByteTrackLiteConfig {
    float high_thresh = 0.55f;
    float low_thresh = 0.10f;
    float match_thresh = 0.35f;
    int track_buffer = 30;
    int confirm_frames = 2;
    int drone_class_id = 0;
    double sign_pan = -1.0;
    double sign_tilt = -1.0;
    double k_pan = 1.0;
    double k_tilt = 1.0;
    std::vector<cv::Vec3d> fov_table;
};

class ByteTrackLite {
public:
    explicit ByteTrackLite(const ByteTrackLiteConfig& cfg = ByteTrackLiteConfig());

    std::vector<ByteTrackOutput> update(const std::vector<ByteTrackDetection>& detections,
                                        const ByteTrackPtzState& ptz);

    const std::vector<ByteTrackOutput>& last_outputs() const { return outputs_; }
    ByteTrackLiteConfig config() const { return cfg_; }

private:
    struct Track {
        int id = -1;
        ByteTrackState state = ByteTrackState::Tracked;
        cv::KalmanFilter kf;
        cv::Rect2f bbox;
        float score = 0.0f;
        int hits = 0;
        int age = 0;
        int lost_frames = 0;
        bool kf_ready = false;
    };

    ByteTrackLiteConfig cfg_;
    std::vector<Track> tracks_;
    std::vector<ByteTrackOutput> outputs_;
    ByteTrackPtzState last_ptz_;
    int next_id_ = 1;

    Track createTrack(const ByteTrackDetection& det);
    void predictTrack(Track& trk);
    void updateTrack(Track& trk, const ByteTrackDetection& det);
    void markLostOrRemoved(Track& trk);
    void applyPtzCompensation(Track& trk, const ByteTrackPtzState& ptz);
    std::vector<std::pair<int, int>> associate(const std::vector<int>& track_indices,
                                               const std::vector<int>& det_indices,
                                               const std::vector<ByteTrackDetection>& dets,
                                               float iou_thresh) const;
    std::vector<int> unmatchedTracks(const std::vector<int>& track_indices,
                                     const std::vector<std::pair<int, int>>& matches) const;
    std::vector<int> unmatchedDetections(const std::vector<int>& det_indices,
                                         const std::vector<std::pair<int, int>>& matches) const;
    ByteTrackOutput toOutput(const Track& trk) const;
};

ByteTrackLiteConfig loadByteTrackLiteConfig(const std::string& path);
const char* byteTrackStateName(ByteTrackState state);
float byteTrackIou(const cv::Rect2f& a, const cv::Rect2f& b);
std::pair<double, double> byteTrackInterpFov(const ByteTrackLiteConfig& cfg, double zoom_ratio);

#endif // BYTETRACK_LITE_H
