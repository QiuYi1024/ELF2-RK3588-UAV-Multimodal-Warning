#include "bytetrack_lite.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

using namespace std;
using namespace cv;

namespace {

Rect2f detToRect(const ByteTrackDetection& d) {
    return Rect2f(d.x1, d.y1, max(1.0f, d.x2 - d.x1), max(1.0f, d.y2 - d.y1));
}

void rectToMeasurement(const Rect2f& r, Mat& m) {
    const float cx = r.x + r.width * 0.5f;
    const float cy = r.y + r.height * 0.5f;
    const float aspect = r.width / max(1.0f, r.height);
    m = (Mat_<float>(4, 1) << cx, cy, aspect, r.height);
}

Rect2f stateToRect(const Mat& s) {
    const float cx = s.at<float>(0);
    const float cy = s.at<float>(1);
    const float aspect = max(0.05f, s.at<float>(2));
    const float h = max(1.0f, s.at<float>(3));
    const float w = max(1.0f, aspect * h);
    return Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
}

double jsonNumber(const string& text, const string& key, double default_value) {
    string needle = "\"" + key + "\"";
    size_t p = text.find(needle);
    if (p == string::npos) return default_value;
    p = text.find(':', p + needle.size());
    if (p == string::npos) return default_value;
    ++p;
    while (p < text.size() && isspace((unsigned char)text[p])) ++p;
    char* end = nullptr;
    double v = strtod(text.c_str() + p, &end);
    if (end == text.c_str() + p || !isfinite(v)) return default_value;
    return v;
}

vector<cv::Vec3d> defaultFovTable() {
    return {
        Vec3d(1.0, 60.000, 36.000),
        Vec3d(1.2, 52.740, 32.327),
        Vec3d(2.4, 28.300, 17.000),
        Vec3d(5.0, 14.642, 8.582),
        Vec3d(10.0, 7.564, 4.419),
        Vec3d(13.1, 5.852, 3.445),
        Vec3d(20.0, 4.057, 2.419),
        Vec3d(37.0, 2.478, 1.261),
    };
}

vector<pair<int, int>> hungarianAssign(const vector<vector<double>>& cost, double max_cost) {
    const int n = (int)cost.size();
    const int m = n > 0 ? (int)cost[0].size() : 0;
    if (n == 0 || m == 0) return {};

    const int sz = max(n, m);
    const double big = 1e6;
    vector<vector<double>> a(sz + 1, vector<double>(sz + 1, big));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i + 1][j + 1] = cost[i][j];
        }
    }

    vector<double> u(sz + 1), v(sz + 1);
    vector<int> p(sz + 1), way(sz + 1);
    for (int i = 1; i <= sz; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(sz + 1, big);
        vector<char> used(sz + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = big;
            int j1 = 0;
            for (int j = 1; j <= sz; ++j) {
                if (used[j]) continue;
                double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= sz; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    vector<pair<int, int>> matches;
    for (int j = 1; j <= sz; ++j) {
        int i = p[j];
        if (i >= 1 && i <= n && j <= m && cost[i - 1][j - 1] <= max_cost) {
            matches.emplace_back(i - 1, j - 1);
        }
    }
    return matches;
}

} // namespace

const char* byteTrackStateName(ByteTrackState state) {
    switch (state) {
        case ByteTrackState::Tracked: return "Tracked";
        case ByteTrackState::Lost: return "Lost";
        case ByteTrackState::Removed: return "Removed";
        default: return "Unknown";
    }
}

float byteTrackIou(const Rect2f& a, const Rect2f& b) {
    const float xx1 = max(a.x, b.x);
    const float yy1 = max(a.y, b.y);
    const float xx2 = min(a.x + a.width, b.x + b.width);
    const float yy2 = min(a.y + a.height, b.y + b.height);
    const float w = max(0.0f, xx2 - xx1);
    const float h = max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

pair<double, double> byteTrackInterpFov(const ByteTrackLiteConfig& cfg, double zoom_ratio) {
    const vector<Vec3d>& table = cfg.fov_table.empty() ? defaultFovTable() : cfg.fov_table;
    double z = max(table.front()[0], min(table.back()[0], zoom_ratio));
    for (size_t i = 0; i + 1 < table.size(); ++i) {
        const Vec3d& a = table[i];
        const Vec3d& b = table[i + 1];
        if (a[0] <= z && z <= b[0]) {
            double t = (z - a[0]) / max(1e-9, b[0] - a[0]);
            return {a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
        }
    }
    return {table.back()[1], table.back()[2]};
}

ByteTrackLite::ByteTrackLite(const ByteTrackLiteConfig& cfg) : cfg_(cfg) {
    if (cfg_.fov_table.empty()) cfg_.fov_table = defaultFovTable();
}

ByteTrackLite::Track ByteTrackLite::createTrack(const ByteTrackDetection& det) {
    Track t;
    t.id = next_id_++;
    t.bbox = detToRect(det);
    t.score = det.score;
    t.hits = 1;
    t.age = 1;
    t.lost_frames = 0;
    t.state = ByteTrackState::Tracked;
    t.kf.init(8, 4, 0);
    t.kf.transitionMatrix = (Mat_<float>(8, 8) <<
        1,0,0,0,1,0,0,0,
        0,1,0,0,0,1,0,0,
        0,0,1,0,0,0,1,0,
        0,0,0,1,0,0,0,1,
        0,0,0,0,1,0,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,1);
    t.kf.measurementMatrix = Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i) t.kf.measurementMatrix.at<float>(i, i) = 1.0f;
    setIdentity(t.kf.processNoiseCov, Scalar::all(0.02));
    setIdentity(t.kf.measurementNoiseCov, Scalar::all(0.10));
    setIdentity(t.kf.errorCovPost, Scalar::all(1.0));
    Mat m;
    rectToMeasurement(t.bbox, m);
    t.kf.statePost = Mat::zeros(8, 1, CV_32F);
    for (int i = 0; i < 4; ++i) t.kf.statePost.at<float>(i) = m.at<float>(i);
    t.kf_ready = true;
    return t;
}

void ByteTrackLite::predictTrack(Track& trk) {
    if (!trk.kf_ready) return;
    Mat p = trk.kf.predict();
    trk.bbox = stateToRect(p);
    trk.age += 1;
}

void ByteTrackLite::updateTrack(Track& trk, const ByteTrackDetection& det) {
    Mat m;
    rectToMeasurement(detToRect(det), m);
    Mat c = trk.kf.correct(m);
    trk.bbox = stateToRect(c);
    trk.score = det.score;
    trk.hits += 1;
    trk.lost_frames = 0;
    trk.state = ByteTrackState::Tracked;
}

void ByteTrackLite::markLostOrRemoved(Track& trk) {
    if (trk.state == ByteTrackState::Removed) return;
    trk.state = ByteTrackState::Lost;
    trk.lost_frames += 1;
    if (trk.lost_frames > cfg_.track_buffer) {
        trk.state = ByteTrackState::Removed;
    }
}

void ByteTrackLite::applyPtzCompensation(Track& trk, const ByteTrackPtzState& ptz) {
    if (!ptz.valid || !last_ptz_.valid) return;
    const int fw = max(1, ptz.frame_width);
    const int fh = max(1, ptz.frame_height);
    const double hfov = ptz.hfov_deg > 0.0 ? ptz.hfov_deg : byteTrackInterpFov(cfg_, ptz.zoom_ratio).first;
    const double vfov = ptz.vfov_deg > 0.0 ? ptz.vfov_deg : byteTrackInterpFov(cfg_, ptz.zoom_ratio).second;
    const double prev_hfov = last_ptz_.hfov_deg > 0.0 ? last_ptz_.hfov_deg : byteTrackInterpFov(cfg_, last_ptz_.zoom_ratio).first;
    const double prev_vfov = last_ptz_.vfov_deg > 0.0 ? last_ptz_.vfov_deg : byteTrackInterpFov(cfg_, last_ptz_.zoom_ratio).second;

    const double dpan = ptz.pan_deg - last_ptz_.pan_deg;
    const double dtilt = ptz.tilt_deg - last_ptz_.tilt_deg;
    const double dx = cfg_.sign_pan * cfg_.k_pan * dpan / max(1e-6, hfov) * fw;
    const double dy = cfg_.sign_tilt * cfg_.k_tilt * dtilt / max(1e-6, vfov) * fh;
    const double sx = max(0.25, min(4.0, prev_hfov / max(1e-6, hfov)));
    const double sy = max(0.25, min(4.0, prev_vfov / max(1e-6, vfov)));

    const float cx = trk.bbox.x + trk.bbox.width * 0.5f + (float)dx;
    const float cy = trk.bbox.y + trk.bbox.height * 0.5f + (float)dy;
    const float nw = max(1.0f, trk.bbox.width * (float)sx);
    const float nh = max(1.0f, trk.bbox.height * (float)sy);
    trk.bbox = Rect2f(cx - nw * 0.5f, cy - nh * 0.5f, nw, nh);

    if (trk.kf_ready) {
        Mat m;
        rectToMeasurement(trk.bbox, m);
        for (int i = 0; i < 4; ++i) trk.kf.statePost.at<float>(i) = m.at<float>(i);
    }
}

vector<pair<int, int>> ByteTrackLite::associate(const vector<int>& track_indices,
                                                const vector<int>& det_indices,
                                                const vector<ByteTrackDetection>& dets,
                                                float iou_thresh) const {
    if (track_indices.empty() || det_indices.empty()) return {};
    vector<vector<double>> cost(track_indices.size(), vector<double>(det_indices.size(), 1e6));
    for (size_t i = 0; i < track_indices.size(); ++i) {
        const Track& t = tracks_[track_indices[i]];
        for (size_t j = 0; j < det_indices.size(); ++j) {
            float iou = byteTrackIou(t.bbox, detToRect(dets[det_indices[j]]));
            if (iou >= iou_thresh) cost[i][j] = 1.0 - iou;
        }
    }
    vector<pair<int, int>> local = hungarianAssign(cost, 1.0 - iou_thresh);
    vector<pair<int, int>> out;
    for (auto& p : local) out.emplace_back(track_indices[p.first], det_indices[p.second]);
    return out;
}

vector<int> ByteTrackLite::unmatchedTracks(const vector<int>& track_indices,
                                           const vector<pair<int, int>>& matches) const {
    vector<int> out;
    for (int ti : track_indices) {
        bool matched = false;
        for (const auto& m : matches) if (m.first == ti) matched = true;
        if (!matched) out.push_back(ti);
    }
    return out;
}

vector<int> ByteTrackLite::unmatchedDetections(const vector<int>& det_indices,
                                               const vector<pair<int, int>>& matches) const {
    vector<int> out;
    for (int di : det_indices) {
        bool matched = false;
        for (const auto& m : matches) if (m.second == di) matched = true;
        if (!matched) out.push_back(di);
    }
    return out;
}

ByteTrackOutput ByteTrackLite::toOutput(const Track& trk) const {
    ByteTrackOutput o;
    o.track_id = trk.id;
    o.x1 = trk.bbox.x;
    o.y1 = trk.bbox.y;
    o.x2 = trk.bbox.x + trk.bbox.width;
    o.y2 = trk.bbox.y + trk.bbox.height;
    o.center_x = trk.bbox.x + trk.bbox.width * 0.5f;
    o.center_y = trk.bbox.y + trk.bbox.height * 0.5f;
    o.score = trk.score;
    o.state = byteTrackStateName(trk.state);
    o.lost_frames = trk.lost_frames;
    o.confirmed = trk.state == ByteTrackState::Tracked && trk.hits >= cfg_.confirm_frames;
    return o;
}

vector<ByteTrackOutput> ByteTrackLite::update(const vector<ByteTrackDetection>& detections,
                                              const ByteTrackPtzState& ptz) {
    vector<ByteTrackDetection> dets;
    for (const auto& d : detections) {
        if (d.class_id == cfg_.drone_class_id && d.score >= cfg_.low_thresh) dets.push_back(d);
    }
    sort(dets.begin(), dets.end(), [](const ByteTrackDetection& a, const ByteTrackDetection& b) {
        return a.score > b.score;
    });

    vector<int> high, low;
    for (int i = 0; i < (int)dets.size(); ++i) {
        if (dets[i].score >= cfg_.high_thresh) high.push_back(i);
        else low.push_back(i);
    }

    vector<int> candidate_tracks;
    for (int i = 0; i < (int)tracks_.size(); ++i) {
        if (tracks_[i].state != ByteTrackState::Removed) {
            predictTrack(tracks_[i]);
            applyPtzCompensation(tracks_[i], ptz);
            candidate_tracks.push_back(i);
        }
    }

    vector<pair<int, int>> matches1 = associate(candidate_tracks, high, dets, cfg_.match_thresh);
    for (const auto& m : matches1) updateTrack(tracks_[m.first], dets[m.second]);

    vector<int> unmatched_after_high = unmatchedTracks(candidate_tracks, matches1);
    vector<pair<int, int>> matches2 = associate(unmatched_after_high, low, dets, max(0.10f, cfg_.match_thresh * 0.7f));
    for (const auto& m : matches2) updateTrack(tracks_[m.first], dets[m.second]);

    vector<pair<int, int>> all_matches = matches1;
    all_matches.insert(all_matches.end(), matches2.begin(), matches2.end());
    for (int ti : unmatchedTracks(candidate_tracks, all_matches)) {
        markLostOrRemoved(tracks_[ti]);
    }

    for (int di : unmatchedDetections(high, all_matches)) {
        tracks_.push_back(createTrack(dets[di]));
    }

    tracks_.erase(remove_if(tracks_.begin(), tracks_.end(), [](const Track& t) {
        return t.state == ByteTrackState::Removed;
    }), tracks_.end());

    outputs_.clear();
    for (const auto& t : tracks_) {
        // 已确认过的轨迹即使短暂 Lost 也输出到日志/UI，便于观察 lost_frames。
        // 云台闭环仍只使用 confirmed=true 的 Tracked 轨迹，避免追随纯预测框。
        if (t.hits >= cfg_.confirm_frames && t.state != ByteTrackState::Removed) {
            outputs_.push_back(toOutput(t));
        }
    }
    sort(outputs_.begin(), outputs_.end(), [](const ByteTrackOutput& a, const ByteTrackOutput& b) {
        if (a.lost_frames != b.lost_frames) return a.lost_frames < b.lost_frames;
        return a.score > b.score;
    });
    last_ptz_ = ptz;
    return outputs_;
}

ByteTrackLiteConfig loadByteTrackLiteConfig(const string& path) {
    ByteTrackLiteConfig cfg;
    cfg.fov_table = defaultFovTable();
    ifstream ifs(path);
    if (!ifs.good()) return cfg;
    string text((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    cfg.high_thresh = (float)jsonNumber(text, "high_thresh", cfg.high_thresh);
    cfg.low_thresh = (float)jsonNumber(text, "low_thresh", cfg.low_thresh);
    cfg.match_thresh = (float)jsonNumber(text, "match_thresh", cfg.match_thresh);
    cfg.track_buffer = (int)jsonNumber(text, "track_buffer", cfg.track_buffer);
    cfg.confirm_frames = (int)jsonNumber(text, "confirm_frames", cfg.confirm_frames);
    cfg.drone_class_id = (int)jsonNumber(text, "drone_class_id", cfg.drone_class_id);
    cfg.sign_pan = jsonNumber(text, "sign_pan", cfg.sign_pan);
    cfg.sign_tilt = jsonNumber(text, "sign_tilt", cfg.sign_tilt);
    cfg.k_pan = jsonNumber(text, "k_pan", cfg.k_pan);
    cfg.k_tilt = jsonNumber(text, "k_tilt", cfg.k_tilt);
    vector<Vec3d> parsed_fov;
    size_t table_pos = text.find("\"fov_table\"");
    if (table_pos != string::npos) {
        size_t pos = table_pos;
        while ((pos = text.find("\"zoom\"", pos)) != string::npos) {
            size_t end = text.find('}', pos);
            if (end == string::npos) break;
            string item = text.substr(pos, end - pos + 1);
            double z = jsonNumber(item, "zoom", -1.0);
            double h = jsonNumber(item, "hfov", -1.0);
            double v = jsonNumber(item, "vfov", -1.0);
            if (z > 0.0 && h > 0.0 && v > 0.0) {
                parsed_fov.push_back(Vec3d(z, h, v));
            }
            pos = end + 1;
        }
    }
    if (!parsed_fov.empty()) {
        sort(parsed_fov.begin(), parsed_fov.end(), [](const Vec3d& a, const Vec3d& b) {
            return a[0] < b[0];
        });
        cfg.fov_table = parsed_fov;
    }
    cfg.high_thresh = max(0.0f, min(1.0f, cfg.high_thresh));
    cfg.low_thresh = max(0.0f, min(cfg.high_thresh, cfg.low_thresh));
    cfg.match_thresh = max(0.05f, min(0.95f, cfg.match_thresh));
    cfg.track_buffer = max(1, cfg.track_buffer);
    cfg.confirm_frames = max(1, cfg.confirm_frames);
    return cfg;
}
