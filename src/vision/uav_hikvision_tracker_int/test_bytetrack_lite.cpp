#include "bytetrack_lite.h"

#include <cassert>
#include <iostream>

int main() {
    ByteTrackLiteConfig cfg;
    cfg.high_thresh = 0.55f;
    cfg.low_thresh = 0.10f;
    cfg.match_thresh = 0.30f;
    cfg.confirm_frames = 2;
    cfg.track_buffer = 2;
    ByteTrackLite tracker(cfg);

    ByteTrackPtzState ptz;
    ptz.valid = true;
    ptz.frame_width = 704;
    ptz.frame_height = 576;
    ptz.pan_deg = 10.0;
    ptz.tilt_deg = 0.0;
    ptz.zoom_ratio = 1.0;
    ptz.hfov_deg = 60.0;
    ptz.vfov_deg = 36.0;

    std::vector<ByteTrackDetection> frame1 = {{100, 100, 150, 150, 0.80f, 0}};
    auto out1 = tracker.update(frame1, ptz);
    assert(out1.empty());

    ptz.pan_deg = 10.2;
    std::vector<ByteTrackDetection> frame2 = {{103, 101, 153, 151, 0.78f, 0}};
    auto out2 = tracker.update(frame2, ptz);
    assert(out2.size() == 1);
    assert(out2[0].track_id == 1);
    assert(out2[0].confirmed);

    ptz.pan_deg = 11.0;
    std::vector<ByteTrackDetection> low = {{108, 102, 158, 152, 0.20f, 0}};
    auto out3 = tracker.update(low, ptz);
    assert(out3.size() == 1);
    assert(out3[0].track_id == 1);

    ptz.pan_deg = 11.3;
    std::vector<ByteTrackDetection> empty;
    auto out4 = tracker.update(empty, ptz);
    assert(out4.size() == 1);
    assert(out4[0].track_id == 1);
    assert(out4[0].state == "Lost");
    assert(out4[0].lost_frames == 1);
    assert(!out4[0].confirmed);

    auto fov = byteTrackInterpFov(cfg, 2.4);
    assert(fov.first > 20.0 && fov.first < 40.0);

    ByteTrackLiteConfig file_cfg = loadByteTrackLiteConfig("src/vision/uav_hikvision_tracker_int/tracker_config.json");
    assert(file_cfg.high_thresh > 0.0f);
    assert(file_cfg.fov_table.size() >= 2);
    auto file_fov = byteTrackInterpFov(file_cfg, 5.0);
    assert(file_fov.first > 10.0 && file_fov.first < 20.0);

    std::cout << "bytetrack_lite tests passed\n";
    return 0;
}
