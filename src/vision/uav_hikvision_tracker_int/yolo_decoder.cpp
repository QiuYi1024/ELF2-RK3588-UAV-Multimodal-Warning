#include "yolo_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using std::vector;

namespace {

const int RAW_HEAD_OUTPUT_COUNT = 6;
const int RAW_HEAD_REG_MAX = 16;
const int RAW_HEAD_REG_CHANNELS = 4 * RAW_HEAD_REG_MAX;
const float RAW_HEAD_MIN_DECODE_CONF = 0.005f;

struct RawHeadScale {
    int reg_index;
    int cls_index;
    int stride;
    int h;
    int w;
};

static inline bool output_size_plausible(const rknn_output& output, int expected_float_count) {
    if (output.size == 0) return true;

    // RKNN C API 在不同版本里对 size 的含义不完全一致：
    // 有的返回 want_float 后的 float 字节数，有的保留模型原始输出字节数。
    // 这里同时接受两种报法，真正读取仍然按 want_float 得到的 float buffer 访问。
    const unsigned int expected_as_elements = static_cast<unsigned int>(expected_float_count);
    const unsigned int expected_as_float_bytes = static_cast<unsigned int>(expected_float_count * sizeof(float));
    return output.size >= expected_as_elements || output.size >= expected_as_float_bytes;
}

static inline float sigmoid_stable(float x) {
    if (x >= 80.0f) return 1.0f;
    if (x <= -80.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// 对一个方向的 16 个 DFL bin 做 softmax，再计算期望距离。
static float dfl_expectation(const float* reg, int spatial, int hw, int direction) {
    const int base_channel = direction * RAW_HEAD_REG_MAX;
    float max_v = -std::numeric_limits<float>::infinity();

    for (int bin = 0; bin < RAW_HEAD_REG_MAX; ++bin) {
        const float v = reg[(base_channel + bin) * hw + spatial];
        if (v > max_v) max_v = v;
    }

    float sum = 0.0f;
    float expected = 0.0f;
    for (int bin = 0; bin < RAW_HEAD_REG_MAX; ++bin) {
        const float v = reg[(base_channel + bin) * hw + spatial];
        const float e = std::exp(v - max_v);
        sum += e;
        expected += e * static_cast<float>(bin);
    }

    if (sum <= 1e-9f || !std::isfinite(sum)) return 0.0f;
    return expected / sum;
}

} // namespace

static inline bool is_reasonable_box(float cx, float cy, float w, float h, int img_size) {
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) {
        return false;
    }

    if (w <= 1.0f || h <= 1.0f) return false;
    if (w > img_size * 1.25f || h > img_size * 1.25f) return false;
    if (cx < -img_size * 0.25f || cx > img_size * 1.25f) return false;
    if (cy < -img_size * 0.25f || cy > img_size * 1.25f) return false;

    return true;
}

std::vector<ObjectBox> decode_yolo11_single_output(rknn_output* outputs, int img_size) {
    vector<ObjectBox> boxes;

    if (outputs == nullptr || outputs[0].buf == nullptr) {
        return boxes;
    }

    const float* data = reinterpret_cast<const float*>(outputs[0].buf);

    int elem_count = 0;
    if (outputs[0].size > 0) {
        elem_count = static_cast<int>(outputs[0].size / sizeof(float));
    } else {
        elem_count = 5 * 8400;
    }

    // 单类别 YOLO11n 正确输出：[1,5,8400]
    // 固定按 CHW 解码，不再自动猜 HWC，避免把坐标误当成 conf=1。
    const int attrs = 5;
    if (elem_count % attrs != 0) {
        static int warn_count = 0;
        if (warn_count++ < 20) {
            std::cerr << "[YOLO11-DECODE] unexpected elem_count="
                      << elem_count << ", cannot divide by 5" << std::endl;
        }
        return boxes;
    }

    const int N = elem_count / attrs;
    boxes.reserve(128);

    float max_conf = 0.0f;
    int conf_over_005 = 0;
    int returned = 0;

    for (int i = 0; i < N; ++i) {
        float cx   = data[0 * N + i];
        float cy   = data[1 * N + i];
        float w    = data[2 * N + i];
        float h    = data[3 * N + i];
        float conf = data[4 * N + i];

        // 你已经验证 RGB_CHW_a5_c4_sig0，所以这里不做 sigmoid。
        if (!std::isfinite(conf)) continue;
        if (conf > max_conf) max_conf = conf;

        if (conf < 0.005f) continue;
        conf_over_005++;

        // 若未来导出得到 0~1 归一化坐标，这里自动放大。
        if (std::fabs(cx) <= 2.0f && std::fabs(cy) <= 2.0f &&
            std::fabs(w)  <= 2.0f && std::fabs(h)  <= 2.0f) {
            cx *= img_size;
            cy *= img_size;
            w  *= img_size;
            h  *= img_size;
        }

        if (!is_reasonable_box(cx, cy, w, h, img_size)) {
            continue;
        }

        boxes.push_back(ObjectBox{cx, cy, w, h, conf});
        returned++;
    }

    static int debug_count = 0;
    if (debug_count < 30) {
        std::cerr << "[YOLO11-DECODE-RGB-CHW] N=" << N
                  << ", max_conf=" << max_conf
                  << ", conf>0.005=" << conf_over_005
                  << ", returned=" << returned
                  << std::endl;
        debug_count++;
    }

    return boxes;
}

std::vector<ObjectBox> decode_yolo11_raw_head_outputs(rknn_output* outputs, int output_count, int img_size) {
    vector<ObjectBox> boxes;

    if (outputs == nullptr || output_count != RAW_HEAD_OUTPUT_COUNT) {
        static int warn_count = 0;
        if (warn_count++ < 20) {
            std::cerr << "[YOLO11-RAW-DECODE] unsupported output_count="
                      << output_count << ", expected 6" << std::endl;
        }
        return boxes;
    }

    const RawHeadScale scales[3] = {
        {0, 1, 8,  80, 80},
        {2, 3, 16, 40, 40},
        {4, 5, 32, 20, 20},
    };

    boxes.reserve(256);

    float max_conf = 0.0f;
    int conf_over_decode = 0;
    int returned = 0;

    for (int scale_idx = 0; scale_idx < 3; ++scale_idx) {
        const RawHeadScale& scale = scales[scale_idx];
        const rknn_output& reg_output = outputs[scale.reg_index];
        const rknn_output& cls_output = outputs[scale.cls_index];

        if (reg_output.buf == nullptr || cls_output.buf == nullptr) {
            static int warn_count = 0;
            if (warn_count++ < 20) {
                std::cerr << "[YOLO11-RAW-DECODE] null output buffer at scale "
                          << scale_idx << std::endl;
            }
            continue;
        }

        const int hw = scale.h * scale.w;
        const int expected_reg = RAW_HEAD_REG_CHANNELS * hw;
        const int expected_cls = hw;
        if (!output_size_plausible(reg_output, expected_reg) ||
            !output_size_plausible(cls_output, expected_cls)) {
            static int warn_count = 0;
            if (warn_count++ < 20) {
                std::cerr << "[YOLO11-RAW-DECODE] unexpected output size at scale "
                          << scale_idx << ": reg_size=" << reg_output.size
                          << " expected elements=" << expected_reg
                          << ", cls_size=" << cls_output.size
                          << " expected elements=" << expected_cls << std::endl;
            }
            continue;
        }

        const float* reg = reinterpret_cast<const float*>(reg_output.buf);
        const float* cls = reinterpret_cast<const float*>(cls_output.buf);

        for (int y = 0; y < scale.h; ++y) {
            for (int x = 0; x < scale.w; ++x) {
                const int spatial = y * scale.w + x;
                const float conf = sigmoid_stable(cls[spatial]);

                if (!std::isfinite(conf)) continue;
                if (conf > max_conf) max_conf = conf;
                if (conf < RAW_HEAD_MIN_DECODE_CONF) continue;
                conf_over_decode++;

                const float left = dfl_expectation(reg, spatial, hw, 0) * scale.stride;
                const float top = dfl_expectation(reg, spatial, hw, 1) * scale.stride;
                const float right = dfl_expectation(reg, spatial, hw, 2) * scale.stride;
                const float bottom = dfl_expectation(reg, spatial, hw, 3) * scale.stride;

                const float anchor_x = (static_cast<float>(x) + 0.5f) * scale.stride;
                const float anchor_y = (static_cast<float>(y) + 0.5f) * scale.stride;
                const float x1 = anchor_x - left;
                const float y1 = anchor_y - top;
                const float x2 = anchor_x + right;
                const float y2 = anchor_y + bottom;
                const float w = x2 - x1;
                const float h = y2 - y1;
                const float cx = (x1 + x2) * 0.5f;
                const float cy = (y1 + y2) * 0.5f;

                if (!is_reasonable_box(cx, cy, w, h, img_size)) {
                    continue;
                }

                boxes.push_back(ObjectBox{cx, cy, w, h, conf});
                returned++;
            }
        }
    }

    static int debug_count = 0;
    if (debug_count < 30) {
        std::cerr << "[YOLO11-RAW-DECODE] max_conf=" << max_conf
                  << ", conf>0.005=" << conf_over_decode
                  << ", returned=" << returned
                  << std::endl;
        debug_count++;
    }

    return boxes;
}
