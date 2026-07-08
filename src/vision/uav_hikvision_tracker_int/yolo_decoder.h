#ifndef YOLO_DECODER_H
#define YOLO_DECODER_H

#include <vector>
#include "rknn_api.h"

// YOLO 解码后的框，坐标在 640x640 letterbox 输入坐标系内。
// x/y 为中心点，w/h 为宽高，conf 为置信度。
struct ObjectBox {
    float x;
    float y;
    float w;
    float h;
    float conf;
};

// Ultralytics YOLO11n 单类别导出到 RKNN 后的旧单输出解码。
// 固定使用已经验证过的正确模式：RGB_CHW_a5_c4_sig0
// output0 = [1, 5, 8400]
// ch0=cx, ch1=cy, ch2=w, ch3=h, ch4=conf
std::vector<ObjectBox> decode_yolo11_single_output(rknn_output* outputs, int img_size = 640);

// YOLO11 raw-head INT8 解码。
// 新版 INT8 模型输出 6 个张量：
//   output0 reg_s8   [1,64,80,80]
//   output1 cls_s8   [1, 1,80,80]
//   output2 reg_s16  [1,64,40,40]
//   output3 cls_s16  [1, 1,40,40]
//   output4 reg_s32  [1,64,20,20]
//   output5 cls_s32  [1, 1,20,20]
// 其中 64 = 4 * reg_max，reg_max = 16；cls 为未 sigmoid 的单类别 logit。
std::vector<ObjectBox> decode_yolo11_raw_head_outputs(
    rknn_output* outputs,
    int output_count,
    int img_size = 640
);

#endif // YOLO_DECODER_H
