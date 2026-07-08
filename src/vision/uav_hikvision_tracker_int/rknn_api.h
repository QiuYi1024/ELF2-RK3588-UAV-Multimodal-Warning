#ifndef ANTIUAV_MINIMAL_RKNN_API_H
#define ANTIUAV_MINIMAL_RKNN_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Minimal RKNN C API declarations used by this project.
 * ELF2 currently provides /usr/lib/librknnrt.so but not the Rockchip
 * development header. Keep this file limited to the symbols referenced by
 * main.cpp and yolo_decoder.cpp, and replace it with the BSP-matched
 * Rockchip header when that header is available on the board.
 */

typedef uint64_t rknn_context;

typedef enum {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8 = 2,
    RKNN_TENSOR_UINT8 = 3,
    RKNN_TENSOR_INT16 = 4,
    RKNN_TENSOR_UINT16 = 5,
    RKNN_TENSOR_INT32 = 6,
    RKNN_TENSOR_UINT32 = 7,
    RKNN_TENSOR_INT64 = 8,
    RKNN_TENSOR_BOOL = 9,
    RKNN_TENSOR_INT4 = 10
} rknn_tensor_type;

typedef enum {
    RKNN_TENSOR_NCHW = 0,
    RKNN_TENSOR_NHWC = 1,
    RKNN_TENSOR_NC1HWC2 = 2,
    RKNN_TENSOR_UNDEFINED = 3
} rknn_tensor_format;

typedef enum {
    RKNN_QUERY_IN_OUT_NUM = 0,
    RKNN_QUERY_INPUT_ATTR = 1,
    RKNN_QUERY_OUTPUT_ATTR = 2
} rknn_query_cmd;

typedef enum {
    RKNN_NPU_CORE_AUTO = 0,
    RKNN_NPU_CORE_0 = 1,
    RKNN_NPU_CORE_1 = 2,
    RKNN_NPU_CORE_2 = 4,
    RKNN_NPU_CORE_0_1 = 3,
    RKNN_NPU_CORE_0_1_2 = 7,
    RKNN_NPU_CORE_ALL = 0xffff
} rknn_core_mask;

typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[16];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    int32_t qnt_type;
    int32_t fl;
    int32_t zp;
    float scale;
    char name[256];
    uint32_t pass_through;
    uint32_t h_stride;
    uint32_t w_stride;
    uint32_t reserved;
} rknn_tensor_attr;

typedef struct {
    uint32_t index;
    void* buf;
    uint32_t size;
    uint8_t pass_through;
    rknn_tensor_type type;
    rknn_tensor_format fmt;
} rknn_input;

typedef struct {
    uint8_t want_float;
    uint8_t is_prealloc;
    uint32_t index;
    void* buf;
    uint32_t size;
} rknn_output;

int rknn_init(rknn_context* context, void* model, uint32_t size, uint32_t flag, void* extend);
int rknn_destroy(rknn_context context);
int rknn_query(rknn_context context, rknn_query_cmd cmd, void* info, uint32_t size);
int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]);
int rknn_run(rknn_context context, void* extend);
int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[], void* extend);
int rknn_outputs_release(rknn_context context, uint32_t n_outputs, rknn_output outputs[]);
int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask);

#ifdef __cplusplus
}
#endif

#endif
