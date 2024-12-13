#include <metal_stdlib>
using namespace metal;

kernel void tensor_add(const device float* input1 [[ buffer(0) ]],
                       const device float* input2 [[ buffer(1) ]],
                       device float* output [[ buffer(2) ]],
                       constant uint& tensor_length [[ buffer(3) ]],
                       uint id [[ thread_position_in_grid ]]) {
    if (id < tensor_length) {
        float a = input1[id];
        float b = input2[id];

        if (isnan(b)) {
            // output[id] = NAN; // Example marker for NaN in input2
            b = clamp(b, -FLT_MAX, FLT_MAX);
            return;
        }

        // Perform addition
        float result = a + b;

        // Handle overflow or underflow
        if (isinf(result)) {
            output[id] = clamp(a + b, -FLT_MAX, FLT_MAX);
        } else {
            output[id] = result;
        }
    }
}

kernel void tensor_subtract(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    output[id] = clamp(input1[id] - input2[id], -FLT_MAX, FLT_MAX);
}

kernel void tensor_multiply(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    output[id] = clamp(input1[id] * input2[id], -FLT_MAX, FLT_MAX);
}

kernel void tensor_divide(const device float* input1 [[ buffer(0) ]],
                          const device float* input2 [[ buffer(1) ]],
                          device float* output [[ buffer(2) ]],
                          uint id [[ thread_position_in_grid ]]) {
    output[id] = clamp(input1[id] / input2[id], -FLT_MAX, FLT_MAX);
}
