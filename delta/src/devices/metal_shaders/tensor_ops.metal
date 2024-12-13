#include <metal_stdlib>
using namespace metal;

kernel void tensor_add(const device float* input1 [[ buffer(0) ]],
                       const device float* input2 [[ buffer(1) ]],
                       device float* output [[ buffer(2) ]],
                       uint id [[ thread_position_in_grid ]]) {
    output[id] = input1[id] + input2[id];
}

kernel void tensor_subtract(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    output[id] = input1[id] - input2[id];
}

kernel void tensor_multiply(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    output[id] = input1[id] * input2[id];
}

kernel void tensor_divide(const device float* input1 [[ buffer(0) ]],
                          const device float* input2 [[ buffer(1) ]],
                          device float* output [[ buffer(2) ]],
                          uint id [[ thread_position_in_grid ]]) {
    output[id] = input1[id] / input2[id];
}
