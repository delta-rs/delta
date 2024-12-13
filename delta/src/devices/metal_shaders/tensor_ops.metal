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
            output[id] = NAN;
            return;
        }

        // Perform addition
        float result = a + b;
        
        output[id] = result;
    }
}

kernel void tensor_subtract(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            constant uint& tensor_length [[ buffer(3) ]],
                            uint id [[ thread_position_in_grid ]]) {
    if (id < tensor_length) {
        float a = input1[id];
        float b = input2[id];

        if (isnan(b)) {
            // output[id] = NAN; // Example marker for NaN in input2
            output[id] = NAN;
            return;
        }

        // Perform addition
        float result = a - b;
        
        output[id] = result;
    } 
}

kernel void tensor_multiply(const device float* input1 [[ buffer(0) ]],
                            const device float* input2 [[ buffer(1) ]],
                            device float* output [[ buffer(2) ]],
                            constant uint& tensor_length [[ buffer(3) ]],
                            uint id [[ thread_position_in_grid ]]) {
    if (id < tensor_length) {
        float a = input1[id];
        float b = input2[id];

        if (isnan(b)) {
            // output[id] = NAN; // Example marker for NaN in input2
            output[id] = NAN;
            return;
        }

        // Perform addition
        float result = a * b;
        
        output[id] = result;
    }
}

kernel void tensor_divide(const device float* input1 [[ buffer(0) ]],
                          const device float* input2 [[ buffer(1) ]],
                          device float* output [[ buffer(2) ]],
                          constant uint& tensor_length [[ buffer(3) ]],
                          uint id [[ thread_position_in_grid ]]) {
    if (id < tensor_length) {
        float a = input1[id];
        float b = input2[id];

        if (isnan(b) || b == 0.0f) {
            // Handle NaN or divide by zero
            output[id] = NAN;
            return;
        }

        // Perform division
        float result = a / b;

        output[id] = result;
    }
}
