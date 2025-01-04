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

kernel void tensor_matmul(const device float* input1 [[ buffer(0) ]],
                          const device float* input2 [[ buffer(1) ]],
                          device float* output [[ buffer(2) ]],
                          constant uint& rows_a [[ buffer(3) ]],
                          constant uint& cols_a [[ buffer(4) ]],
                          constant uint& cols_b [[ buffer(5) ]],
                          uint id [[ thread_position_in_grid ]]) {
    uint row = id / cols_b; // Compute row of output matrix
    uint col = id % cols_b; // Compute column of output matrix

    if (row < rows_a && col < cols_b) {
        float sum = 0.0;
        for (uint k = 0; k < cols_a; k++) {
            sum += input1[row * cols_a + k] * input2[k * cols_b + col];
        }
        output[row * cols_b + col] = sum;
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

kernel void tensor_power(const device float* input1 [[ buffer(0) ]],
                         constant float& power [[ buffer(4) ]],
                         device float* output [[ buffer(2) ]],
                         constant uint& tensor_length [[ buffer(3) ]],
                         uint id [[ thread_position_in_grid ]]) {
    if (id < tensor_length) {
        float a = input1[id];

        // Perform power operation
        float result = pow(a, power);

        output[id] = result;
    }
}