#pragma once
#include <cuda_runtime.h>

// replace glm::vec3
struct Float3
{
    float x;
    float y;
    float z;

    __host__ __device__ Float3(): x(0.0f), y(0.0f), z(0.0f) {};
    __host__ __device__ Float3(float _x, float _y, float _z): x(_x), y(_y), z(_z) {};
};