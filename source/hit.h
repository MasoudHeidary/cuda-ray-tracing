#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <limits>

struct Hit
{
    bool hit;
    float t;          // Distance from the ray origin to the intersection point
    glm::vec3 color;  // Color of the intersected object
    glm::vec3 normal; // Surface normal at the intersection point
    // glm::vec3 center; // The center of the object hit (for spheres)

    __host__ __device__ Hit() : hit(false), t(std::numeric_limits<float>::max()), color(0.0f) {}
};
