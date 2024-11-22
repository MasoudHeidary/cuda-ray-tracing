#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __host__ __device__ Ray(const glm::vec3& o, const glm::vec3& d) : origin(o), direction(glm::normalize(d)) {}


    __host__ __device__ glm::vec3 at(float t) const {
        return origin + t * direction;
    }
};
