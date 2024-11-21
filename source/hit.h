#pragma once
#include <glm/glm.hpp>

struct Hit {
    float t;               // Distance from the ray origin to the intersection point
    glm::vec3 color;       // Color of the intersected object
    glm::vec3 normal;      // Surface normal at the intersection point
    glm::vec3 center;    // The center of the object hit (for spheres)

    Hit() : t(0), color(0.0f) {}

};


