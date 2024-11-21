
#include "gobject.h"
using namespace GeoShape;


const float epsilon = 0.001f;
bool is_in_shadow(const glm::vec3& point, const glm::vec3& normal, const Light& light, const std::vector<Sphere>& spheres) {
    // Create a shadow ray from a point slightly above the surface to the light source
    glm::vec3 light_dir = glm::normalize(light.position - point);
    glm::vec3 shadow_origin = point + epsilon * normal;  // Offset by epsilon along the normal
    Ray shadow_ray(shadow_origin, light_dir);

    // Distance to the light
    float light_distance = glm::length(light.position - point);

    // Check if any object is between the point and the light
    for (const Sphere& sphere : spheres) {
        Hit hit_record;
        if (sphere.intersect(shadow_ray, hit_record)) {
            // If the intersection point is closer than the light, it's in shadow
            if (hit_record.t < light_distance) {
                return true;  // There is an object between the point and the light
            }
        }
    }

    return false;  // No object blocks the light
}



glm::vec3 compute_lighting(const glm::vec3& point, const glm::vec3& normal, const std::vector<Light>& lights) {
    glm::vec3 light_color(0.0f);
    //glm::vec3 ambient_light(0.2f, 0.2f, 0.2f);
    //glm::vec3 light_color = ambient_light;

    for (const auto& light : lights) {
        glm::vec3 light_dir = glm::normalize(light.position - point);
        float diffuse_intensity = std::max(glm::dot(normal, light_dir), 0.0f);
        light_color += diffuse_intensity * light.color;
    }

    return glm::clamp(light_color, 0.0f, 1.0f);
}



glm::vec3 compute_lighting_and_shadow(const glm::vec3& point, const glm::vec3& normal, const std::vector<Light>& lights, const std::vector<Sphere>& spheres) {
    glm::vec3 light_color(0.0f);
    //glm::vec3 ambient_light(0.2f, 0.2f, 0.2f);
    //glm::vec3 light_color = ambient_light;

    for (const auto& light : lights) {
        if (is_in_shadow(point, normal, light, spheres)) {
            continue;
        }

        glm::vec3 light_dir = glm::normalize(light.position - point);
        float diffuse_intensity = std::max(glm::dot(normal, light_dir), 0.0f);
        light_color += diffuse_intensity * light.color;
    }

    return glm::clamp(light_color, 0.0f, 1.0f); 
}