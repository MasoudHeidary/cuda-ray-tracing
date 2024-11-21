#pragma once
#include <glm/glm.hpp>
#include "ray.h"
#include "hit.h"

// gobject (graphical object)

// GeoShape (Geometrical Object)
namespace GeoShape {
    
    struct Light {
        glm::vec3 position;
        glm::vec3 color;
    };

    class Camera {
    public:
        glm::vec3 position;
        glm::vec3 look;
        glm::vec3 up;
        float fov;

        Camera(const glm::vec3& pos, const glm::vec3& look_dir, const glm::vec3& up_dir, float field_of_view)
            : position(pos), look(glm::normalize(look_dir - pos)), up(glm::normalize(up_dir)), fov(field_of_view) {}

        Ray get_ray(float u, float v) const {
            float aspect_ratio = 1.0f; // Adjust this as needed
            float theta = glm::radians(fov);
            float h = glm::tan(theta / 2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            glm::vec3 w = glm::normalize(position - look);
            glm::vec3 u_vec = glm::normalize(glm::cross(up, w));
            glm::vec3 v_vec = glm::cross(w, u_vec);

            glm::vec3 lower_left = position - (viewport_width / 2.0f) * u_vec - (viewport_height / 2.0f) * v_vec - w;
            glm::vec3 horizontal = viewport_width * u_vec;
            glm::vec3 vertical = viewport_height * v_vec;

            glm::vec3 direction = glm::normalize(lower_left + u * horizontal + v * vertical - position);
            return Ray(position, direction);
        }
    };


    struct Sphere {
        glm::vec3 center;
        float radius;
        glm::vec3 color;

            // Default constructor
        Sphere() : center(glm::vec3(0.0f, 0.0f, 0.0f)), radius(1.0f), color(glm::vec3(1.0f, 1.0f, 1.0f)) {}


        Sphere(const glm::vec3& c, float r, const glm::vec3& col) : center(c), radius(r), color(col) {}

        bool intersect(const Ray& ray, Hit& hit) const {
            glm::vec3 oc = ray.origin - center;
            float a = glm::dot(ray.direction, ray.direction);
            float b = 2.0 * glm::dot(oc, ray.direction);
            float c = glm::dot(oc, oc) - radius * radius;
            float discriminant = b * b - 4 * a * c;

            if (discriminant > 0) {
                float t = (-b - sqrt(discriminant)) / (2.0 * a);
                if (t < 0) t = (-b + sqrt(discriminant)) / (2.0 * a);  // Check second solution if first is behind

                if (t >= 0) {
                    hit.t = t;
                    hit.color = color;
                    glm::vec3 point = ray.at(t);
                    hit.normal = glm::normalize(point - center);  // Normal at the intersection point
                    hit.center = center;  // Store the center of the sphere
                    return true;
                }
            }
            return false;
        }
    };


    class Plane {
    public:
        glm::vec3 point;   // A point on the plane
        glm::vec3 normal;  // The normal vector of the plane
        glm::vec3 color;   // The color of the plane

        Plane(const glm::vec3& p, const glm::vec3& n, const glm::vec3& col)
            : point(p), normal(glm::normalize(n)), color(col) {}

        bool intersect(const Ray& ray, Hit& hit) const {
            // Compute the denominator D�N (ray direction dot plane normal)
            float denom = glm::dot(ray.direction, normal);

            // If denom is close to zero, the ray is parallel to the plane
            if (fabs(denom) > 1e-6) {  // 1e-6 is a small value to avoid precision issues
                // Compute the intersection distance (t)
                float t = glm::dot(point - ray.origin, normal) / denom;

                // Check if the intersection is in front of the ray (positive t)
                if (t >= 0) {
                    hit.t = t;
                    hit.color = color;
                    hit.normal = normal;
                    return true;
                }
            }
            return false;
        }
    };


    struct Triangle {
        glm::vec3 v0, v1, v2;  // Triangle vertices
        glm::vec3 normal;      // Precomputed normal for the triangle
        glm::vec3 color;       // Color for the triangle (you can use a default value)

        Triangle(const glm::vec3& _v0, const glm::vec3& _v1, const glm::vec3& _v2, const glm::vec3& col)
            : v0(_v0), v1(_v1), v2(_v2), color(col) {
            normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));  // Precompute the normal
        }

        bool intersect(const Ray& ray, Hit& hit) const {
            const float EPSILON = 1e-8;
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 h = glm::cross(ray.direction, edge2);
            float a = glm::dot(edge1, h);
            if (fabs(a) < EPSILON) {
                return false;  // Ray is parallel to the triangle
            }
            float f = 1.0f / a;
            glm::vec3 s = ray.origin - v0;
            float u = f * glm::dot(s, h);
            if (u < 0.0 || u > 1.0) {
                return false;
            }
            glm::vec3 q = glm::cross(s, edge1);
            float v = f * glm::dot(ray.direction, q);
            if (v < 0.0 || u + v > 1.0) {
                return false;
            }
            float t = f * glm::dot(edge2, q);
            if (t > EPSILON) {  // Ray intersection
                hit.t = t;
                hit.normal = normal;
                hit.color = color;  // Set the triangle's color
                return true;
            }
            else {
                return false;  // No hit, ray is parallel to the triangle
            }
        }
    };


}


