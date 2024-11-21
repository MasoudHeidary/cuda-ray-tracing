#pragma once

// #include <tira/image.h>
// #include <tira/parser.h>
#include <iostream>
#include <fstream>

#include "gobject.h"
using namespace GeoShape;

bool load_scene(const std::string &scene_file, std::vector<Sphere> &spheres, std::vector<Plane> &planes, Camera &camera, std::vector<Light> &lights, glm::vec3 &background, bool log_print = false)
{
    std::ifstream file(scene_file);
    if (!file.is_open())
    {
        std::cerr << "Failed to open scene file: " << scene_file << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "camera_position")
        {
            iss >> camera.position.x >> camera.position.y >> camera.position.z;
            if (log_print)
                std::cout << "camera position: " << camera.position.x << " ," << camera.position.y << " ," << camera.position.z << std::endl;
        }
        else if (key == "camera_look")
        {
            iss >> camera.look.x >> camera.look.y >> camera.look.z;
            if (log_print)
                std::cout << "camera look: " << camera.look.x << " ," << camera.look.y << " ," << camera.look.z << std::endl;
        }
        else if (key == "camera_up")
        {
            iss >> camera.up.x >> camera.up.y >> camera.up.z;
            if (log_print)
                std::cout << "camera up: " << camera.up.x << " ," << camera.up.y << " ," << camera.up.z << std::endl;
        }
        else if (key == "camera_fov")
        {
            iss >> camera.fov;
            if (log_print)
                std::cout << "camera fov: " << camera.fov << std::endl;
        }
        else if (key == "sphere")
        {
            Sphere sphere;
            glm::vec3 center, color;
            float radius;
            iss >> radius >> center.x >> center.y >> center.z >> color.x >> color.y >> color.z;
            spheres.emplace_back(center, radius, color);
            if (log_print)
                std::cout << "Loaded sphere: "
                          << "Radius: " << radius << ", "
                          << "Center: (" << center.x << ", " << center.y << ", " << center.z << "), "
                          << "Color: (" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        }
        else if (key == "light")
        {
            glm::vec3 position, color;
            iss >> position.x >> position.y >> position.z >> color.x >> color.y >> color.z;
            lights.emplace_back(Light{position, color});
            if (log_print)
                std::cout << "Loaded light: "
                          << "Position: (" << position.x << ", " << position.y << ", " << position.z << "), "
                          << "Color: (" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        }
        else if (key == "background")
        {
            iss >> background.x >> background.y >> background.z;
            if (log_print)
                std::cout << "Background color: (" << background.x << ", " << background.y << ", " << background.z << ")" << std::endl;
        }
        else if (key == "plane")
        {
            glm::vec3 point, normal, color;
            iss >> point.x >> point.y >> point.z >> normal.x >> normal.y >> normal.z >> color.x >> color.y >> color.z;
            planes.emplace_back(point, normal, color);
            if (log_print)
                std::cout << "Loaded plane: "
                          << "Point: (" << point.x << ", " << point.y << ", " << point.z << "), "
                          << "Normal: (" << normal.x << ", " << normal.y << ", " << normal.z << "), "
                          << "Color: (" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        }
    }

    return true;
}

// OBJ loader function
bool load_obj(const std::string &file_path, std::vector<Triangle> &triangles, const glm::vec3 &color)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open OBJ file: " << file_path << std::endl;
        return false;
    }

    std::vector<glm::vec3> vertices;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v")
        {
            // Parse vertex
            glm::vec3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        }
        else if (prefix == "f")
        {
            // Parse face
            int idx0, idx1, idx2;
            iss >> idx0 >> idx1 >> idx2;
            // OBJ file indices are 1-based, so we need to subtract 1 for 0-based indexing
            triangles.push_back(Triangle(vertices[idx0 - 1], vertices[idx1 - 1], vertices[idx2 - 1], color));
        }
        // else if (prefix == "f") {
        //     // Parse face with support for "//" (ignoring normals)
        //     int idx0, idx1, idx2;
        //     char dummy;  // For ignoring slashes and normal indices
        //     iss >> idx0 >> dummy >> dummy >> std::ws;
        //     iss >> idx1 >> dummy >> dummy >> std::ws;
        //     iss >> idx2 >> dummy >> dummy >> std::ws;

        //    // OBJ file indices are 1-based, so we need to subtract 1 for 0-based indexing
        //    triangles.push_back(Triangle(vertices[idx0 - 1], vertices[idx1 - 1], vertices[idx2 - 1], color));
        //}
    }
    return true;
}
