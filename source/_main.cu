#include <iostream>
#include <vector>
#include <cstdlib> // run command on terminal
#include <string>
#include <chrono>  // For high-resolution clock
#include <iomanip> // For setting precision
#include <thread>

#include "setting.h"
#include "command_line_tool.h"
#include "loader.h"
// #include "gobject.h"
#include "compute.h"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace GeoShape;
using namespace std::chrono; // For convenient timing functions


__global__ void render_image(
    Camera camera,
    glm::vec3 *image,
    const int image_width,
    const int image_height,
    Sphere *spheres,
    int num_spheres,
    Plane *planes,
    int num_plains,
    Triangle *triangles,
    int num_triangles,
    Light *lights,
    int num_lights,
    const glm::vec3 background)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_width && j < image_height)
    {
        float u = float(image_width - 1 - i) / float(image_width - 1);
        float v = float(image_height - 1 - j) / float(image_height - 1); // Flip the v coordinate

        // Create the ray from the camera
        // Ray ray(camera_position, direction_of_camera);  // Define your camera position and direction
        Ray ray = camera.get_ray(u, v);

        glm::vec3 pixel_color = background;
        Hit closest_hit_record;
        closest_hit_record.hit = false;

        // Iterate through spheres for intersection (simple sphere intersection for now)
        Hit hit_record;
        for (int i = 0; i < num_spheres; i++)
        {
            // if (intersect_sphere(ray, sphere.center, sphere.radius, hit_record)) {
            if (spheres[i].intersect(ray, hit_record))
            {
                if (hit_record.t < closest_hit_record.t)
                {
                    closest_hit_record = hit_record;
                }
            }
        }

        for (int i = 0; i < num_plains; i++)
        {
            if (planes[i].intersect(ray, hit_record))
            {
                if (hit_record.t < closest_hit_record.t)
                {
                    closest_hit_record = hit_record;
                }
            }
        }

        for (int i=0; i<num_triangles; i++)
        {
            if(triangles[i].intersect(ray, hit_record))
            {
                if(hit_record.t < closest_hit_record.t)
                {
                    closest_hit_record = hit_record;
                }
            }
        }

        // Compute lighting and color based on the closest hit
        if (closest_hit_record.hit)
        {
            glm::vec3 point = ray.at(closest_hit_record.t);
            glm::vec3 lighting = compute_lighting(point, closest_hit_record.normal, lights, num_lights);
            // if (i == 500 && j == 500) {
            //     printf("num lights: %d\n", num_lights);
            //     printf("lightning color: %f %f %f\n", lighting.r, lighting.g, lighting.b);
            // }
            pixel_color = glm::clamp(closest_hit_record.color * lighting, 0.0f, 1.0f);
        }
        // else {
        //     pixel_color = background;  // If no intersection, use the background color
        // }

        // Set pixel color in the image (CUDA memory access)
        image[j * image_width + i] = pixel_color;

        // if (i == 1 && j == 1)
        // {
        //     printf("Ray origin: (%f, %f, %f)\n", ray.origin.x, ray.origin.y, ray.origin.z);
        //     printf("Ray direction: (%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
        //     printf("Closest hit: %d, T: %f\n", closest_hit_record.hit, closest_hit_record.t);
        //     printf("Closest hit color: %f %f %f\n", closest_hit_record.color.r, closest_hit_record.color.g, closest_hit_record.color.b);
        //     printf("plane color: %f %f %f\n", planes[0].color.r, planes[0].color.g, planes[0].color.b);
        //     printf("pixel color: %f %f %f\n", pixel_color.r, pixel_color.g, pixel_color.b);
        // }
    }
}

int _main(int argc, char *argv[])
{
    cArg::CommandLineArgs args;
    cArg::ErrorCode error_code = cArg::parse_arguments(&args, argc, argv);

    if (error_code)
    {
        if (error_code == cArg::ErrorCode::HELP_REQUEST)
        {
            std::cout << cArg::__help_str__() << std::endl;
            return 0;
        }
        else
        {
            std::cerr << "CommandLineArgs Error, " << cArg::get_error_description(error_code) << std::endl;
            return error_code;
        }
    }
    std::cout << cArg::__str__(args) << std::endl;

    const std::string scene_file_name = args.scene_file;
    const std::string obj_file_name = args.obj_file;
    const std::string output_file_name = args.out_file;
    const int image_width = args.width;
    const int image_height = args.height;
    const bool shadow_enable = args.shadow;
    const bool cuda_enable = args.cuda;
    const int num_threads = args.num_threads;

    cv::Mat image(image_height, image_width, CV_8UC3);
    glm::vec3 background(0.0f, 0.0f, 0.0f);
    Camera camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), 0.0f);

    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    std::vector<Triangle> triangles;
    std::vector<Light> lights;

    if (!load_scene(scene_file_name, spheres, planes, camera, lights, background))
    {
        std::cerr << "Error loading scene!\t" << scene_file_name << std::endl;
        return -1;
    }

    if (obj_file_name != "")
    {
        if (!load_obj(obj_file_name, triangles, glm::vec3(1.0, 1.0, 1.0)))
        {
            std::cerr << "Error loading OBJ file!\t" << obj_file_name << std::endl;
            return -1;
        }
    }

    std::cout << "lights: " << lights.size() << std::endl;
    std::cout << "spheres: " << spheres.size() << std::endl;
    std::cout << "planes: " << planes.size() << std::endl;
    std::cout << "triangle: " << triangles.size() << std::endl;

    auto start_time = high_resolution_clock::now();

    if (true)
    {
        glm::vec3 *d_image;
        cudaMalloc(&d_image, image_width * image_height * sizeof(glm::vec3));

        // Define spheres and lights
        // std::vector<Sphere> spheres;
        // std::vector<Light> lights;

        // // Add some spheres and lights
        // spheres.push_back(Sphere(glm::vec3(0.0f, 0.0f, -5.0f), 1.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
        // lights.push_back(Light{glm::vec3(5.0f, 5.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)});

        // Allocate memory for spheres and lights on the device
        Sphere *d_spheres;
        cudaMalloc(&d_spheres, spheres.size() * sizeof(Sphere));
        Plane *d_planes;
        cudaMalloc(&d_planes, planes.size() * sizeof(Plane));
        Triangle *d_triangles;
        cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle));
        Light *d_lights;
        cudaMalloc(&d_lights, lights.size() * sizeof(Light));

        // Copy spheres and lights data to device
        cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
        cudaMemcpy(d_planes, planes.data(), planes.size() * sizeof(Plane), cudaMemcpyHostToDevice);
        cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

        // Define CUDA block and grid sizes
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((image_width + 15) / 16, (image_height + 15) / 16);

        // Launch the kernel
        render_image<<<numBlocks, threadsPerBlock>>>(
            camera,
            d_image,
            image_width,
            image_height,
            d_spheres,
            spheres.size(),
            d_planes,
            planes.size(),
            d_triangles,
            triangles.size(),
            d_lights,
            lights.size(),
            background);

        // Check for CUDA errors
        cudaDeviceSynchronize();

        // Copy the result from device to host
        glm::vec3 *h_image = new glm::vec3[image_width * image_height];
        cudaMemcpy(h_image, d_image, image_width * image_height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

        cudaFree(d_image);
        cudaFree(d_spheres);
        cudaFree(d_planes);
        cudaFree(d_triangles);
        cudaFree(d_lights);

        // End total render time
        auto end_time = high_resolution_clock::now();
        double total_render_time = duration<double>(end_time - start_time).count();

        // Output profiling results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total render time: " << total_render_time << " seconds" << std::endl;

        // Save or display the image (e.g., using OpenCV or another library)
        // Convert h_image (glm::vec3*) to cv::Mat
        cv::Mat image(image_height, image_width, CV_8UC3);

        for (int j = 0; j < image_height; ++j)
        {
            for (int i = 0; i < image_width; ++i)
            {
                glm::vec3 color = h_image[j * image_width + i];
                image.at<cv::Vec3b>(j, i)[0] = static_cast<unsigned char>(255.0f * color.b); // Blue channel
                image.at<cv::Vec3b>(j, i)[1] = static_cast<unsigned char>(255.0f * color.g); // Green channel
                image.at<cv::Vec3b>(j, i)[2] = static_cast<unsigned char>(255.0f * color.r); // Red channel
                // std::cout << image.at<cv::Vec3b>(j, i) << std::endl;
            }
        }

        // Save the image to file
        if (cv::imwrite(output_file_name, image))
        {
            std::cout << "Image saved successfully to " << output_file_name << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not save the image!" << std::endl;
        }

        // Clean up
        delete[] h_image;
    }

    // for (int j = 0; j < image_height; ++j) {
    //     for (int i = 0; i < image_width; ++i) {

    //         float u = float(image_width - 1 - i) / float(image_width - 1);
    //         float v = float(image_height - 1 - j) / (image_height - 1);  // Flip the v coordinate

    //         Ray ray = camera.get_ray(u, v);
    //         glm::vec3 pixel_color = background;
    //         Hit closest_hit_record;  // Store the closest hit information
    //         // bool hit_anything = false;
    //         // float closest_t = std::numeric_limits<float>::max();  // Set to a very large number initially

    //         // Iterate through each sphere to find the closest intersection
    //         for (const Sphere& sphere : spheres) {
    //             Hit hit_record;
    //             if (sphere.intersect(ray, hit_record)) {
    //                 if (hit_record.t < closest_hit_record.t) {
    //                     // closest_hit_record.t = hit_record.t;
    //                     closest_hit_record = hit_record;
    //                     // hit_anything = true;
    //                     closest_hit_record.hit = true;
    //                 }
    //             }
    //         }

    //         // // Check plane intersections
    //         // for (const Plane& plane : planes) {
    //         //     Hit hit_record;
    //         //     if (plane.intersect(ray, hit_record)) {
    //         //         if (hit_record.t < closest_t) {
    //         //             closest_t = hit_record.t;
    //         //             closest_hit_record = hit_record;
    //         //             hit_anything = true;
    //         //         }
    //         //     }
    //         // }

    //         // // Check triangle intersections
    //         // for (const Triangle& triangle : triangles) {
    //         //     Hit hit_record;
    //         //     if (triangle.intersect(ray, hit_record)) {
    //         //         if (hit_record.t < closest_t) {
    //         //             closest_t = hit_record.t;
    //         //             closest_hit_record = hit_record;
    //         //             hit_anything = true;
    //         //         }
    //         //     }
    //         // }

    //         // Compute lighting and color based on the closest hit
    //         if (closest_hit_record.hit) {
    //             glm::vec3 point = ray.at(closest_hit_record.t);

    //             glm::vec3 lighting;
    //             // if (shadow_enable)
    //             //     lighting = compute_lighting_and_shadow(point, closest_hit_record.normal, lights, spheres);
    //             // else
    //                 lighting = compute_lighting(point, closest_hit_record.normal, lights);

    //             pixel_color = glm::clamp(closest_hit_record.color * lighting, 0.0f, 1.0f);
    //         }
    //         else {
    //             pixel_color = background;  // If no intersection, use the background color
    //         }

    //         // Set pixel color in the image
    //         image.at<cv::Vec3b>(j, i)[0] = static_cast<unsigned char>(255.0 * pixel_color.b);  // Blue channel
    //         image.at<cv::Vec3b>(j, i)[1] = static_cast<unsigned char>(255.0 * pixel_color.g);  // Green channel
    //         image.at<cv::Vec3b>(j, i)[2] = static_cast<unsigned char>(255.0 * pixel_color.r);  // Red channel
    //     }
    // }

    // // End total render time
    // auto end_time = high_resolution_clock::now();
    // double total_render_time = duration<double>(end_time - start_time).count();

    // // Output profiling results
    // std::cout << std::fixed << std::setprecision(6);
    // std::cout << "Total render time: " << total_render_time << " seconds" << std::endl;

    // if (cv::imwrite(output_file_name, image)) {
    //     std::cout << "Image saved successfully to " << output_file_name << std::endl;
    // } else {
    //     std::cout << "Error: Could not save the image!" << std::endl;
    // }

    return 0;
}