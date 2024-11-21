
#include <iostream>

#include "setting.h"


// tira fix bug
// #include <cstddef> // for std::byte
// std::byte myByte = std::byte(0xFF);
// #define byte windows_byte
// #include <Windows.h>
// #undef byte
// #include <cstddef> 

#include <opencv2/opencv.hpp>


#include <iostream>
#include <vector>
#include <cstdlib>  // run command on terminal
#include <string>

#include <chrono> // For high-resolution clock
#include <iomanip> // For setting precision

#include <thread>

// #include <tira/image.h>
// #include <tira/parser.h>

//#include "ray.h"
//#include "hit.h"
#include "loader.h"
#include "compute.h"
#include "gobject.h"
#include "command_line_tool.h"


using namespace GeoShape;
using namespace std::chrono; // For convenient timing functions







int _main(int argc, char* argv[]) {
    cArg::CommandLineArgs args;
    cArg::ErrorCode error_code = cArg::parse_arguments(&args, argc, argv);
    
    if (error_code) {
        if (error_code == cArg::ErrorCode::HELP_REQUEST) {
            std::cout << cArg::__help_str__() << std::endl;
            return 0;
        }
        else {
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
    const int num_threads = args.num_threads;

     

    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    std::vector<Triangle> triangles;
    std::vector<Light> lights;
    Camera camera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), 0.0f);
    glm::vec3 background(0.0f, 0.0f, 0.0f);
    // tira::image<unsigned char> image(image_width, image_height, 3);
    cv::Mat image(image_height, image_width, CV_8UC3);


    if (!load_scene(scene_file_name, spheres, planes, camera, lights, background)) {
        std::cerr << "Error loading scene!\t" << scene_file_name << std::endl;
        return -1;
    }

    if (obj_file_name != "") {
        if (!load_obj(obj_file_name, triangles, glm::vec3(1.0, 1.0, 1.0))) {
            std::cerr << "Error loading OBJ file!\t" << obj_file_name << std::endl;
            return -1;
        }
    }

    // std::cout << "spheres: " << spheres.size() << std::endl;
    // std::cout << "lights: " << lights.size() << std::endl;
    // std::cout << "camera: " << camera.fov << std::endl;


    auto start_time = high_resolution_clock::now();
    //double total_lighting_time = 0.0;
    //double total_pixel_time = 0.0;

    std::vector<std::thread> threads;
    std::vector<double> thread_execution_times(num_threads, 0.0);

    // chunk render
    auto render_chunk = [&](int start_row, int end_row, int thread_id) {
        auto start_time = std::chrono::high_resolution_clock::now();  // Start timing the thread


        // Render loop
        for (int j = start_row; j < end_row; ++j) {
            //std::cout << "Rendering row " << "[" << j << "/" << image_height << "]" << std::endl;
            for (int i = 0; i < image_width; ++i) {
                //auto pixel_start_time = high_resolution_clock::now();  // Start timing pixel

                float u = float(image_width - 1 - i) / float(image_width - 1);
                float v = float(image_height - 1 - j) / (image_height - 1);  // Flip the v coordinate

                Ray ray = camera.get_ray(u, v);
                glm::vec3 pixel_color = background;
                Hit closest_hit_record;  // Store the closest hit information
                bool hit_anything = false;
                float closest_t = std::numeric_limits<float>::max();  // Set to a very large number initially

                // Iterate through each sphere to find the closest intersection
                for (const Sphere& sphere : spheres) {
                    Hit hit_record;
                    if (sphere.intersect(ray, hit_record)) {
                        if (hit_record.t < closest_t) {
                            closest_t = hit_record.t;
                            closest_hit_record = hit_record;
                            hit_anything = true;
                        }
                    }
                }

                // Check plane intersections
                for (const Plane& plane : planes) {
                    Hit hit_record;
                    if (plane.intersect(ray, hit_record)) {
                        if (hit_record.t < closest_t) {
                            closest_t = hit_record.t;
                            closest_hit_record = hit_record;
                            hit_anything = true;
                        }
                    }
                }

                // Check triangle intersections
                for (const Triangle& triangle : triangles) {
                    Hit hit_record;
                    if (triangle.intersect(ray, hit_record)) {
                        if (hit_record.t < closest_t) {
                            closest_t = hit_record.t;
                            closest_hit_record = hit_record;
                            hit_anything = true;
                        }
                    }
                }

                // Compute lighting and color based on the closest hit
                if (hit_anything) {
                    glm::vec3 point = ray.at(closest_hit_record.t);

                    // Measure lighting calculation time
                    //auto lighting_start_time = high_resolution_clock::now();
                    glm::vec3 lighting;
                    if (shadow_enable)
                        lighting = compute_lighting_and_shadow(point, closest_hit_record.normal, lights, spheres);
                    else
                        lighting = compute_lighting(point, closest_hit_record.normal, lights);
                    //auto lighting_end_time = high_resolution_clock::now();

                    // Add to total lighting time
                    //total_lighting_time += duration<double>(lighting_end_time - lighting_start_time).count();

                    pixel_color = glm::clamp(closest_hit_record.color * lighting, 0.0f, 1.0f);
                }
                else {
                    pixel_color = background;  // If no intersection, use the background color
                }

                // Set pixel color in the image
                // image(i, j, 0) = static_cast<unsigned char>(255.99 * pixel_color.r);
                // image(i, j, 1) = static_cast<unsigned char>(255.99 * pixel_color.g);
                // image(i, j, 2) = static_cast<unsigned char>(255.99 * pixel_color.b);
                image.at<cv::Vec3b>(j, i)[0] = static_cast<unsigned char>(255.0 * pixel_color.b);  // Blue channel
                image.at<cv::Vec3b>(j, i)[1] = static_cast<unsigned char>(255.0 * pixel_color.g);  // Green channel
                image.at<cv::Vec3b>(j, i)[2] = static_cast<unsigned char>(255.0 * pixel_color.r);  // Red channel


                // Measure pixel calculation time
                //auto pixel_end_time = high_resolution_clock::now();
                //total_pixel_time += duration<double>(pixel_end_time - pixel_start_time).count();
            }
        }


        auto end_time = std::chrono::high_resolution_clock::now();  // End timing the thread
        thread_execution_times[thread_id] = std::chrono::duration<double>(end_time - start_time).count();  // Store the duration

    //end of chunk render
    };

     
    // Launch threads
    int rows_per_thread = image_height / num_threads;  // Divide the rows evenly among threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? image_height : start_row + rows_per_thread;
        threads.push_back(std::thread(render_chunk, start_row, end_row, t));
    }

    // Join threads (wait for them to finish)
    for (std::thread& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    for (unsigned int t = 0; t < num_threads; ++t) {
        // std::cout << "Thread " << t << " execution time: " << thread_execution_times[t] << " seconds" << std::endl;
    }
    

    // End total render time
    auto end_time = high_resolution_clock::now();
    double total_render_time = duration<double>(end_time - start_time).count();

    // Output profiling results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total render time: " << total_render_time << " seconds" << std::endl;
    //std::cout << "Average time per pixel: " << total_pixel_time / (image_width * image_height) << " seconds" << std::endl;
    //std::cout << "Total lighting calculation time: " << total_lighting_time << " seconds" << std::endl;
    //std::cout << "Average lighting calculation time per pixel: " << total_lighting_time / (image_width * image_height) << " seconds" << std::endl;



    // open the picture after saving
    // image.save(output_file_name); 
    if (cv::imwrite(output_file_name, image)) {
        std::cout << "Image saved successfully to " << output_file_name << std::endl;
    } else {
        std::cout << "Error: Could not save the image!" << std::endl;
    }

    // Assuming image is a cv::Mat with 3 channels (RGB) and CV_8UC3 type
    // for (int i = 0; i < image_height; ++i) {
    //     for (int j = 0; j < image_width; ++j) {
    //         // Access the pixel at (i, j)
    //         cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);

    //         // Extract RGB values
    //         unsigned char blue = pixel[0];  // Blue channel
    //         unsigned char green = pixel[1]; // Green channel
    //         unsigned char red = pixel[2];   // Red channel

    //         // Print the RGB values of the pixel
    //         std::cout << "Pixel (" << i << ", " << j << ") - "
    //                 << "R: " << static_cast<int>(red) << " "
    //                 << "G: " << static_cast<int>(green) << " "
    //                 << "B: " << static_cast<int>(blue) << std::endl;
    //     }
    // }

    // system(output_file_name.c_str());

    return 0;
}