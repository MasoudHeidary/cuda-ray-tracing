#pragma once

#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>

#include "multi_thread_tool.h"

namespace cArg {
    int default_width = 1000;
    int default_height = 1000;
    bool default_shadow = true;
    std::string default_out_file = "output.png";
    int default_num_thread(void) {
        int device_num_thread = std::thread::hardware_concurrency();
        if (device_num_thread == 0)
            return 4;
        return device_num_thread;
    }

    struct CommandLineArgs {
        std::string scene_file = "";
        std::string obj_file = "";
        std::string out_file = default_out_file;
        int num_threads = default_num_thread();
        int width = default_width;
        int height = default_height;
        bool shadow = default_shadow;
    };

    std::string __str__(cArg::CommandLineArgs args) {
        return \
            "scene file: " + args.scene_file + "\n" + \
            "obj file: " + args.obj_file + "\n" + \
            "num threads: " + std::to_string(args.num_threads) + "\n" + \
            "width: " + std::to_string(args.width) + "\n" + \
            "height: " + std::to_string(args.height) + "\n" + \
            "shadow: " + (args.shadow ? "true" : "false") + "\n";
    }

    std::string __help_str__(void) {
        return \
            std::string() +\
            "--help | -h" + "\tprinting help (this prompt)\n" + \
            "--scene <scene_file_name>" + "\n" + \
            "--obj <object_file_name> [optional]" + "\n" + \
            "--threads <number_of_threads> [optional, default:max available threads]" + "\n" + \
            "--width <width_of_output_image> [optional, default:1000]" + "\n" + \
            "--height <height_of_output_image> [optional, default: 1000]" + "\n" + \
            "--no-shadow" + "\tturn of the shadow feature\n";
    }

    enum ErrorCode {
        SUCCESS = 0,

        HELP_REQUEST = 1,

        UNKNOWN_ARGUMENT,
        INCOMPLETE_ARGUMENT,
        INCOMPLETE_SCENE_FILE,
        INCOMPLETE_OBJ_FILE,
        INCOMPLETE_NUM_THREAD,
        INCOMPLETE_WIDTH,
        INCOMPLETE_HEIGHT,

        NO_SCENE_FILE_DEFINED, 
        INVALID_THREAD_NUM,
        INVALID_WIDTH,
        INVALID_HEIGHT,
        UNKNOWN_ERROR,
    };

    std::string get_error_description(ErrorCode error_code) {
        switch (error_code) {
        case ErrorCode::UNKNOWN_ARGUMENT:
            return TO_STRING(ErrorCode::UNKNOWN_ARGUMENT) + std::string(", the provided --flag is not valid");
        case ErrorCode::INCOMPLETE_ARGUMENT:
            return TO_STRING(ErrorCode::INCOMPLETE_ARGUMENT) + std::string(", one of flags provided has no valid value");
        //TODO: complete the error prompts
        //TODO: add custumizing error message possible
        case ErrorCode::NO_SCENE_FILE_DEFINED:
            return TO_STRING(ErrorCode::NO_SCENE_FILE_DEFINED);
        case ErrorCode::INVALID_THREAD_NUM:
            return TO_STRING(ErrorCode::INVALID_THREAD_NUM);
        case ErrorCode::INVALID_WIDTH:
            return TO_STRING(ErrorCode::INVALID_WIDTH);
        case ErrorCode::INVALID_HEIGHT:
            return TO_STRING(ErrorCode::INVALID_HEIGHT);
        
        case ErrorCode::UNKNOWN_ERROR:
        default:
            return TO_STRING(ErrorCode::UNKNOWN_ERROR);
        }
    }


    ErrorCode validate_arguments(CommandLineArgs args) {
        if (args.scene_file == "")
            return ErrorCode::NO_SCENE_FILE_DEFINED;
        if (args.num_threads <= 0)
            return ErrorCode::INVALID_THREAD_NUM;
        if (args.height <= 0)
            return ErrorCode::INVALID_HEIGHT;
        if (args.width <= 0)
            return ErrorCode::INVALID_WIDTH;
        return ErrorCode::SUCCESS;
    }

    ErrorCode parse_arguments(CommandLineArgs* args, int argc, char* argv[]) {        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" or arg == "-h") {
                return ErrorCode::HELP_REQUEST;
            }

            if (arg == "--scene") {
                if (i + 1 >= argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->scene_file = argv[++i];
            }
            else if (arg == "--obj") {
                if (i + 1 >= argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->obj_file = argv[++i];
            }
            else if (arg == "--out") {
                if (i + 1 > argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->out_file = argv[++i];
            }
            else if (arg == "--threads") {
                if (i + 1 >= argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->num_threads = std::atoi(argv[++i]);
            }
            else if (arg == "--width") {
                if (i + 1 >= argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->width = std::atoi(argv[++i]);
            }
            else if (arg == "--height") {
                if (i + 1 >= argc)
                    return ErrorCode::INCOMPLETE_ARGUMENT;
                args->height = std::atoi(argv[++i]);
            }
            else if (arg == "--no-shadow") {
                args->shadow = false;
            }
            else {
                /*std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
                exit(1);*/
                //return ErrorCode::INCOMPLETE_ARGUMENT;
                return ErrorCode::UNKNOWN_ARGUMENT;
            }
        }

        return validate_arguments(*args);
    }

}
