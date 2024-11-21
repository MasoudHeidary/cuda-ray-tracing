// mproject.cpp : Defines the entry point for the application.
//

#include "mproject.h"
using namespace std;

#include "setting.h"


#if CHECK_CMAKE

    // tira fix bug
    #include <cstddef> // for std::byte
    std::byte myByte = std::byte(0xFF);
    #define byte windows_byte
    #include <Windows.h>
    #undef byte
    #include <cstddef> 

    #include <glm/glm.hpp> // Include GLM headers
    #include <tira/image.h>

    int main() {
        glm::vec3 _vector;
        std::cout << "glm functioning" << std::endl;

        tira::image<unsigned char> _image;
        std::cout << "tira functioning" << std::endl;

        
        return 0;
}
#else
    #include"_main.h"
    int main(int argc, char* argv[]) {
        return _main(argc, argv);
        }
#endif

