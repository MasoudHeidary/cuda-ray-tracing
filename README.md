# cuda-ray-tracing
A high-performance ray tracing engine implemented in CUDA, GPU Course, University of Houston


g++ -O3 -o main main.cpp $(pkg-config --cflags --libs opencv4) -I/usr/include/glm && ./main --scene spheramid.scene 