# cuda-ray-tracing
A high-performance ray tracing engine implemented in CUDA, GPU Course, University of Houston


g++ -O3 -o main main.cpp $(pkg-config --cflags --libs opencv4) -I/usr/include/glm && ./main --scene spheramid.scene 

g++ -O3 -o main main.cpp `pkg-config --cflags --libs opencv4` && ./main --scene spheramid.scene 

g++ -O3 -o out/main main.cpp `pkg-config --cflags --libs opencv4` && cd out && ./main --scene spheramid.scene && cd ..



nvcc -O3 -o out/cuda main.cu `pkg-config --cflags --libs opencv4` --disable-warnings  && cd out && ./cuda --scene mesh.scene --obj subdiv.obj && cd ..