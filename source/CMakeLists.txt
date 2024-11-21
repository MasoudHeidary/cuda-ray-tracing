# CMakeList.txt : CMake project for mproject, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (mproject "mproject.cpp" "mproject.h" "setting.h" "_main.h" "ray.h" "hit.h" "gobject.h" "loader.h" "compute.h" "command_line_tool.h" "multi_thread_tool.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET mproject PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
configure_file(basic.scene basic.scene COPYONLY)
configure_file(spheramid.scene spheramid.scene COPYONLY)
configure_file(mesh.scene mesh.scene COPYONLY)
configure_file(subdiv.obj subdiv.obj COPYONLY)
