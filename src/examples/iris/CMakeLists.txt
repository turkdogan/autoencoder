cmake_minimum_required(VERSION 2.8)

project(iris_example)
 


file(GLOB SRC 
    "../../nn/*.h"
    "../../nn/*.cpp"
    "iris_example.cpp"
)

configure_file(iris.data iris.data COPYONLY)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(iris_example ${SRC})
