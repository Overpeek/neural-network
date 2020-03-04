cmake_minimum_required(VERSION 3.16)

project("neural-network")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Program Files (x86)")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Program Files")
find_package(Armadillo REQUIRED)

message(${ARMADILLO_INCLUDE_DIR})
message(${ARMADILLO_LIBRARIES})

add_executable("perceptron" 
    "src/main.cpp"
    "src/layer.hpp"
    "src/helper.hpp"
)
target_include_directories("perceptron" PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries("perceptron" PRIVATE ${ARMADILLO_LIBRARIES})
target_precompile_headers("perceptron" PRIVATE 
<iostream> 
<armadillo>
<random>
)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()