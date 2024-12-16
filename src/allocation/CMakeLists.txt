add_executable(allocation allocation.cpp)
target_link_libraries(allocation PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(loic loic.cpp)
target_link_libraries(loic PRIVATE ${GLOBAL_DEPENDENCIES})


