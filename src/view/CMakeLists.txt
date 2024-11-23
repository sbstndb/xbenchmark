add_executable(view_all view_all.cpp)
target_link_libraries(view_all PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(view_stride view_stride.cpp)
target_link_libraries(view_stride PRIVATE ${GLOBAL_DEPENDENCIES})

