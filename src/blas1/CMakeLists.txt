add_executable(blas1_simple_operation vector.cpp)
target_link_libraries(blas1_simple_operation PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(blas1_fma fma.cpp)
target_link_libraries(blas1_fma PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(blas1_complex_operation complex.cpp)
target_link_libraries(blas1_complex_operation PRIVATE ${GLOBAL_DEPENDENCIES})
