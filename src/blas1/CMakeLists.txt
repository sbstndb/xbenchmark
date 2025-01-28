add_executable(blas1_vector vector.cpp)
target_link_libraries(blas1_vector PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(blas1_fma fma.cpp)
target_link_libraries(blas1_fma PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(blas1_complex_operation complex.cpp)
target_link_libraries(blas1_complex_operation PRIVATE ${GLOBAL_DEPENDENCIES})

add_executable(blas1_add_scalar add_scalar.cpp)
target_link_libraries(blas1_add_scalar PRIVATE ${GLOBAL_DEPENDENCIES})


add_executable(blas1_logic logic.cpp)
target_link_libraries(blas1_logic PRIVATE ${GLOBAL_DEPENDENCIES})



