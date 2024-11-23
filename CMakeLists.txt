cmake_minimum_required(VERSION 3.28)

project(XBenchmarks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(benchmark REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -mavx2 -Ofast -funroll-loops -ftree-vectorize -g")


if(XBENCHMARK_USE_XTENSOR)
	find_package(xtensor REQUIRED)
	add_compile_definitions(XBENCHMARK_USE_XTENSOR)
endif()

if(XBENCHMARK_USE_IMMINTRIN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXS_FLAGS} -mavx2")
	add_compile_definitions(XBENCHMARK_USE_IMMINTRIN)
endif()

if(XBENCHMARK_USE_EIGEN)
	find_package(eigen3)
	add_compile_definitions(XBENCHMARK_USE_EIGEN)
endif()



include_directories(include)
add_subdirectory(src)


# Ajouter l'exécutable
add_executable(VectorSumBenchmark src/vector.cpp)
add_executable(VectorFmaBenchmark src/fma.cpp)
add_executable(VectorComplexBenchmark src/complex.cpp)
add_executable(ViewBenchmark src/view_all.cpp)
add_executable(ViewStrideBenchmark src/view_stride.cpp)
add_executable(FindBenchmark src/find.cpp)



# Lier Google Benchmark à l'exécutable
target_link_libraries(VectorSumBenchmark benchmark::benchmark)
target_link_libraries(VectorFmaBenchmark benchmark::benchmark)
target_link_libraries(VectorComplexBenchmark benchmark::benchmark)
target_link_libraries(ViewBenchmark benchmark::benchmark)
target_link_libraries(ViewStrideBenchmark benchmark::benchmark)
target_link_libraries(FindBenchmark benchmark::benchmark)



if (XBENCHMARK_USE_XTENSOR)
	target_link_libraries(VectorSumBenchmark xtensor xtensor::use_xsimd xtensor::optimize)
	target_link_libraries(VectorFmaBenchmark xtensor xtensor::use_xsimd xtensor::optimize)
	target_link_libraries(VectorComplexBenchmark xtensor xtensor::use_xsimd xtensor::optimize)
        target_link_libraries(ViewBenchmark xtensor xtensor::use_xsimd xtensor::optimize)
	target_link_libraries(ViewStrideBenchmark xtensor xtensor::use_xsimd xtensor::optimize)
	target_link_libraries(FindBenchmark xtensor xtensor::use_xsimd xtensor::optimize)



endif()

# Optimisation (optionnel) : Ajustez si nécessaire
target_compile_options(VectorSumBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
target_compile_options(VectorFmaBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
target_compile_options(VectorComplexBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
target_compile_options(ViewBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
target_compile_options(ViewStrideBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
target_compile_options(FindBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)






