cmake_minimum_required(VERSION 3.28)

project(XBenchmarks)

#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED True)


#find_package(benchmark REQUIRED)

include(FetchContent)

# Télécharger Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3  # Spécifiez la version souhaitée
)

# Disable Google Test integration
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Test")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing")

# Intégrer Google Benchmark dans votre projet
FetchContent_MakeAvailable(benchmark)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -mavx2 -O3 -funroll-loops -ftree-vectorize -g")


set("GLOBAL_DEPENDENCIES" benchmark::benchmark)

if(XBENCHMARK_USE_XTENSOR)
	find_package(xtensor REQUIRED)
	find_package(xsimd)
	if (xsimd_FOUND)
		include_directories(${xsimd_INCLUDE_DIRS})
		add_definitions("-DXTENSOR_USE_XSIMD=1")
	endif()
	include_directories(${XTENSOR_INCLUDE_DIRS})
	include_directories(${GBENCHMARK_INCLUDE_DIRS})
	add_compile_definitions(XBENCHMARK_USE_XTENSOR)
	list(APPEND GLOBAL_DEPENDENCIES benchmark::benchmark xtensor)#xtensor::use_xsimd xtensor::optimize)#, voir ce qu'on doit ajouter 

endif()

if(XBENCHMARK_USE_IMMINTRIN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	add_compile_definitions(XBENCHMARK_USE_IMMINTRIN)
endif()

if(XBENCHMARK_USE_EIGEN)
	find_package(Eigen3 REQUIRED)
	add_compile_definitions(XBENCHMARK_USE_EIGEN)
	include_directories(${EIGEN3_INCLUDE_DIRS})
	add_compile_definitions(XBENCHMARK_USE_EIGEN)
	list(APPEND GLOBAL_DEPENDENCIES Eigen3::Eigen)
	# ajouter : 
	# list(APPEND GLOBAL_DEPENDENCIES eigen3)
endif()

if(XBENCHMARK_USE_KOKKOS)
	find_package(Kokkos REQUIRED)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)

	add_compile_definitions(XBENCHMARK_USE_KOKKOS)
	list(APPEND GLOBAL_DEPENDENCIES Kokkos::kokkos)
endif()




include_directories(include)
add_subdirectory(src)


