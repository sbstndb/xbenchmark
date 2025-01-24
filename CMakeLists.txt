cmake_minimum_required(VERSION 3.28)

project(XBenchmarks)

#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(benchmark REQUIRED)

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
	list(APPEND GLOBAL_DEPENDENCIES xtensor)#xtensor::use_xsimd xtensor::optimize)#, voir ce qu'on doit ajouter 

endif()

if(XBENCHMARK_USE_IMMINTRIN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	add_compile_definitions(XBENCHMARK_USE_IMMINTRIN)
endif()

if(XBENCHMARK_USE_EIGEN)
	find_package(eigen3)
	add_compile_definitions(XBENCHMARK_USE_EIGEN)
	# ajouter : 
	# list(APPEND GLOBAL_DEPENDENCIES eigen3)
endif()



include_directories(include)
add_subdirectory(src)


