cmake_minimum_required(VERSION 3.28)

project(XBenchmarks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(benchmark REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -mavx2 -O3 -funroll-loops -ftree-vectorize -g")


set("GLOBAL_DEPENDENCIES" benchmark::benchmark)

if(XBENCHMARK_USE_XTENSOR)
	find_package(xtensor REQUIRED)
	add_compile_definitions(XBENCHMARK_USE_XTENSOR)
	list(APPEND GLOBAL_DEPENDENCIES xtensor xtensor::use_xsimd xtensor::optimize)#, voir ce qu'on doit ajouter 

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



if (XBENCHMARK_USE_XTENSOR)
	#target_link_libraries(ViewBenchmark ${GLOBAL_DEPENDANCIES} xtensor xtensor::use_xsimd xtensor::optimize)
	#target_link_libraries(ViewStrideBenchmark ${GLOBAL_DEPENDANCIES} xtensor xtensor::use_xsimd xtensor::optimize)
	#	target_link_libraries(FindBenchmark ${GLOBAL_DEPENDANCIES} xtensor xtensor::use_xsimd xtensor::optimize)



endif()

# Optimisation (optionnel) : Ajustez si nécessaire
#target_compile_options(ViewBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
#target_compile_options(ViewStrideBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)
#target_compile_options(FindBenchmark PRIVATE -O3 -mavx2 -mfma -march=native -mtune=native)






