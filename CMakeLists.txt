cmake_minimum_required(VERSION 3.28)

# Nom du projet
project(XtensorBenchmarks)

# Exige une version de C++ standard (au moins C++11 pour Google Benchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Chercher Google Benchmark installé sur le système
#find_package(xtensor REQUIRED)
find_package(benchmark REQUIRED)

# Ajouter l'exécutable
add_executable(VectorSumBenchmark src/main.cpp)

# Lier Google Benchmark à l'exécutable
target_link_libraries(VectorSumBenchmark benchmark::benchmark)

# Optimisation (optionnel) : Ajustez si nécessaire
target_compile_options(VectorSumBenchmark PRIVATE -O3)

