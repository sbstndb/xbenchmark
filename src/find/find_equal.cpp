#include <benchmark/benchmark.h>
#include <iostream>

#include <find/utils.hpp>
#include <find/linear_equal.hpp>
#include <find/binary_equal.hpp>

const int MS = 2 ; // Min_size of arrays
const int RM = 4 ; /// RangeMultiplier
const int PS = 8 ; // pow size

#include <utils/custom_arguments.hpp>
int min = 1 ;
int max = 100000 ;
int threshold1 = 1024 ;
int threshold2 = 8096 ;



void FIND_equal_naive(benchmark::State& state){
        const int size = state.range(0) ;
        int* vector = (int*) malloc(sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
		index = find_equal_naive(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);	
}


void FIND_equal_no_break(benchmark::State& state){
        const int size = state.range(0) ;
        int* vector = (int*) aligned_alloc(64, sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
                index = find_equal_no_break(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);
}



void FIND_equal_compare(benchmark::State& state){
        const int size = state.range(0) ;
        int* vector = (int*) aligned_alloc(64, sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
                index = find_equal_compare(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);
}


void FIND_equal_std_find(benchmark::State& state){
        const int size = state.range(0) ;
        int* vector = (int*) aligned_alloc(64, sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
                index = find_equal_std_find(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);
}

void FIND_equal_std_lower_bound(benchmark::State& state){
        const int size = state.range(0) ;
        int* vector = (int*) aligned_alloc(64, sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
                index = find_equal_std_lower_bound(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);
}



void FIND_equal_intrinsic(benchmark::State& state){
        int size = state.range(0) ;
        // !! AVX
        if (size < 8) {
                size = 8 ;
        }
        int* vector = (int*) malloc(sizeof(int) * size) ;
        int value = 1 ;
        init_vector(vector, size, value, size-1);
        int index ;
        for (auto _ : state){
                index = find_equal_intrinsic(vector, size, value);
                benchmark::DoNotOptimize(index);
        }
        state.SetItemsProcessed(state.iterations() * size);
        free(vector);
}




BENCHMARK(FIND_equal_naive)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(FIND_equal_no_break)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(FIND_equal_compare)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(FIND_equal_std_find)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(FIND_equal_std_lower_bound)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(FIND_equal_intrinsic)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_MAIN() ; 

