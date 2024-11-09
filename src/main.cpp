#include <benchmark/benchmark.h>
#include <vector>

void BM_RawSum(benchmark::State& state) {
    const int vector_size = state.range(0);  // Vector size defined by benchmark range

    for (auto _ : state) {
        int* vec1 = static_cast<int*>(std::malloc(vector_size * sizeof(int)));
        int* vec2 = static_cast<int*>(std::malloc(vector_size * sizeof(int)));
        int* result = static_cast<int*>(std::malloc(vector_size * sizeof(int)));

        // Initialize arrays
        for (int i = 0; i < vector_size; ++i) {
            vec1[i] = 1;
            vec2[i] = 2;
        }

	// compute loop
        for (int i = 0; i < vector_size; ++i) {
            result[i] = vec1[i] + vec2[i];
        }

        benchmark::DoNotOptimize(result); // compiler artifice 
					  //
	free(vec1) ; 
	free(vec2) ; 
	free(result) ;
    }

    // report throughput
    state.SetItemsProcessed(state.iterations() * vector_size);
}


void BM_AlignedAllocSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    constexpr std::size_t alignment = 64; 

    for (auto _ : state) {
        // Allocate aligned memory using std::aligned_alloc
        int* vec1 = static_cast<int*>(std::aligned_alloc(alignment, vector_size * sizeof(int)));
        int* vec2 = static_cast<int*>(std::aligned_alloc(alignment, vector_size * sizeof(int)));
        int* result = static_cast<int*>(std::aligned_alloc(alignment, vector_size * sizeof(int)));

        // Check for successful allocation
        if (!vec1 || !vec2 || !result) {
            state.SkipWithError("Aligned allocation failed.");
            break;
        }

        // Initialize arrays
        for (int i = 0; i < vector_size; ++i) {
            vec1[i] = 1;
            vec2[i] = 2;
        }

        // Perform the addition
        for (int i = 0; i < vector_size; ++i) {
            result[i] = vec1[i] + vec2[i];
        }

        benchmark::DoNotOptimize(result); // Prevent compiler optimizations

        // Free aligned memory
        std::free(vec1);
        std::free(vec2);
        std::free(result);
    }

    state.SetItemsProcessed(state.iterations() * vector_size);
}

void BM_VectorSum(benchmark::State& state) {
    const int vector_size = state.range(0);  // Vector size defined by benchmark range

    for (auto _ : state) {
        std::vector<int> vec1(vector_size, 1);
        std::vector<int> vec2(vector_size, 2);
        std::vector<int> result(vector_size);

        // compute loop
        for (int i = 0; i < vector_size; ++i) {
            result[i] = vec1[i] + vec2[i];
        }

        benchmark::DoNotOptimize(result); // compiler artifice 
    }

    // report throughput
    state.SetItemsProcessed(state.iterations() * vector_size);
}


// Power of two rule
BENCHMARK(BM_RawSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK(BM_AlignedAllocSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK(BM_VectorSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK_MAIN();




