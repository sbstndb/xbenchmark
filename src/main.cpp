#include <benchmark/benchmark.h>
#include <vector>

#ifdef XBENCHMARK_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#endif


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

#ifdef XBENCHMARK_USE_XTENSOR
void BM_XArraySum(benchmark::State& state) {
    const int vector_size = state.range(0);
    for (auto _ : state) {
        xt::xarray<int> vec1 = xt::xarray<int>::from_shape({vector_size});
        xt::xarray<int> vec2 = xt::xarray<int>::from_shape({vector_size});
        xt::xarray<int> result = xt::xarray<int>::from_shape({vector_size});

        vec1.fill(1);
        vec2.fill(2);

        xt::noalias(result) = vec1 + vec2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
void BM_XTensorSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    for (auto _ : state) {
        xt::xtensor<int, 1> vec1   = xt::xtensor<int,1>::from_shape({vector_size});
        xt::xtensor<int, 1> vec2   = xt::xtensor<int,1>::from_shape({vector_size});
        xt::xtensor<int, 1> result = xt::xtensor<int,1>::from_shape({vector_size});

        vec1.fill(1);
        vec2.fill(2);

        xt::noalias(result) = vec1 + vec2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t T>
void BM_XTensorFixedSum(benchmark::State& state) {
    for (auto _ : state) {
        xt::xtensor_fixed<int, xt::xshape<T>> vec1 ; 
        xt::xtensor_fixed<int, xt::xshape<T>> vec2 ;
        xt::xtensor_fixed<int, xt::xshape<T>> result;

        vec1.fill(1);
        vec2.fill(2);

        xt::noalias(result) = vec1 + vec2;
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * T);
}
#endif


// Power of two rule
BENCHMARK(BM_RawSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK(BM_AlignedAllocSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK(BM_VectorSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);

#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK(BM_XArraySum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK(BM_XTensorSum)->RangeMultiplier(2)->Range(1 << 0, 1 << 24);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 2);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 4);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 16);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 32);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 64);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 128);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 256);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 512);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 1024);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 2048);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 4096);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8192);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 16384);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 32768);
BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 65536);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 131072);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 262144);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 524288);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 1048576);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 2097152);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 4194304);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8388608);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 16777216);
#endif



BENCHMARK_MAIN();




