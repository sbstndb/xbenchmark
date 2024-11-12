#include <benchmark/benchmark.h>
#include <vector>

#ifdef XBENCHMARK_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#endif

const int MS = 1024 ; // Min_size of arrays
const int RM = 128 ; /// RangeMultiplier
const int PS = 21 ; // pow size

template <typename T>
void BM_RawSum(benchmark::State& state) {
    const int vector_size = state.range(0);  // Vector size defined by benchmark range
    T* vec1 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
    T* vec2 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
    T* result = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
    // Initialize arrays
    for (int i = 0; i < vector_size; ++i) {
        vec1[i] = 1;
        vec2[i] = 2;
	result[i] = 0 ; 
    }
    for (auto _ : state) {
        // compute loop
        for (int i = 0; i < vector_size; ++i) {
                result[i] = vec1[i] + vec2[i];
        }
	benchmark::DoNotOptimize(result); // compiler artifice 
    }
    free(vec1) ;
    free(vec2) ;
    free(result) ;
    // report throughput
    state.SetItemsProcessed(state.iterations() * vector_size);
}

template <typename T>
void BM_AlignedAllocSum(benchmark::State& state) {
    const int vector_size = state.range(0);

    constexpr std::size_t alignment = 64;

    // Allocate aligned memory using std::aligned_alloc
    T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
    T* vec2 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
    T* result = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
    // Initialize arrays
    for (int i = 0; i < vector_size; ++i) {
        vec1[i] = 1;
        vec2[i] = 2;
        result[i] = 0 ;
    }
    for (auto _ : state) {
        // compute loop
        for (int i = 0; i < vector_size; ++i) {
                result[i] =  vec1[i] + vec2[i] ;
        }
        benchmark::DoNotOptimize(result); // Prevent compiler optimizations
    }
    // Free aligned memory
    std::free(vec1);
    std::free(vec2);
    std::free(result);
    state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void BM_XArrayViewSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    xt::xarray<T> vec1 = xt::xarray<T>::from_shape({vector_size});
    xt::xarray<T> vec2 = xt::xarray<T>::from_shape({vector_size});
    xt::xarray<T> result = xt::xarray<T>::from_shape({vector_size});
    vec1.fill(1) ; 
    vec2.fill(2) ; 
    result.fill(0) ; 
    for (auto _ : state) {
        auto view1 = xt::view(vec1, xt::all()) ;     
	auto view2 = xt::view(vec2, xt::all()) ;
        xt::noalias(result) = view1 + view2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void BM_XTensorViewSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
    vec1.fill(1) ;
    vec2.fill(2) ;
    result.fill(0) ; 
    for (auto _ : state) {
        auto view1 = xt::view(vec1, xt::all()) ;
        auto view2 = xt::view(vec2, xt::all()) ;
        xt::noalias(result) = view1 + view2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void BM_XTensorStridedViewAllSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
    vec1.fill(1) ;
    vec2.fill(2) ;
    result.fill(0) ; 
    for (auto _ : state) {
        auto view1 = xt::strided_view(vec1, {xt::all()}) ;
        auto view2 = xt::strided_view(vec2, {xt::all()}) ;
        xt::noalias(result) = view1 + view2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void BM_XTensorStridedViewAllRangeSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
    vec1.fill(1) ;
    vec2.fill(2) ;
    result.fill(0) ;
    for (auto _ : state) {
        auto view1 = xt::strided_view(vec1, {xt::range(0, vector_size, 1)}) ;
        auto view2 = xt::strided_view(vec2, {xt::range(0, vector_size, 1)}) ;
        xt::noalias(result) = view1 + view2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


// TODO --> xt::masked_view


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void BM_XTensorMaskedViewAllSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
    vec1.fill(1) ;
    vec2.fill(2) ;
    result.fill(0) ;
    // define mask 
    for (auto _ : state) {
//        auto view1 = xt::masked_view(vec1, mask) ;
//        auto view2 = xt::masked_view(vec2, mask) ;
//        xt::noalias(result) = vec1 + vec2;

        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



// Power of two rule
//
BENCHMARK_TEMPLATE(BM_RawSum, float     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, float     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArrayViewSum, float	)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorViewSum, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorStridedViewAllSum, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorStridedViewAllRangeSum, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorMaskedViewAllSum, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);



BENCHMARK_MAIN();




