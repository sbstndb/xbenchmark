#include <benchmark/benchmark.h>
#include <vector>

#include <functional>             
#include <type_traits>           

#ifdef XBENCHMARK_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xmath.hpp>
#endif

#include <utils/custom_arguments.hpp>
int min = 1 ;
int max = 1000000 ;
int threshold1 = 1024 ;
int threshold2 = 8096 ;


const int MS = 4 ; // Min_size of arrays
const int RM = 128 ; /// RangeMultiplier
const int PS = 21 ; // pow size


// Note : I cant just use Operations like std::plus<> to reduce code size because I can't 
// achieve to use it with XTensor in limited time.
// So I decided to badly duplicate code for now ...

template <typename T, typename Op>
void ALLOC_raw(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	for (auto _ : state) {
		T* vec = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
		for (int i = 0 ; i < vector_size; i++){
			vec[i] = 1.0 ; 
		}
		benchmark::DoNotOptimize(vec); // compiler artifice 
		free(vec);
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}

template <typename T, typename Op>
void ALLOC_aligned(benchmark::State& state) {
	const int vector_size = state.range(0);
	Op operation ; 
	constexpr std::size_t alignment = 64; 

	for (auto _ : state) {
		T* vec = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));		
		for (int i = 0 ; i < vector_size ; i++){
			vec[i] = 1.0 ; 
		}
		benchmark::DoNotOptimize(vec); // Prevent compiler optimizations
		free(vec) ; 
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T, typename Op>
void ALLOC_std_vector(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	for (auto _ : state) {
		std::vector<T> vec(vector_size, 1);
		benchmark::DoNotOptimize(vec); // compiler artifice 
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void ALLOC_xarray(benchmark::State& state) {
	const unsigned long vector_size = static_cast<unsigned long>(state.range(0));
	for (auto _ : state) {
		xt::xarray<T> vec = xt::xarray<T>::from_shape({vector_size});
//		for (int i = 0 ; i < vector_size; i++){
//			vec[i] = 1.0 ; 
//		}
		vec.fill(1.0) ; 
		benchmark::DoNotOptimize(vec);
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void ALLOC_xtensor(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	for (auto _ : state) {
		xt::xtensor<T, 1> vec = xt::xtensor<T,1>::from_shape({vector_size});
		vec.fill(1.0) ; 
		benchmark::DoNotOptimize(vec);
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
#ifdef XTENSOR_USE_XSIMD
template <typename T, typename Op>
void ALLOC_xtensor_aligned(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
        for (auto _ : state) {
                xt::xtensor<T, 1> vec = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, 64>>::from_shape({vector_size});
                vec.fill(1.0) ;
                benchmark::DoNotOptimize(vec);
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void ALLOC_xtensor_fixed(benchmark::State& state) {
	for (auto _ : state) {
		xt::xtensor_fixed<int, xt::xshape<S>> vec ;
		vec.fill(1.0) ; 
		benchmark::DoNotOptimize(vec.data());
	}
	state.SetItemsProcessed(state.iterations() * S);
}
#endif


// Power of two rule
BENCHMARK_TEMPLATE(ALLOC_raw, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(ALLOC_aligned, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(ALLOC_std_vector, float,		std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(ALLOC_xarray, float, 	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(ALLOC_xtensor, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(ALLOC_xtensor_aligned, float,        std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#endif



// --> Not really dynamic here ...
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 2);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 4);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 8);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 16);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 32);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 64);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 128);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 256);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 512);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 1024);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 2048);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 4096);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 8192);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 16384);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 32768);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 65536);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 131072);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 262144);
BENCHMARK_TEMPLATE(ALLOC_xtensor_fixed, 524288);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 1048576);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 2097152);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 4194304);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8388608);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 16777216);
#endif


BENCHMARK_MAIN();




