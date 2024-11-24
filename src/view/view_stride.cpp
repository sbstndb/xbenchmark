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

#ifdef XBENCHMARK_USE_IMMINTRIN
#include <immintrin.h>
#endif

const int MS = 1024 ; // Min_size of arrays
const int RM = 128 ; /// RangeMultiplier
const int PS = 21 ; // pow size

const int stride = 4 ; 

template <typename T>
void VIEW_stride_aligned_masked(benchmark::State& state) {
	const int vector_size = state.range(0);

	constexpr std::size_t alignment = 64;
	// Allocate aligned memory using std::aligned_alloc
	T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* vec2 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* result = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	bool* mask = static_cast<bool*>(std::aligned_alloc(alignment, vector_size * sizeof(bool))) ; 
	// Initialize arrays
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
		vec2[i] = 2;
		result[i] = 0 ;
		mask[i] = 0 ; 
	}
	for (int i = 0 ; i < vector_size; i+=stride){
		mask[i] = 1 ; 
	}
	for (auto _ : state) {
		// compute loop
		for (int i = 0; i < vector_size; i++) {
			result[i] = (mask[i] == true) ? (vec1[i] + vec2[i]) : result[i] ;
			//		if (mask[i] == true)   
			//			result[i] = vec1[i] + vec2[i] ; 
			//
		}
		benchmark::DoNotOptimize(result); // Prevent compiler optimizations
	}
	// Free aligned memory
	std::free(vec1);
	std::free(vec2);
	std::free(result);
	std::free(mask);
	state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_IMMINTRIN
// TODO : optimize this kernel : we want this to compile into avx mask instructions
// !!! T should be float in this experimental case
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xarray(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xarray<T> vec1 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> vec2 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> result = xt::xarray<T>::from_shape({vector_size});
	vec1.fill(1) ; 
	vec2.fill(2) ; 
	result.fill(0) ; 
	for (auto _ : state) {
		auto view1 = xt::view(vec1, xt::range(0, vector_size, stride)) ;     
		auto view2 = xt::view(vec2, xt::range(0, vector_size, stride)) ;
		xt::noalias(result) = view1 + view2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ; 
	for (auto _ : state) {
		auto view1 = xt::view(vec1, xt::range(0, vector_size, stride)) ;
		auto view2 = xt::view(vec2, xt::range(0, vector_size, stride)) ;
		xt::noalias(result) = view1 + view2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor_strided(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ; 
	for (auto _ : state) {
		auto view1 = xt::strided_view(vec1, {xt::range(0, vector_size, stride)}) ;
		auto view2 = xt::strided_view(vec2, {xt::range(0, vector_size, stride)}) ;
		xt::noalias(result) = view1 + view2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor_strided_range(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;
	for (auto _ : state) {
		auto view1 = xt::strided_view(vec1, {xt::range(0, vector_size, stride)}) ;
		auto view2 = xt::strided_view(vec2, {xt::range(0, vector_size, stride)}) ;
		xt::noalias(result) = view1 + view2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


// TODO --> xt::masked_view


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor_masked(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;
	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(0) ; 
	for (int i = 0 ; i < vector_size ; i+=stride){
		mask[i] = true ; 
	}
	// define mask 
	for (auto _ : state) {
		auto view1 = xt::masked_view(vec1, mask) ;
		auto view2 = xt::masked_view(vec2, mask) ;
		xt::noalias(result) = xt::eval(view1) + xt::eval(view2);

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor_masked2(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;

	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(0);
	for (int i = 0 ; i < vector_size ; i+=stride){
		mask[i] = true ;
	}    
	// define mask 
	for (auto _ : state) {
		xt::masked_view(result, mask) = vec1 + vec2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_stride_xtensor_raw_masked(benchmark::State& state) {
	const int vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;

	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(0);
	for (int i = 0 ; i < vector_size ; i+=stride){
		mask[i] = true ;
	}

	// define mask 
	for (auto _ : state) {
		for (int i = 0 ; i < vector_size; i++){
			result[i] = (mask[i] == true) ? (vec1[i] + vec2[i]) : result[i] ;
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



// Power of two rule
//
BENCHMARK_TEMPLATE(VIEW_stride_aligned_masked, float     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
#ifdef XBENCHMARK_USE_IMMINTRIN
#endif
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(VIEW_stride_xarray, float	)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor_strided, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor_strided, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor_masked, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor_masked2, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(VIEW_stride_xtensor_raw_masked, float      )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);


#endif


BENCHMARK_MAIN();




