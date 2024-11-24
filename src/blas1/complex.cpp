#include <benchmark/benchmark.h>
#include <vector>

#ifdef XBENCHMARK_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xeval.hpp>
#endif

const int MS = 1024 ; // Min_size of arrays
const int RM = 128 ; /// RangeMultiplier
const int PS = 21 ; // pow size


template <typename T = double>
struct complex_op {
	T operator()(const T& a, const T& b, const T& c, const T& d, const T& e) const {
		return a * b +  c * d * e;
	}
};

// Note : I cant just use Operations like std::plus<> to reduce code size because I can't 
// achieve to use it with XTensor in limited time.
// So I decided to badly duplicate code for now ...

template <typename T, typename Op>
void BLAS1_complex_raw(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation ; 
	T a = static_cast<T>(2.0) ; 
	T* vec1 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* vec2 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* vec3 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* vec4 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* result = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
		vec2[i] = 2;
		vec3[i] = 3 ; 
		vec4[i] = 4 ; 
		result[i] = 0;
	}
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(a, vec1[i], vec2[i], vec3[i], vec4[i]) ; 
		}
		benchmark::DoNotOptimize(result); 
	}
	free(vec1) ;
	free(vec2) ;
	free(vec3) ; 
	free(vec4) ; 
	free(result) ;
	state.SetItemsProcessed(state.iterations() * vector_size);
}

template <typename T, typename Op>
void BLAS1_complex_aligned(benchmark::State& state) {
	const int vector_size = state.range(0);
	Op operation ; 
	T a = static_cast<T>(2.0) ;
	constexpr std::size_t alignment = 64; 
	T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* vec2 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* vec3 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* vec4 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* result = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
		vec2[i] = 2;
		vec3[i] = 3; 
		vec4[i] = 4; 
		result[i] = 0 ;
	}
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(a, vec1[i] , vec2[i] , vec3[i] , vec4[i]) ;

		}
		benchmark::DoNotOptimize(result); // Prevent compiler optimizations
	}
	std::free(vec1);
	std::free(vec2);
	std::free(vec3) ; 
	std::free(vec4) ;
	std::free(result);
	state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T, typename Op>
void BLAS1_complex_std_vector(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation;
	T a = static_cast<T>(2.0) ;
	std::vector<T> vec1(vector_size, 1);
	std::vector<T> vec2(vector_size, 2);
	std::vector<T> vec3(vector_size, 3);
	std::vector<T> vec4(vector_size, 4);
	std::vector<T> result(vector_size, 0);
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(a, vec1[i] , vec2[i], vec3[i] , vec4[i]) ; 
		}
		benchmark::DoNotOptimize(result); // compiler artifice 
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_complex_xarray(benchmark::State& state) {
	const int vector_size = state.range(0);
	T a = static_cast<T>(2.0) ;
	xt::xarray<T> vec1 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> vec2 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> vec3 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> vec4 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> result = xt::xarray<T>::from_shape({vector_size});
	vec1.fill(1) ; 
	vec2.fill(2) ; 
	vec3.fill(3) ; 
	vec4.fill(4) ; 
	result.fill(0) ; 
	for (auto _ : state) {
		// lot of constexpr because of xtensor itself
		if constexpr(std::is_same_v<Op, complex_op<T>>){
			xt::noalias(result) = a * vec1 + vec2 * vec3 * vec4;
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_complex_xtensor(benchmark::State& state) {
	const int vector_size = state.range(0);
	T a = static_cast<T>(2.0) ;    
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec3   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec4   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1) ; 
	vec2.fill(2) ; 
	vec3.fill(3) ; 
	vec4.fill(4) ; 
	result.fill(0) ; 
	for (auto _ : state) {
		if constexpr(std::is_same_v<Op, complex_op<T>>){
			xt::noalias(result) = a * vec1 + vec2 * vec3 * vec4 ;
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_complex_xtensor_eval(benchmark::State& state) {
	const int vector_size = state.range(0);
	T a = static_cast<T>(2.0) ;    
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec3   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec4   = xt::xtensor<T,1>::from_shape({vector_size});    
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1);
	vec2.fill(2);
	vec3.fill(3) ; 
	vec4.fill(4) ; 
	result.fill(0);
	for (auto _ : state) {
		if constexpr(std::is_same_v<Op, complex_op<T>>){
			xt::noalias(result) = xt::eval(a * vec1 + vec2 * vec3 * vec4);
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_complex_xtensor_auto_eval(benchmark::State& state) {
	const int vector_size = state.range(0);
	T a = static_cast<T>(2.0) ;
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec3   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec4   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1);
	vec2.fill(2);
	vec3.fill(3) ;
	vec4.fill(4) ;
	result.fill(0);
	for (auto _ : state) {
		if constexpr(std::is_same_v<Op, complex_op<T>>){
			auto eval = xt::eval(a * vec1 + vec2 * vec3 * vec4);
			xt::noalias(result) = eval ; 
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_complex_xtensor_sum_only_auto_eval(benchmark::State& state) {
	const int vector_size = state.range(0);
	T a = static_cast<T>(2.0) ;
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec3   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec4   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1);
	vec2.fill(2);
	vec3.fill(3) ;
	vec4.fill(4) ;
	result.fill(0);
	for (auto _ : state) {
		if constexpr(std::is_same_v<Op, complex_op<T>>){
			auto eval = xt::eval(a * vec1 + vec2 * vec3 * vec4);
		}
		//        benchmark::DoNotOptimize(eval);
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



// Power of two rule
BENCHMARK_TEMPLATE(BLAS1_complex_raw, float,	complex_op<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_aligned, float,	complex_op<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_std_vector, float,		complex_op<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(BLAS1_complex_xarray, float, 	complex_op<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_xtensor, float,	complex_op<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_xtensor_eval, float,        complex_op<      float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_xtensor_auto_eval, float,        complex_op<      float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BLAS1_complex_xtensor_only_auto_eval, float,        complex_op<      float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
#endif

BENCHMARK_MAIN();




