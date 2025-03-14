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

#include <utils/custom_arguments.hpp>
int min = 1 ;
int max = 1000000 ;
int threshold1 = 1024 ;
int threshold2 = 8096 ;



const int MS = 16 ; // Min_size of arrays
const int RM = 128 ; /// RangeMultiplier
const int PS = 21 ; // pow size

template <typename T>
void VIEW_all_raw(benchmark::State& state) {
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
void VIEW_all_aligned(benchmark::State& state) {
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

template <typename T>
void VIEW_all_aligned_masked(benchmark::State& state) {
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
		mask[i] = 1 ; 
	}
	for (auto _ : state) {
		// compute loop
		for (int i = 0; i < vector_size; ++i) {
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
void VIEW_all_xarray(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
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
void VIEW_all_xtensor(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
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
void VIEW_all_xtensor_range(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
        xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
        xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
        vec1.fill(1) ;
        vec2.fill(2) ;
        result.fill(0) ;
        for (auto _ : state) {
                auto view1 = xt::view(vec1, xt::range(0, vector_size, 1)) ;
                auto view2 = xt::view(vec2, xt::range(0, vector_size, 1)) ;
                xt::noalias(result) = view1 + view2;

                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_all_xtensor_range_only_one(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
        xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
        xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
        vec1.fill(1) ;
        vec2.fill(2) ;
        result.fill(0) ;
        for (auto _ : state) {
                auto view1 = xt::view(vec1, xt::range(vector_size-1, vector_size, 1)) ;
                auto view2 = xt::view(vec2, xt::range(vector_size-1, vector_size, 1)) ;
                xt::noalias(result) = view1 + view2;

                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * 1);
}
#endif




#ifdef XBENCHMARK_USE_XTENSOR
template <typename T>
void VIEW_all_xtensor_strided(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
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
void VIEW_all_xtensor_strided_range(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
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
// Observation : very slow !  We should try with raw pointers to compare potential performances. 
template <typename T>
void VIEW_all_xtensor_masked(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;

	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(1);
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
// Observation : very slow !  We should try with raw pointers to compare potential performances. 
template <typename T>
void VIEW_all_xtensor_masked_2(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;

	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(1);
	// define mask 
	for (auto _ : state) {
		xt::masked_view(result, mask) = vec1 + vec2;

		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
// Observation : very slow !  We should try with raw pointers to compare potential performances. 
template <typename T>
void VIEW_all_xtensor_raw_masked(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2 = xt::xtensor<T, 1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({vector_size});
	vec1.fill(1) ;
	vec2.fill(2) ;
	result.fill(0) ;

	xt::xtensor<bool,1> mask = xt::xtensor<bool, 1>::from_shape({vector_size});
	mask.fill(1);
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
BENCHMARK_TEMPLATE(VIEW_all_raw, float     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_aligned, float     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_aligned_masked, float     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#ifdef XBENCHMARK_USE_IMMINTRIN
#endif
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(VIEW_all_xarray, float	)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_range, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_strided, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_strided_range, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_range_only_one,float    )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_masked, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_masked_2, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(VIEW_all_xtensor_raw_masked, float      )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#endif


BENCHMARK_MAIN();




