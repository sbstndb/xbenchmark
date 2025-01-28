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
void BLAS1_op_raw(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation ; 

	T* vec1 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* result = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
	}
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , static_cast<T>(1.0)) ;
		}
		benchmark::DoNotOptimize(result); // compiler artifice 
	}
	free(vec1) ;
	free(result) ;
	state.SetItemsProcessed(state.iterations() * vector_size);
}

template <typename T, typename Op>
void BLAS1_op_aligned(benchmark::State& state) {
	const int vector_size = state.range(0);
	Op operation ; 
	constexpr std::size_t alignment = 64; 

	T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* result = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
	}
	for (auto _ : state) {
		// compute loop
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , static_cast<T>(1.0)) ;

		}
		benchmark::DoNotOptimize(result); // Prevent compiler optimizations
	}
	std::free(vec1);
	std::free(result);
	state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T, typename Op>
void BLAS1_op_std_vector(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation;
	std::vector<T> vec1(vector_size, 1);
	std::vector<T> result(vector_size);
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , static_cast<T>(1.0)) ; 
		}
		benchmark::DoNotOptimize(result); // compiler artifice 
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xarray(benchmark::State& state) {
	const unsigned long vector_size = static_cast<unsigned long>(state.range(0));
	xt::xarray<T> vec1 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<T> result = xt::xarray<T>::from_shape({vector_size});
	vec1.fill(1.0) ; 
	for (auto _ : state) {
		// lot of constexpr because of xtensor itself
		if constexpr(std::is_same_v<Op, std::plus<T>>){
			xt::noalias(result) = xt::eval(vec1 + static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = xt::eval(vec1 - static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = xt::eval(vec1 * static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = xt::eval(vec1 / static_cast<T>(1.0));
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1) ; 
	for (auto _ : state) {
		// lots of constexpr becaus of xtensor itself
		if constexpr(std::is_same_v<Op, std::plus<T>>){
			xt::noalias(result) = vec1 + static_cast<T>(1.0);
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = vec1 - static_cast<T>(1.0);
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = vec1 * static_cast<T>(1.0);
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = vec1 / static_cast<T>(1.0);
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_aligned_64(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
	const int align_size = 64 ; 
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        vec1.fill(1) ;
        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::plus<T>>){
                        xt::noalias(result) = vec1 + static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / static_cast<T>(1.0);
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_explicit(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
        const int align_size = 64 ;
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
        vec1.fill(1) ;
        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::plus<T>>){
                        for (int i = 0 ; i < vector_size ; i++){
                                result(i) = vec1(i) + static_cast<T>(1.0) ;
                        }
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / static_cast<T>(1.0);
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_explicit_aligned_16(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
        const int align_size = 16 ;
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        vec1.fill(1) ;
        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::plus<T>>){
                        for (int i = 0 ; i < vector_size ; i++){
                                result(i) = vec1(i) + static_cast<T>(1.0) ;
                        }
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / static_cast<T>(1.0);
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_explicit_aligned_64(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
	const int align_size = 64 ; 
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        vec1.fill(1) ;
        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::plus<T>>){
			for (int i = 0 ; i < vector_size ; i++){
				result(i) = vec1(i) + static_cast<T>(1.0) ; 
			}
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * static_cast<T>(1.0);
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / static_cast<T>(1.0);
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



/**
#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_explicit_xsimd(benchmark::State& state) {
        const unsigned long vector_size = state.range(0);
	const int align_size = 64 ; 
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1, xt::layout_type::row_major, xsimd::aligned_allocator<T, align_size>>::from_shape({vector_size});
        vec1.fill(1) ;
	using simd_type = xt_simd::simd_type<double>;
	size_t simd_size = simd_type::size;

        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::plus<T>>){
		        for (size_t i = 0; i < vec1.size(); i += simd_size) {
		                xt_simd::store_as(
		                        result.data() + i,
		                        xt_simd::load_as<double>(vec1.data() + i) + xt_simd::broadcast_as<double>(1.0),
		                        xt_simd::aligned_mode() // aligned or unaligned mode depending on your data
		                );
		        } 
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - 1.0;
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * 1.0;
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / 1.0;
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif
**/


#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_eval(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});
	vec1.fill(1);
	result.fill(0);
	for (auto _ : state) {
		// lots of constexpr becaus of xtensor itself
		if constexpr(std::is_same_v<Op, std::plus<T>>){
			xt::noalias(result) = xt::eval(vec1 + static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = xt::eval(vec1 - static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = xt::eval(vec1 * static_cast<T>(1.0));
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = xt::eval(vec1 / static_cast<T>(1.0));
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void BLAS1_op_xtensor_fixed(benchmark::State& state) {
	xt::xtensor_fixed<float, xt::xshape<S>> vec1 ;
	xt::xtensor_fixed<float, xt::xshape<S>> result;
	vec1.fill(1);
	for (auto _ : state) {
		result = vec1 + static_cast<float>(1.0);
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * S);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void BLAS1_op_xtensor_fixed_noalias(benchmark::State& state) {
        xt::xtensor_fixed<float, xt::xshape<S>> vec1 ;
        xt::xtensor_fixed<float, xt::xshape<S>> result;
        vec1.fill(1);
        for (auto _ : state) {
                xt::noalias(result) = vec1 + static_cast<float>(1.0);
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * S);
}
#endif

#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void BLAS1_op_xtensor_fixed_explicit(benchmark::State& state) {
        xt::xtensor_fixed<float, xt::xshape<S>> vec1 ;
        xt::xtensor_fixed<float, xt::xshape<S>> result;
        vec1.fill(1);
        for (auto _ : state) {
		for (int i = 0 ; i < S ; i++){
	                result(i) = vec1(i) + static_cast<float>(1.0);
		}
	        benchmark::DoNotOptimize(result.data());

        }
        state.SetItemsProcessed(state.iterations() * S);
}
#endif



// Power of two rule
BENCHMARK_TEMPLATE(BLAS1_op_raw, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_aligned, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_std_vector, float,		std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;

#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(BLAS1_op_xarray, float, 	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor, float,	std::plus<	float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_aligned_64,float,  std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_explicit, float,     std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_explicit_aligned_16, float,     std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_explicit_aligned_64, float,     std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_eval, float,        std::plus<      float>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#endif



// --> Not really dynamic here ...
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 4);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 128);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 16384);
//
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 4);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 128);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 16384);

BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_explicit, 4);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_explicit, 128);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_explicit, 16384);
#endif


BENCHMARK_MAIN();




