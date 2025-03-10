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

#ifdef XBENCHMARK_USE_EIGEN
#include <Eigen/Dense>
#endif


#include <utils/custom_arguments.hpp>

int min = 1 ;
int max = 1000000 ;
int threshold1 = 1024 ;
int threshold2 = 8096 ;


#ifdef XBENCHMARK_USE_EIGEN
template <typename T, typename Op>
void BLAS1_op_eigen_matrix(benchmark::State& state){
        const int vector_size = state.range(0);  // Vector size defined by benchmark range
        Op operation ;

        Eigen::Matrix<T, Eigen::Dynamic, 1>  vec1 = Eigen::Matrix<T, Eigen::Dynamic, 1>::Constant(vector_size, 1.0) ;
        Eigen::Matrix<T, Eigen::Dynamic, 1>  vec2 = Eigen::Matrix<T, Eigen::Dynamic, 1>::Constant(vector_size, 2.0) ;
        Eigen::Matrix<T, Eigen::Dynamic, 1>  result(vector_size) ;


        for (auto _ : state){
                result = vec1.binaryExpr(vec2, operation);
                benchmark::DoNotOptimize(result); // compiler artifice 
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



template <typename T, typename Op>
void BLAS1_op_raw(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation ; 

	T* vec1 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	T* vec2 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
	bool* result = static_cast<bool*>(std::malloc(vector_size * sizeof(bool)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
		vec2[i] = 2;
	}
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , vec2[i]) ;
		}
		benchmark::DoNotOptimize(result); // compiler artifice 
	}
	free(vec1) ;
	free(vec2) ;
	free(result) ;
	state.SetItemsProcessed(state.iterations() * vector_size);
}

template <typename T, typename Op>
void BLAS1_op_aligned(benchmark::State& state) {
	const int vector_size = state.range(0);
	Op operation ; 
	constexpr std::size_t alignment = 64; 

	T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	T* vec2 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
	bool* result = static_cast<bool*>(std::aligned_alloc(alignment, vector_size * sizeof(bool)));
	for (int i = 0; i < vector_size; ++i) {
		vec1[i] = 1;
		vec2[i] = 2;
	}
	for (auto _ : state) {
		// compute loop
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , vec2[i]) ;

		}
		benchmark::DoNotOptimize(result); // Prevent compiler optimizations
	}
	std::free(vec1);
	std::free(vec2);
	std::free(result);
	state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T, typename Op>
void BLAS1_op_std_vector(benchmark::State& state) {
	const int vector_size = state.range(0);  // Vector size defined by benchmark range
	Op operation;
	std::vector<T> vec1(vector_size, 1);
	std::vector<T> vec2(vector_size, 2);
	std::vector<bool> result(vector_size);
	for (auto _ : state) {
		for (int i = 0; i < vector_size; ++i) {
			result[i] = operation(vec1[i] , vec2[i]) ; 
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
	xt::xarray<T> vec2 = xt::xarray<T>::from_shape({vector_size});
	xt::xarray<bool> result = xt::xarray<bool>::from_shape({vector_size});
	vec1.fill(1.0) ; 
	vec2.fill(2.0);
//	vec1 = 1.0 ; 
//	vec2 = 2.0 ; 
	for (auto _ : state) {
		// lot of constexpr because of xtensor itself
		if constexpr(std::is_same_v<Op, std::less<T>>){
			xt::noalias(result) = xt::eval(vec1 < vec2);
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = xt::eval(vec1 - vec2);
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = xt::eval(vec1 * vec2);
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = xt::eval(vec1 / vec2);
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
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<bool, 1> result = xt::xtensor<bool,1>::from_shape({vector_size});
	vec1.fill(1) ; 
	vec2.fill(2) ; 
	for (auto _ : state) {
		// lots of constexpr becaus of xtensor itself
		if constexpr(std::is_same_v<Op, std::less<T>>){
			xt::noalias(result) = vec1 < vec2;
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = vec1 - vec2;
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = vec1 * vec2;
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = vec1 / vec2;
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
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<bool, 1> result = xt::xtensor<bool,1>::from_shape({vector_size});
        vec1.fill(1) ;
        vec2.fill(2) ;
        for (auto _ : state) {
                // lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::less<T>>){
			for (size_t i =  0 ; i < vector_size ; i++){
				result(i) = vec1(i) < vec2(i) ; 
			}
                } else if constexpr(std::is_same_v<Op, std::minus<T>>){
                        xt::noalias(result) = vec1 - vec2;
                } else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
                        xt::noalias(result) = vec1 * vec2;
                } else if constexpr(std::is_same_v<Op, std::divides<T>>){
                        xt::noalias(result) = vec1 / vec2;
                }
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BLAS1_op_xtensor_eval(benchmark::State& state) {
	const unsigned long vector_size = state.range(0);
	xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
	xt::xtensor<bool, 1> result = xt::xtensor<bool,1>::from_shape({vector_size});
	vec1.fill(1);
	vec2.fill(2);
	result.fill(0);
	for (auto _ : state) {
		// lots of constexpr becaus of xtensor itself
                if constexpr(std::is_same_v<Op, std::less<T>>){
                        xt::noalias(result) = xt::eval(vec1 < vec2);
		}
		if constexpr(std::is_same_v<Op, std::plus<T>>){
			xt::noalias(result) = xt::eval(vec1 + vec2);
		} else if constexpr(std::is_same_v<Op, std::minus<T>>){
			xt::noalias(result) = xt::eval(vec1 - vec2);
		} else if constexpr(std::is_same_v<Op, std::multiplies<T>>){
			xt::noalias(result) = xt::eval(vec1 * vec2);
		} else if constexpr(std::is_same_v<Op, std::divides<T>>){
			xt::noalias(result) = xt::eval(vec1 / vec2);
		}
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * vector_size);
}
#endif



#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void BLAS1_op_xtensor_fixed(benchmark::State& state) {
	xt::xtensor_fixed<int, xt::xshape<S>> vec1 ;
	xt::xtensor_fixed<int, xt::xshape<S>> vec2 ;
	xt::xtensor_fixed<bool, xt::xshape<S>> result;
	vec1.fill(1);
	vec2.fill(2);
	for (auto _ : state) {
		result = vec1 < vec2;
		benchmark::DoNotOptimize(result.data());
	}
	state.SetItemsProcessed(state.iterations() * S);
}
#endif


#ifdef XBENCHMARK_USE_XTENSOR
template <std::size_t S>
void BLAS1_op_xtensor_fixed_noalias(benchmark::State& state) {
        xt::xtensor_fixed<int, xt::xshape<S>> vec1 ;
        xt::xtensor_fixed<int, xt::xshape<S>> vec2 ;
        xt::xtensor_fixed<bool, xt::xshape<S>> result;
        vec1.fill(1);
        vec2.fill(2);
        for (auto _ : state) {
                xt::noalias(result) = vec1 < vec2;
                benchmark::DoNotOptimize(result.data());
        }
        state.SetItemsProcessed(state.iterations() * S);
}
#endif

#ifdef XBENCHMARK_USE_EIGEN
//BENCHMARK_TEMPLATE(BLAS1_op_eigen_matrix, int,   std::less<      int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#endif

// Power of two rule
BENCHMARK_TEMPLATE(BLAS1_op_raw, int,	std::less<	int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_aligned, int,	std::less<	int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_std_vector, int,		std::less<	int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(BLAS1_op_xarray, int, 	std::less<	int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor, int,	std::less<	int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_explicit, int,     std::less<      int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_eval, int,        std::less<      int>)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
#endif



// --> Not really dynamic here ...
#ifdef XBENCHMARK_USE_XTENSOR
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 8);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 2);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 4);
///BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 8);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 16);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 32);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 64);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 128);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 256);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 512);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 1024);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 2048);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 4096);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 8192);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 16384);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 32768);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 65536);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 131072);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 262144);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 524288);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 1048576);
//BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed, 2097152);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 4194304);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 8388608);
//BENCHMARK_TEMPLATE(BM_XTensorFixedSum, 16777216);
//
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 4);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 128);
BENCHMARK_TEMPLATE(BLAS1_op_xtensor_fixed_noalias, 16384);
#endif


BENCHMARK_MAIN();




