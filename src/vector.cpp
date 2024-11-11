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


// Note : I cant just use Operations like std::plus<> to reduce code size because I can't 
// achieve to use it with XTensor in limited time.
// So I decided to badly duplicate code for now ...

template <typename T, typename Op>
void BM_RawSum(benchmark::State& state) {
    const int vector_size = state.range(0);  // Vector size defined by benchmark range
    Op operation ; 
    for (auto _ : state) {
        T* vec1 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
        T* vec2 = static_cast<T*>(std::malloc(vector_size * sizeof(T)));
        T* result = static_cast<T*>(std::malloc(vector_size * sizeof(T)));

        // Initialize arrays
        for (int i = 0; i < vector_size; ++i) {
            vec1[i] = 1;
            vec2[i] = 2;
        }

        // compute loop
        for (int i = 0; i < vector_size; ++i) {
                result[i] = operation(vec1[i] , vec2[i]) ;
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

template <typename T, typename Op>
void BM_AlignedAllocSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    Op operation ; 
    constexpr std::size_t alignment = 64; 

    for (auto _ : state) {
        // Allocate aligned memory using std::aligned_alloc
        T* vec1 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
        T* vec2 = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
        T* result = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));

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

        // compute loop
        for (int i = 0; i < vector_size; ++i) {
                result[i] = operation(vec1[i] , vec2[i]) ;

        }

        benchmark::DoNotOptimize(result); // Prevent compiler optimizations

        // Free aligned memory
        std::free(vec1);
        std::free(vec2);
        std::free(result);
    }

    state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T, typename Op>
void BM_VectorSum(benchmark::State& state) {
    const int vector_size = state.range(0);  // Vector size defined by benchmark range
    Op operation;
    for (auto _ : state) {
        std::vector<T> vec1(vector_size, 1);
        std::vector<T> vec2(vector_size, 2);
        std::vector<T> result(vector_size);

        // compute loop
        for (int i = 0; i < vector_size; ++i) {
		result[i] = operation(vec1[i] , vec2[i]) ; 
	}
        benchmark::DoNotOptimize(result); // compiler artifice 
    }

    // report throughput
    state.SetItemsProcessed(state.iterations() * vector_size);
}

#ifdef XBENCHMARK_USE_XTENSOR
template <typename T, typename Op>
void BM_XArraySum(benchmark::State& state) {
    const int vector_size = state.range(0);
    for (auto _ : state) {
        xt::xarray<T> vec1 = xt::xarray<T>::from_shape({vector_size});
        xt::xarray<T> vec2 = xt::xarray<T>::from_shape({vector_size});
        xt::xarray<T> result = xt::xarray<T>::from_shape({vector_size});

        vec1.fill(1);
        vec2.fill(2);

	// lot of constexpr because of xtensor itself
	if constexpr(std::is_same_v<Op, std::plus<T>>){
	        xt::noalias(result) = vec1 + vec2;
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
void BM_XTensorSum(benchmark::State& state) {
    const int vector_size = state.range(0);
    for (auto _ : state) {
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});

        vec1.fill(1);
        vec2.fill(2);

	// lots of constexpr becaus of xtensor itself
        if constexpr(std::is_same_v<Op, std::plus<T>>){
                xt::noalias(result) = vec1 + vec2;
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
void BM_XTensorSumEval(benchmark::State& state) {
    const int vector_size = state.range(0);
    for (auto _ : state) {
        xt::xtensor<T, 1> vec1   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> vec2   = xt::xtensor<T,1>::from_shape({vector_size});
        xt::xtensor<T, 1> result = xt::xtensor<T,1>::from_shape({vector_size});

        vec1.fill(1);
        vec2.fill(2);

        // lots of constexpr becaus of xtensor itself
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
void BM_XTensorFixedSum(benchmark::State& state) {
    for (auto _ : state) {
        xt::xtensor_fixed<int, xt::xshape<S>> vec1 ; 
        xt::xtensor_fixed<int, xt::xshape<S>> vec2 ;
        xt::xtensor_fixed<int, xt::xshape<S>> result;

        vec1.fill(1);
        vec2.fill(2);

        xt::noalias(result) = vec1 + vec2;
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * S);
}
#endif


// Power of two rule
BENCHMARK_TEMPLATE(BM_RawSum, int32_t,	std::plus<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_RawSum, int32_t,	std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_RawSum, int32_t,	std::divides<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_RawSum, float,	std::plus<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_RawSum, float,	std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_RawSum, float,	std::divides<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);

BENCHMARK_TEMPLATE(BM_AlignedAllocSum, int32_t,	std::plus<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, int32_t,	std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, int32_t,	std::divides<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, float,	std::plus<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, float,	std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_AlignedAllocSum, float,	std::divides<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);

BENCHMARK_TEMPLATE(BM_VectorSum, int32_t,	std::plus<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_VectorSum, int32_t, 	std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_VectorSum, int32_t, 	std::divides<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_VectorSum, float,		std::plus<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_VectorSum, float, 	std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_VectorSum, float, 	std::divides<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);

#ifdef XBENCHMARK_USE_XTENSOR
BENCHMARK_TEMPLATE(BM_XArraySum, int32_t,	std::plus<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArraySum, int32_t,	std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArraySum, int32_t,	std::divides<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArraySum, float, 	std::plus<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArraySum, float, 	std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XArraySum, float, 	std::divides<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);

BENCHMARK_TEMPLATE(BM_XTensorSum, int32_t,	std::plus<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSum, int32_t,	std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSum, int32_t,	std::divides<	int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSum, float,	std::plus<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSum, float,	std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSum, float,	std::divides<	float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);


BENCHMARK_TEMPLATE(BM_XTensorSumEval, int32_t,      std::plus<      int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSumEval, int32_t,      std::multiplies<int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSumEval, int32_t,      std::divides<   int32_t>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSumEval, float,        std::plus<      float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSumEval, float,        std::multiplies<float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_XTensorSumEval, float,        std::divides<   float>)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);

#endif


/**
// --> Not really dynamic here ...
#ifdef XBENCHMARK_USE_XTENSOR
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
**/


BENCHMARK_MAIN();




