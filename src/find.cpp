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

const int MS = 2 ; // Min_size of arrays
const int RM = 2 ; /// RangeMultiplier
const int PS = 12 ; // pow size


const int stride = 4 ; 


int get_mid_index(int vector_size){
	return static_cast<int>(vector_size/2)- 1;
}

template <typename T>
void BM_NaiveLinearFind(benchmark::State& state) {
    const int vector_size = state.range(0);
    T* array = (T*) malloc ( sizeof(T) * vector_size ) ;
    // randomisation du vecteur
    // Remplir le tableau avec des valeurs aléatoires et le trier
    std::generate(array, array + vector_size, []() { return rand() % 10000; });
    std::sort(array, array + vector_size);

    int mid_index = get_mid_index(vector_size);
    int target = array[mid_index] ; // we want to iterate on one half of the vector to be fair :-)

    int index = - 1 ;
    for (auto _ : state) {
        index = -1 ;
        for (int i = 0 ; i < vector_size ; i++){
                if (array[i] == target){
                        index = i ;
			break; 
                }
        }

        benchmark::DoNotOptimize(index); // Prevent compiler optimizations
    }
    std::free(array);
    state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T>
void BM_NoExitLinearFind(benchmark::State& state) {
    const int vector_size = state.range(0);
    const int alignment = 64 ;
    T* array = static_cast<T*>(std::aligned_alloc(alignment, vector_size * sizeof(T)));
    // randomisation du vecteur
    // Remplir le tableau avec des valeurs aléatoires et le trier
    std::generate(array, array + vector_size, []() { return rand() % 10000; });
    std::sort(array, array + vector_size); 

    int mid_index = get_mid_index(vector_size);
    int target = array[mid_index] ; // we want to iterate on one half of the vector to be fair :-)


    int index = - 1 ; 
    for (auto _ : state) {
	index = -1 ; 
	for (int i = 0 ; i < vector_size ; i++){
		if (array[i] == target){
			index = i ; 
		}
	}	
        benchmark::DoNotOptimize(index); // Prevent compiler optimizations
    }
    std::free(array) ; 
    state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T>
void BM_StdFind(benchmark::State& state) {
    const int vector_size = state.range(0);
    std::vector<T> array(vector_size);
    // randomisation du vecteur
    // Remplir le tableau avec des valeurs aléatoires et le trier
    std::generate(array.begin(), array.end(), []() { return rand() % 10000; });
    std::sort(array.begin(), array.end());

    int mid_index = get_mid_index(vector_size);
    int target = array[mid_index] ; // we want to iterate on one half of the vector to be fair :-)


    int index = - 1 ;
    for (auto _ : state) {
        index = -1 ;

        auto it = std::find(array.begin(), array.end(), target);
        if (it != array.end() && *it == target) {	
            index = std::distance(array.begin(), it);
        }
        benchmark::DoNotOptimize(index); // Prevent compiler optimizations
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}


template <typename T>
void BM_LowerBoundFind(benchmark::State& state) {
    const int vector_size = state.range(0);
    std::vector<T> array(vector_size);
    // randomisation du vecteur
    // Remplir le tableau avec des valeurs aléatoires et le trier
    std::generate(array.begin(), array.end(), []() { return rand() % 10000; });
    std::sort(array.begin(), array.end());

    int mid_index = get_mid_index(vector_size);
    int target = array[mid_index] ; // we want to iterate on one half of the vector to be fair :-)


    int index = - 1 ;
    for (auto _ : state) {
        index = -1 ;

	auto it = std::lower_bound(array.begin(), array.end(), target);
        if (it != array.end() && *it == target) {
            index = std::distance(array.begin(), it);
        }
        benchmark::DoNotOptimize(index); // Prevent compiler optimizations
    }
    state.SetItemsProcessed(state.iterations() * vector_size);
}


#ifdef XBENCHMARK_USE_IMMINTRIN
// TODO : optimize this kernel : we want this to compile into avx mask instructions
// !!! T should be float in this experimental case
#endif




// Power of two rule
//
BENCHMARK_TEMPLATE(BM_NaiveLinearFind, uint32_t     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_NoExitLinearFind, uint32_t     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_StdFind, int32_t     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_TEMPLATE(BM_LowerBoundFind, uint32_t     )->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);



#ifdef XBENCHMARK_USE_IMMINTRIN
#endif


BENCHMARK_MAIN();




