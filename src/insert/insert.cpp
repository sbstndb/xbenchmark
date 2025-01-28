#include <benchmark/benchmark.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>

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


const int MS = 4 ; // Min_size of arrays
const int RM = 64 ; /// RangeMultiplier
const int PS = 12 ; // pow size


// structure de données à insérer dans les cas _struct
template <int S>
struct data {
	int value ; 
	char padding[S] ; 
};


// Mesurer le temps de génération des nombres aléatoirtes
// On s'attend à un cout de génération faible
// Permet de vérifier qur nos mesures ont du sens lorsque le cout de génération est relativement faible
void INSERT_timer(benchmark::State& state) {
        const int size = state.range(0);  // Vector size defined by benchmark range

        std::random_device rd ;
        std::mt19937 gen(rd()) ;
        std::uniform_int_distribution<> distrib(0, 10000) ;

        for (auto _ : state) {
                for (int i = 0 ; i < size ; i++){		
	        	int randomValue = distrib(gen) ;
		}
        }
        // report throughput
        state.SetItemsProcessed(state.iterations() * size);
}



// inserer des entiersd aléatoires dans une std::map
// On s'attend à un cout d'insertion faible
void INSERT_map(benchmark::State& state) {
	const int size = state.range(0);  // Vector size defined by benchmark range

	std::random_device rd ; 
	std::mt19937 gen(rd()) ; 
	std::uniform_int_distribution<> distrib(0, 10000) ; 

	for (auto _ : state) {
		std::map<int, int> map ;
		for (int i = 0 ; i < size ; i++){
			int randomValue = distrib(gen) ; 
			map[randomValue] = randomValue ; 
		}
	}
	// report throughput
	state.SetItemsProcessed(state.iterations() * size);
}

// inserer des entiers aléatoires dans une std::unordered_map
// On s'attend à un cout d'insertion encore plus faible
// Utilisée à titre de comparaison pour mettre à défaut le mythe (ou non) du 
// "Il faut utiliser une unoreded_map en AMR". 
void INSERT_unordered_map_unsorted(benchmark::State& state) {
        const int size = state.range(0);  // Vector size defined by benchmark range


        std::random_device rd ;
        std::mt19937 gen(rd()) ;
        std::uniform_int_distribution<> distrib(0, 10000) ;

        for (auto _ : state) {
		std::unordered_map<int, int> map ;
		for (int i = 0 ; i < size ; i++){
	                int randomValue = distrib(gen) ;
	                map[randomValue] = randomValue ;
		}
	}
        // report throughput
        state.SetItemsProcessed(state.iterations() * size);
}


// inserer des entiers aléatoires dans une std::vector
// Principe NAIF : on décalle (copie) à chaque insertion les éléments à droite. 
// On s'attend à un cout d'insertion fort
// Utilisée pour voir dans quelle mesure on peut remplacer la std::map via la vectorisation et l'alignement mémoire

void INSERT_vector_insert(benchmark::State& state) {
        const int size = state.range(0);  // Vector size defined by benchmark range


        std::random_device rd ;
        std::mt19937 gen(rd()) ;
        std::uniform_int_distribution<> distrib(0, 10000) ;

        for (auto _ : state) {
		std::vector<int> vector ;
		for (int i = 0 ; i < size ; i++){
	                int randomValue = distrib(gen) ;
			auto it = std::lower_bound(vector.begin(), vector.end() , randomValue) ; 
			vector.insert(it, randomValue) ;
		}
		benchmark::DoNotOptimize(vector); 	

        }
        // report throughput
        state.SetItemsProcessed(state.iterations() * size);
}

// à partir d'ici, on fait la même chose mais sur des struct, pour mieux representer notre cas d'usage sur samurai. 


template <int S>
void INSERT_map_struct(benchmark::State& state) {
        const int size = state.range(0);  // Vector size defined by benchmark range
	using data = data<S> ; 


        std::random_device rd ;
        std::mt19937 gen(rd()) ;
        std::uniform_int_distribution<> distrib(0, 10000) ;

        for (auto _ : state) {
                std::map<int, data> map ;
                for (int i = 0 ; i < size ; i++){
                        int randomValue = distrib(gen) ;
			data myData {randomValue, 1, 2, 3} ; 
                        map[randomValue] = myData ;
                }
        }
        // report throughput
        state.SetItemsProcessed(state.iterations() * size);
}

template <int S>
void INSERT_vector_insert_struct(benchmark::State& state) {
        const int size = state.range(0);  // Vector size defined by benchmark range
	using data = data<S> ; 

        std::random_device rd ;
        std::mt19937 gen(rd()) ;
        std::uniform_int_distribution<> distrib(0, 10000) ;

        for (auto _ : state) {
                std::vector<data> vector ;
                for (int i = 0 ; i < size ; i++){
                        int randomValue = distrib(gen) ;
                        auto it = std::lower_bound(vector.begin(), vector.end() , randomValue,
					[](const data& d, int value) { 
						return d.value < value;
					});
			data myData ;//{randomValue, 1, 2, 3} ;
			myData.value = randomValue ; 
                        vector.insert(it, myData) ;
                }
                benchmark::DoNotOptimize(vector);

        }
        // report throughput
        state.SetItemsProcessed(state.iterations() * size);
}



// Power of two rule
//
BENCHMARK(INSERT_timer)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(INSERT_map)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(INSERT_unordered_map_unsorted)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK(INSERT_vector_insert)->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;

BENCHMARK_TEMPLATE(INSERT_map_struct, 12     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(INSERT_vector_insert_struct, 12     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;

BENCHMARK_TEMPLATE(INSERT_map_struct, 1020     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;
BENCHMARK_TEMPLATE(INSERT_vector_insert_struct, 1020     )->Apply([](benchmark::internal::Benchmark* b) {CustomArguments(b, min, max, threshold1, threshold2);});;


BENCHMARK_MAIN();





