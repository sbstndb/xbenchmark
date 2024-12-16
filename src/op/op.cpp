#include <benchmark/benchmark.h>
#include <iostream>
#include <cmath>

const int MS = 2 ; // Min_size of arrays
const int RM = 2 ; /// RangeMultiplier
const int PS = 12 ; // pow size



void OP_square(benchmark::State& state){
        const int size = state.range(0) ;
	double a = 2.0 ; 
	double *b1 = new double ; 
        double *b2 = new double ;
        double *b3 = new double ;
        double *b4 = new double ;



        for (auto _ : state){
		*b1 = std::pow(a, 2.0);
                *b2 = std::pow(*b1, 2.0);
                *b3 = std::pow(*b2, 2.0);
                *b4 = std::pow(*b3, 2.0);
                benchmark::DoNotOptimize(b4);
		benchmark::ClobberMemory() ; 
        }
}

void OP_square_multiply(benchmark::State& state){
        const int size = state.range(0) ;
        double a = 2.0 ;
        double *b1 = new double ;
        double *b2 = new double ;
        double *b3 = new double ;
        double *b4 = new double ;

        for (auto _ : state){
                *b1 = a*a;
                *b2 = (*b1)*(*b1);
                *b3 = (*b2)*(*b2);
                *b4 = (*b3)*(*b3);
                benchmark::DoNotOptimize(b4);
                benchmark::ClobberMemory() ;
        }
}





BENCHMARK(OP_square)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK(OP_square_multiply)->RangeMultiplier(RM)->Range(MS << 0, 1 << PS);
BENCHMARK_MAIN() ; 

