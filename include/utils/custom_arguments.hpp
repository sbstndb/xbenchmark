#pragma once
#include <benchmark/benchmark.h>

// Déclaration avec valeurs par défaut basées sur des macros
void CustomArguments(
    benchmark::internal::Benchmark* b,
    int start = 1,          // Default: 1
    int end = 1000000,    // Default: 1048576
    int threshold1 = 1024,      // Default: 32
    int threshold2 = 4096
);

