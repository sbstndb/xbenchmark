#include <array>
#include <benchmark/benchmark.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>

inline auto& f_array(std::array<double, 4>& a)
{
    std::array<double, 4> b;

    for (std::size_t i = 0; i < 4; ++i)
    {
        a[i] = 4 * b[i] + 1;
    }
    return a;
}

inline auto& f_xfixed_unroll(xt::xtensor_fixed<double, xt::xshape<4>>& a)
{
    xt::xtensor_fixed<double, xt::xshape<4>> b;

    for (std::size_t i = 0; i < 4; ++i)
    {
        a[i] = 4 * b[i] + 1;
    }
    return a;
}

inline auto& f_xfixed_lazy(xt::xtensor_fixed<double, xt::xshape<4>>& a)
{
    xt::xtensor_fixed<double, xt::xshape<4>> b;
    xt::noalias(a) = 4 * b + 1;
    return a;
}

static void BM_std_array(benchmark::State& state)
{
    std::array<double, 4> a;

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(f_array(a));
        benchmark::ClobberMemory();
    }
}

static void BM_xt_xfixed_unroll(benchmark::State& state)
{
    xt::xtensor_fixed<double, xt::xshape<4>> a;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(f_xfixed_unroll(a));
        benchmark::ClobberMemory();
    }
}

static void BM_xt_xfixed_lazy(benchmark::State& state)
{
    xt::xtensor_fixed<double, xt::xshape<4>> a;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(f_xfixed_lazy(a));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_std_array);
BENCHMARK(BM_xt_xfixed_unroll);
BENCHMARK(BM_xt_xfixed_lazy);
BENCHMARK_MAIN();
