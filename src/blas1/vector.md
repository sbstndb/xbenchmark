# Purpose

Assume you have three vectors `a`, `b` and `c` of size `n`. 
The size `n` could by tiny (around 4) or very large (around 1M).
We are looking for a container ( or a way) to achieve max performance for basic vector operations like `c = a+b` or `c = a*b`.

The naive loop is something like : 
```
for i in [0,n] do c[i] = a[i] + b[i]
```

These operations can be fully vectorized. Hence, this benchmark exposes how well the compiler can vectorize some container libraries. 

We do not time the allocation time but only the for loop. You can find an allocation benchmark in the project.


# BLAS1_op_raw : using naive malloc and implementation

# BLAS1_op_aligned : using aligned alloc and naive implementation

# BLAS1_op_std_vector : using `std::vector` container

# BLAS1_op_xarray : using `xt::xarray` container
It uses `xt::noalias` and `xt::eval` that can change the performance

# BLAS1_op_xtensor : using `xt::xtensor` container
It uses `xt::noalias` that can change the performanc
e

# BLAS1_op_xtensor_eval : using `xt::xtensor` using `xt::eval`
It uses `xt::eval` that can change performance

# BLAS1_op_xtensor_fixed : using static `xt::xtensor_fixed`

# BLAS1_op_xtensor_fixed_noalias : using static `xt::xtensor_fixed`
It uses `xt::noalias` that can change performance


## Result
We found that xtensor is slower for very short containers (4 values) but average otherwise. 
Furthermore, the `xt::eval` slow down to lot, in all cases. We really need to investiguate this part.
`xt::xtensor_fixed` is better for very tiny arrays but worst for lrge arrays. Hence, I suggest using them for tiny arrays according to this analysis. 

```
------------------------------------------------------------------------------------------------------------------
Benchmark                                                        Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------------------
BLAS1_op_raw<float, std::plus< float>>/4                      1.81 ns         1.81 ns    362694652 items_per_second=2.20521G/s
BLAS1_op_raw<float, std::plus< float>>/128                    9.88 ns         9.87 ns     66263587 items_per_second=12.9623G/s
BLAS1_op_raw<float, std::plus< float>>/16384                  3005 ns         3004 ns       225223 items_per_second=5.45381G/s
BLAS1_op_raw<float, std::plus< float>>/2097152              956304 ns       956015 ns          741 items_per_second=2.19364G/s
BLAS1_op_aligned<float, std::plus< float>>/4                  2.10 ns         2.10 ns    330660651 items_per_second=1.90231G/s
BLAS1_op_aligned<float, std::plus< float>>/128                7.99 ns         7.98 ns     88004935 items_per_second=16.0372G/s
BLAS1_op_aligned<float, std::plus< float>>/16384              2379 ns         2378 ns       294004 items_per_second=6.88954G/s
BLAS1_op_aligned<float, std::plus< float>>/2097152          916576 ns       916243 ns          768 items_per_second=2.28886G/s
BLAS1_op_std_vector<float, std::plus< float>>/4               1.61 ns         1.61 ns    433608932 items_per_second=2.48245G/s
BLAS1_op_std_vector<float, std::plus< float>>/128             7.78 ns         7.78 ns     90000564 items_per_second=16.4599G/s
BLAS1_op_std_vector<float, std::plus< float>>/16384           3305 ns         3304 ns       211885 items_per_second=4.95888G/s
BLAS1_op_std_vector<float, std::plus< float>>/2097152       909209 ns       909029 ns          774 items_per_second=2.30702G/s
BLAS1_op_xarray<float, std::plus< float>>/4                   36.9 ns         36.9 ns     18962830 items_per_second=108.441M/s
BLAS1_op_xarray<float, std::plus< float>>/128                 36.9 ns         36.9 ns     18964602 items_per_second=3.46732G/s
BLAS1_op_xarray<float, std::plus< float>>/16384               36.8 ns         36.8 ns     18923131 items_per_second=444.99G/s
BLAS1_op_xarray<float, std::plus< float>>/2097152             36.8 ns         36.8 ns     18935712 items_per_second=57.007T/s
BLAS1_op_xtensor<float, std::plus< float>>/4                  3.23 ns         3.22 ns    216453315 items_per_second=1.24061G/s
BLAS1_op_xtensor<float, std::plus< float>>/128                9.24 ns         9.24 ns     75807238 items_per_second=13.8508G/s
BLAS1_op_xtensor<float, std::plus< float>>/16384              3015 ns         3014 ns       232040 items_per_second=5.43546G/s
BLAS1_op_xtensor<float, std::plus< float>>/2097152          902177 ns       901915 ns          756 items_per_second=2.32522G/s
BLAS1_op_xtensor_eval<float, std::plus< float>>/4             24.5 ns         24.5 ns     28614813 items_per_second=163.149M/s
BLAS1_op_xtensor_eval<float, std::plus< float>>/128           32.7 ns         32.7 ns     21359420 items_per_second=3.91873G/s
BLAS1_op_xtensor_eval<float, std::plus< float>>/16384         4748 ns         4746 ns       147492 items_per_second=3.45192G/s
BLAS1_op_xtensor_eval<float, std::plus< float>>/2097152    1920507 ns      1919757 ns          349 items_per_second=1.0924G/s
BLAS1_op_xtensor_fixed<4>                                    0.806 ns        0.806 ns    866971304 items_per_second=4.96093G/s
BLAS1_op_xtensor_fixed<128>                                   30.1 ns         30.1 ns     23226041 items_per_second=4.24707G/s
BLAS1_op_xtensor_fixed<16384>                                 5036 ns         5035 ns       138895 items_per_second=3.2542G/s
BLAS1_op_xtensor_fixed_noalias<4>                            0.409 ns        0.409 ns   1000000000 items_per_second=9.78928G/s
BLAS1_op_xtensor_fixed_noalias<128>                           12.7 ns         12.7 ns     55174431 items_per_second=10.0923G/s
BLAS1_op_xtensor_fixed_noalias<16384>                         3051 ns         3051 ns       229301 items_per_second=5.37038G/s
```



## TODO 
- What is we operate between mixed containers like `xt::xtensor_fixed + std::vector` ? 
- Someone sugests that `xt::xtensor_fixed` is slow. I have to investiguate in what extent it is the case. 

