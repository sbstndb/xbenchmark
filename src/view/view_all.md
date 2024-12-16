## Purpose 
Assume you want to use Views to execute some soperations on predefined indexes.
We want to benchmark the overhead of these views implementations vs without any views.

Then, we have some benchmarks without any views, like VIEW_all_raw and VIEW_all_aligned or VIEW_all_xtensor. THey are taken for reference.
 And, we have view based operations like VIEW_all_strided. We hope find a View implementation that is as fast as the unview one ...


The operation done is `result = vec1 + vec2` for each value of them. In this case, the concept of views is irrevelent but still applied for benchmarking purpose

- VIEW_all_raw : raw implementation without views. Just a simple loop.

- VIEW_all_aligned : same but with aligned allocation. Aligned allocations enables better performance.

- VIEW_all_aligned_masked : we implement a mask array which represents a minimal implementation of masked view driven by a branch condition. 

- VIEW_all_xarray : `xt::xarray` implementation with `xt::views(all)` views.

- VIEW_all_xtensor : `xt::xtensor` implementation with `xt::views(all)` views.

- VIEW_all_xtensor_strided : `xt::xtensor` implementation with `xt::strided_views(all)` views.

- VIEW_all_xtensor_strided_range : same but using a range (I don't remember why I do this test)

- VIEW_all_xtensor_masked : `xt::xtensor` imlementation with `xt::masked_view` view. 

- VIEW_all_xtensor_masked_2 : same but view used in the result side.

- VIEW_all_xtensor_raw_masked : `xt::xtensor` imlementation with custom raw mask view. 



# Observations
- `xt::masked_view` it **VERY SLOW**, much more than the naive raw implementation. I suppose a lack of vectorization while it is easy to enable it in a raw way... As a proof, timings from VIEW_all_aligned_masked and VIEW_all_xtensor_raw_masked are the same : the slowness is all about the `xt::masked_view` and not the `xt` container itself.
- The raw masked view is not fast and can be improved. This is probably due to branch conditions that leads to bad vectorization. 
- in the case of `xarray` and `xtensor` that uses `xt::view`, the performance is very great. **Nevertheless** Do not hope high performance for tiny views... (less than 16 elements) 
- Unfortunately, it seems that the `xt::strided_view` is not as good as the `xt::view` one by a factor 2. Maybe it's because i disable XSIMD ? hummm.... The performance is really poor for tiny views.
- `xt::range` performs a little bit slower than `xt::all()` but it is negligable.

# Improvements
There are a huge potential improvements in `masked_view` and also in `strided_view`. I have to benchmark with XSIMD enable but if the performancd is'nt here in this case, it means that there is a lack of vectorization. masked_view is really easy to vectorize. I have to do an intrinsic demonstrator. For `strided_view` also it is easy by using pre defined instructions during the operations.



# Result

```
[200~--------------------------------------------------------------------------------------------------------
Benchmark                                              Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------
VIEW_all_raw<float>/16                              2.03 ns         2.02 ns    332567165 items_per_second=7.90347G/s
VIEW_all_raw<float>/128                             10.2 ns         10.2 ns     71958370 items_per_second=12.5677G/s
VIEW_all_raw<float>/16384                           3039 ns         3037 ns       230532 items_per_second=5.39467G/s
VIEW_all_raw<float>/2097152                       960233 ns       959881 ns          731 items_per_second=2.1848G/s
VIEW_all_aligned<float>/16                          2.19 ns         2.19 ns    319218466 items_per_second=7.29327G/s
VIEW_all_aligned<float>/128                         7.96 ns         7.96 ns     88136508 items_per_second=16.086G/s
VIEW_all_aligned<float>/16384                       2594 ns         2594 ns       270004 items_per_second=6.31553G/s
VIEW_all_aligned<float>/2097152                   975284 ns       975019 ns          741 items_per_second=2.15088G/s
VIEW_all_aligned_masked<float>/16                   13.5 ns         13.5 ns     52145694 items_per_second=1.18938G/s
VIEW_all_aligned_masked<float>/128                   114 ns          114 ns      6145383 items_per_second=1.12269G/s
VIEW_all_aligned_masked<float>/16384               13238 ns        13232 ns        52924 items_per_second=1.23819G/s
VIEW_all_aligned_masked<float>/2097152           2118492 ns      2117406 ns          339 items_per_second=990.435M/s
VIEW_all_xarray<float>/16                           20.2 ns         20.2 ns     34643825 items_per_second=791.463M/s
VIEW_all_xarray<float>/128                          23.3 ns         23.3 ns     30911133 items_per_second=5.50055G/s
VIEW_all_xarray<float>/16384                        4275 ns         4273 ns       164490 items_per_second=3.83408G/s
VIEW_all_xarray<float>/2097152                   1026720 ns      1026427 ns          730 items_per_second=2.04316G/s
VIEW_all_xtensor<float>/16                          6.67 ns         6.67 ns    103581689 items_per_second=2.39979G/s
VIEW_all_xtensor<float>/128                         13.6 ns         13.6 ns     51874086 items_per_second=9.43707G/s
VIEW_all_xtensor<float>/16384                       4828 ns         4827 ns       144950 items_per_second=3.39417G/s
VIEW_all_xtensor<float>/2097152                  1021809 ns      1021464 ns          673 items_per_second=2.05309G/s
VIEW_all_xtensor_strided<float>/16                   138 ns          138 ns      5102203 items_per_second=116.362M/s
VIEW_all_xtensor_strided<float>/128                  142 ns          142 ns      4942327 items_per_second=900.02M/s
VIEW_all_xtensor_strided<float>/16384               4671 ns         4670 ns       149942 items_per_second=3.5081G/s
VIEW_all_xtensor_strided<float>/2097152          1018010 ns      1017724 ns          678 items_per_second=2.06063G/s
VIEW_all_xtensor_strided_range<float>/16             144 ns          144 ns      4845234 items_per_second=110.914M/s
VIEW_all_xtensor_strided_range<float>/128            148 ns          148 ns      4760033 items_per_second=865.312M/s
VIEW_all_xtensor_strided_range<float>/16384         4689 ns         4685 ns       149330 items_per_second=3.49675G/s
VIEW_all_xtensor_strided_range<float>/2097152     952896 ns       952673 ns          737 items_per_second=2.20133G/s
VIEW_all_xtensor_masked<float>/16                    149 ns          149 ns      4727526 items_per_second=107.705M/s
VIEW_all_xtensor_masked<float>/128                   492 ns          491 ns      1420750 items_per_second=260.45M/s
VIEW_all_xtensor_masked<float>/16384               49272 ns        49259 ns        14219 items_per_second=332.608M/s
VIEW_all_xtensor_masked<float>/2097152          24562172 ns     24554709 ns           28 items_per_second=85.4073M/s
VIEW_all_xtensor_masked_2<float>/16                 66.9 ns         66.9 ns     10498363 items_per_second=239.073M/s
VIEW_all_xtensor_masked_2<float>/128                 293 ns          293 ns      2387139 items_per_second=436.533M/s
VIEW_all_xtensor_masked_2<float>/16384             33581 ns        33557 ns        21319 items_per_second=488.248M/s
VIEW_all_xtensor_masked_2<float>/2097152         6510758 ns      6505962 ns           93 items_per_second=322.343M/s
VIEW_all_xtensor_raw_masked<float>/16               15.2 ns         15.2 ns     46156139 items_per_second=1.05233G/s
VIEW_all_xtensor_raw_masked<float>/128               120 ns          120 ns      5822324 items_per_second=1.06856G/s
VIEW_all_xtensor_raw_masked<float>/16384           15072 ns        15061 ns        43734 items_per_second=1.08787G/s
VIEW_all_xtensor_raw_masked<float>/2097152       2052688 ns      2051398 ns          340 items_per_second=1.0223G/s

```

