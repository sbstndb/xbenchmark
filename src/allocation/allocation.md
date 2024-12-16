## Purpose
Analysis of allocation time for dynamic structures + time taken to fill it with value 1.0. 

# ALLOC_raw : standard malloc as reference

# ALLOC_aligned : standard aligned alloc as reference

# ALLOC_std_vector : std::vector allocation

# ALLOC_xarray : xt::xarray allocation

# ALLOC_xtensor : xt::xtensor allocation


# Results : 
For large arrays around 16k values or more, the difference between containers is negligable. Nevertheless, for small arrays around 4 or 128 values, raw and aligned allocations are faster by a factor of 1.3-1.4. Furthermore, xt::xarray is slower than xt::xtensor and xt::xtensor is slighly slower than std::array.




