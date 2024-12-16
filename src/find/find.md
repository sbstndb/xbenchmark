

## Purpose

Assume you have an **integer array** `A` of length `n`. Assue the values are **sorted**. 
You are **searching** for the **index** `i` where `A[i] = target`. You want this to be **fast**.


# `NAIVE_find` : naive implementation of find algorithm
- Advantages : 
    -   simple to implement
    -   can work for unaligned data structures like linked lsit
- Drawbacks :
    -   suffer from branching
    -   Cannot vectorize any operations due to the return statement on the for loop.

# `NOBREAK_find` : no branchy break implementation
In this case, we do not return the result on the branch condition `if`. Hence, the loop can vectorize. As a consequence, the algorithm always traverse the whole vector and do more operations.
- Advantages
    -   simple to implement
    -   great vectorization on aligned array
- Drawbacks
    -   bad if number to find is at the beginning 
- Note : you should reverse the for loop if the array could contains duplicates. 

# `COMPARE_find` : using add trick
We want better vectorisarion. We thus want to expose to the compiler explicit operations that can vectorize well. One of them are `compare`. Hence, this is one implementation that use explicit compare to find a number on a sorted array. The compare is able to vetorize. 
As the previous testcse named NOBREAK_find, you need to traverse the whole array. 
- Advantages :
    -   very good vectorization on aligned array
- Drawbacks
    -   less readable


# `STD_find` : c++ stl implementation
We use the c++ stl implementation of find. 
- Advantages : 
    - flexibility of use
- Drawbacks
    - poor performances


# `INTRINCIS_find` : intrinsic implementation using compare strategy
The intrinsic implementation use explicit SIMD registers. It uses `compare=` SIMD operations and return the value without having to traverse the whole array. As a result, it is the fastest implementation. 
- Advantages
    -   fastest implementation
- Drawbacks
    -   not poertable
    -   only for aligned arrays



## Results : 
Results seems to depend on the compiler we use. 

Here are the results of the benchmak : 
