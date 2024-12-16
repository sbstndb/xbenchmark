# xbenchmark

A benchmarking suite for common operations in scientific computing, comparing different implementation approaches. It helps me identify weaknesses in certain C++ implementations and optimize them for maximum performance, particularly within the exaScale `Samurai` project.

## Overview

`xbenchmark` is a C++ project designed to benchmark various operations commonly encountered in numerical and scientific computing. It aims to compare the performance of these operations when implemented using different methods, including:

-   **Plain C and C++**: Standard C and C++ implementations, optimized for basic efficiency.
-   **Intrinsics**: Leveraging processor-specific instruction set extensions (e.g., AVX2) for potential performance gains.
-   **xtensor**: Using the `xtensor` library for multi-dimensional array manipulation.
-   **Eigen**: Utilizing the `Eigen` library, known for high-performance linear algebra operations.

The benchmarked operations include:

-   **Views**: Creation and manipulation of views on data.
-   **BLAS1 Operations**: Basic vector operations (e.g., addition, scaling).
-   **BLAS2 Operations**: Basic matrix-vector operations.
-   **Linear Search**: Searching for elements within an array.
-   ANd so on.

This project provides insights into the trade-offs between different implementation methods and highlights the potential benefits of using optimized libraries or intrinsic functions.

## Compilation

This project uses CMake for building. You will need a C++ compiler that supports C++17 or later.

### Prerequisites

-   CMake (version 3.10 or higher)
-   A C++ compiler supporting C++17 (e.g., GCC, Clang. MSVC is untested)

### Optional Dependencies

-   **xtensor**: If you want to benchmark using `xtensor`, you must install it first. Instructions can be found on the [xtensor GitHub page](https://github.com/xtensor-stack/xtensor).
-   **Eigen**: If you want to benchmark using `Eigen`, you must install it first. Instructions can be found on the [Eigen website](https://eigen.tuxfamily.org/index.php?title=Main_Page).

### CMake Configuration

To configure the project, use CMake with the following options:

-   `-DXBENCHMARK_USE_IMMINTRIN=ON|OFF`: Enables or disables the use of intrinsics. When enabled, the code will attempt to use AVX2 instructions if available. Default is `OFF`. If your CPU doesn't support AVX2 instructons, then you should disable this option.
-   `-DXBENCHMARK_USE_XTENSOR=ON|OFF`: Enables or disables the use of `xtensor`. When enabled, the code will benchmark against operations implemented using `xtensor`. Default is `OFF`.

### Example Build Process

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd xbenchmark
    ```
2.  **Create a Build Directory:**

    ```bash
    mkdir build
    cd build
    ```
3.  **Configure with CMake:**

    ```bash
    cmake .. -DXBENCHMARK_USE_IMMINTRIN=ON -DXBENCHMARK_USE_XTENSOR=ON
    ```
    
    _Adjust the flags as needed_
    
    You can replace `ON` with `OFF` if you don't wish to use intrinsics or xtensor.
4.  **Build the Project:**

    ```bash
    cmake --build .
    ```

### Running the Benchmarks

After a successful build, many executables will be generated in the build/src directory. To execute, simply run the program:

```bash
./std/<benchmark_folder>/benchark
```

The program will output benchmark results for each operation, comparing the different implementation methods.

## Details

The project is structured in the following way:

-   `src/`: contains the source files for the operations being benchmarked, with different implementations.
-   `include/`: contains the header files.
-   `benchmarks/`: contains the benchmarking code, using a suitable benchmarking framework (e.g. Google Benchmark or nanobench).
-   `CMakeLists.txt`: CMake configuration file.

