#include <algorithm>
#include <immintrin.h>



int find_equal_naive(int* vector, int size, int value){
        int index = -1 ;
        for (int i = 0 ; i < size ; i++){
                if (vector[i] == value){
                        index = i ;
                        return index;
                }
        }
        return index ;
}


int find_equal_no_break(int* vector, int size, int value){
        int index = -1 ;
        for (int i = 0 ; i < size ; i++){
                if (vector[i] == value){
                        index = i ;
                        // Note : work if we do not have twice the value.
                        // If the array is sorted, you can travel the vector
                        //      backwards and then get the first value
                }
        }
        return index ;
}


int find_equal_compare(int* vector, int size, int value){
        int index = -1 ;
        for (int i = 0 ; i < size ; i++){
                index += (vector[i] < value) ;
        }
        return index ;
}


int find_equal_std_find(int* vector, int size, int value){
        int *pindex = std::find(vector, vector + size, value) ;
        // Theorically, we have to verify vector[pindex] == value ...
        return pindex - vector ;
}


int find_equal_intrinsic(int* vector, int size, int value){
        // experimental
        int index = -1 ;
        __m256i target = _mm256_set1_epi32(value) ;
// In my cpu : optimal unrolling around 2 SIMD finds per loop
        for (int i = 0 ; i < size ; i+=16){//avx2
                __m256i chunk  = _mm256_loadu_si256((const __m256i_u*)&vector[i]); // load vector in AVX reg
                __m256i chunk2 = _mm256_loadu_si256((const __m256i_u*)&vector[i+8]); // load vector in AVX reg
                                                                                      //
                __m256i cmp  = _mm256_cmpeq_epi32(chunk , target);
                __m256i cmp2 = _mm256_cmpeq_epi32(chunk2, target);

                int mask_result  = _mm256_movemask_epi8(cmp );
                int mask_result2 = _mm256_movemask_epi8(cmp2);

                if (mask_result !=0){
                        int index = __builtin_ctz(mask_result) / 4 ;
                        return i+index ;
                }
                if (mask_result2 !=0){
                        int index = __builtin_ctz(mask_result2) / 4 ;
                        return i+index+8 ;
                }
        }
        return index ;
}
