#include <algorithm>

int find_equal_std_lower_bound(int* vector, int size, int value){
        int *pindex = std::lower_bound(vector, vector + size, value) ;
        // Theorically, we have to verify vector[pindex] == value ...
        return pindex - vector ;
}

