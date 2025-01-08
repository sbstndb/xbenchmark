#include <algorithm>

int find_gt_std_lower_bound(int* vector, int size, int value){
        int *pindex = std::lower_bound(vector, vector + size, value,[](int a, int b) { return a > b;}) ;
        // Theorically, we have to verify vector[pindex] == value ...
        return pindex - vector ;
}

