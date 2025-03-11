#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <omp.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <utility>
#include <stack>

void optimized_pre_phase1(size_t) {}

void optimized_post_phase1() {}

void optimized_pre_phase2(size_t) {}

void optimized_post_phase2() {}

void bitonic_merge(float* data, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        #pragma omp parallel for simd
        for (int i = low; i < low + k; i++) {
            float val_i = data[i];
            float val_ik = data[i + k];
            if (dir == (val_i > val_ik)) {
                // 手动交换变量
                data[i] = val_ik;
                data[i + k] = val_i;
            }
        }
        bitonic_merge(data, low, k, dir);
        bitonic_merge(data, low + k, k, dir);
    }
}

void bitonic_sort(float* data, int n) {
    for (int k = 2; k <= n; k = 2 * k) {
        for (int j = k / 2; j > 0; j = j / 2) {
            #pragma omp parallel for simd
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float val_i = data[i];
                    float val_ixj = data[ixj];
                    if ((i & k) == 0 && val_i > val_ixj) {
                        // 手动交换变量
                        data[i] = val_ixj;
                        data[ixj] = val_i;
                    }
                    if ((i & k) != 0 && val_i < val_ixj) {
                        // 手动交换变量
                        data[i] = val_ixj;
                        data[ixj] = val_i;
                    }
                }
            }
        }
    }
}

void optimized_do_phase1(float* data, size_t size) {
    size_t n = 1;
    while (n < size) n <<= 1;

    float* extended_data = new float[n];
    #pragma omp parallel for 
    for (size_t i = 0; i < size; ++i) {
        extended_data[i] = data[i];
    }
    #pragma omp parallel for 
    for (size_t i = size; i < n; ++i) {
        extended_data[i] = std::numeric_limits<float>::max();
    }

    bitonic_sort(extended_data, n);

    #pragma omp parallel for 
    for (size_t i = 0; i < size; ++i) {
        float temp = extended_data[i];
        data[i] = temp;
    }

    delete[] extended_data;
}

void optimized_do_phase2(size_t* result, float* data, float* query, size_t size) {
    #pragma omp parallel for schedule(static) shared(result, data, query) 
    for (size_t i = 0; i < size; ++i) {
        size_t l = 0, r = size;
        float query_val = query[i];
        while (l < r) {
            size_t m = l + (r - l) / 2;
            float data_val = data[m];
            if (data_val < query_val) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        result[i] = l;
    }
}
