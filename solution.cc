#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <omp.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <utility>

void optimized_pre_phase1(size_t) {}

void optimized_post_phase1() {}

void optimized_pre_phase2(size_t) {}

void optimized_post_phase2() {}

void bitonic_merge(float* data, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        #pragma omp parallel for simd
        for (int i = low; i < low + k; i++) {
            if (dir == (data[i] > data[i + k])) {
                std::swap(data[i], data[i + k]);
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
                    if ((i & k) == 0 && data[i] > data[ixj]) {
                        std::swap(data[i], data[ixj]);
                    }
                    if ((i & k) != 0 && data[i] < data[ixj]) {
                        std::swap(data[i], data[ixj]);
                    }
                }
            }
        }
    }
}

void optimized_do_phase1(float* data, size_t size) {
    size_t n = 1;
    while (n < size) n <<= 1;

    float* padded_data = new float[n];
    for (size_t i = 0; i < size; ++i) {
        padded_data[i] = data[i];
    }
    for (size_t i = size; i < n; ++i) {
        padded_data[i] = std::numeric_limits<float>::max();
    }

    bitonic_sort(padded_data, n);

    for (size_t i = 0; i < size; ++i) {
        data[i] = padded_data[i];
    }

    delete[] padded_data;
}

void optimized_do_phase2(size_t* result, float* data, float* query, size_t size) {
    #pragma omp parallel for schedule(static) shared(data, query, result) 
    for (size_t i = 0; i < size; ++i) {
        size_t l = 0, r = size;
        while (l < r) {
            size_t m = l + (r - l) / 2;
            if (data[m] < query[i]) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        result[i] = l;
    }
}