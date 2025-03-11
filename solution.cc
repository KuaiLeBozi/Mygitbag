#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <algorithm>

void optimized_pre_phase1(size_t) {}

void optimized_post_phase1() {}

void optimized_pre_phase2(size_t) {}

void optimized_post_phase2() {}

void merge(float* data, size_t left, size_t mid, size_t right) {
    size_t n1 = mid - left + 1;
    size_t n2 = right - mid;

    float* L = (float*)malloc(n1 * sizeof(float));
    float* R = (float*)malloc(n2 * sizeof(float));

    #pragma omp parallel for simd
    for (size_t i = 0; i < n1; ++i)
        L[i] = data[left + i];
    #pragma omp parallel for simd
    for (size_t j = 0; j < n2; ++j)
        R[j] = data[mid + 1 + j];

    size_t i = 0, j = 0, k = left;
    #pragma omp parallel for simd
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            data[k] = L[i];
            ++i;
        } else {
            data[k] = R[j];
            ++j;
        }
        ++k;
    }
    #pragma omp parallel for simd
    while (i < n1) {
        data[k] = L[i];
        ++i;
        ++k;
    }
    #pragma omp parallel for simd
    while (j < n2) {
        data[k] = R[j];
        ++j;
        ++k;
    }

    free(L);
    free(R);
}

void mergeSort(float* data, size_t left, size_t right) {
    if (right - left <= 1000) {
        std::sort(data + left, data + right + 1);
        return;
    }

    if (left < right) {
        size_t mid = left + (right - left) / 2;
        #pragma omp taskgroup
        {
            #pragma omp task shared(data)
            mergeSort(data, left, mid);
            #pragma omp task shared(data)
            mergeSort(data, mid + 1, right);
            #pragma omp taskyield
        }

        merge(data, left, mid, right);
    }
}

void optimized_do_phase1(float* data, size_t size) {
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(data, 0, size - 1);
    }
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