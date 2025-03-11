#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>

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
    while (i < n1) {
        data[k] = L[i];
        ++i;
        ++k;
    }
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
    const __m512i one = _mm512_set1_epi64(1);
#define BATCH_SIZE 8
#pragma omp parallel for
    for (size_t batch = 0; batch < size - BATCH_SIZE; batch += BATCH_SIZE) {
        // 加载查询批次
        const __m256 key = _mm256_loadu_ps(query + batch);
        __m512i left = _mm512_set1_epi64(0);
        __m512i right = _mm512_set1_epi64(size);
        while (_mm512_cmp_epi64_mask(left, right, _MM_CMPINT_LT)) {
            // 计算中间值
            const __m512i mid = _mm512_add_epi64(left,
                _mm512_srai_epi64(_mm512_sub_epi32(right, left), 1)
            );
            // 从data中获取当前值
            __m256 current = _mm512_i64gather_ps(mid, data, sizeof(float));
            // 比较当前值和key
            const __mmask8 mask_lt = _mm256_cmp_ps_mask(current, key, _MM_CMPINT_LT);
            const __mmask8 mask_ge = mask_lt ^ 0xff;
            const __m512i mid_plus1 = _mm512_add_epi64(mid, one);
            // 更新left和right
            left = _mm512_mask_blend_epi64(mask_lt, left, mid_plus1);
            right = _mm512_mask_blend_epi64(mask_ge, right, mid);
        }

        // 存储结果
        _mm512_storeu_epi64(result + batch, left);
    }

    // 处理剩余元素
    for (size_t i = size - BATCH_SIZE; i < size; ++i) {
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