/* test_gemv_host.c -- standalone, CUDA-free unit test for gemv_host.h.
 * Build:  gcc -std=c11 -I ../src -o /tmp/test_gemv_host test_gemv_host.c -lm
 * Run:    /tmp/test_gemv_host   (prints PASS per case, "ALL PASS" + exit 0 at the end;
 *                                 aborts via assert() with a nonzero exit on first failure)
 */
#include "../src/gemv_host.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

static void test_gen_csr_row_shape(void) {
    int *rowOff, *colInd; float *vals; long nnz;
    long S = 1024; double density = 0.05;
    int rc = gemv_gen_csr(S, density, &rowOff, &colInd, &vals, &nnz, 42);
    assert(rc == 0);
    int expected_npr = (int)llround(density * (double)S);
    assert(rowOff[0] == 0);
    for (long r = 0; r <= S; r++) {
        assert(rowOff[r] == r * expected_npr);
    }
    assert(nnz == (long)expected_npr * S);
    for (long i = 0; i < nnz; i++) {
        assert(colInd[i] >= 0 && colInd[i] < S);
    }
    free(rowOff); free(colInd); free(vals);
    printf("test_gen_csr_row_shape: PASS\n");
}

static void test_gen_csr_density_floor(void) {
    /* density so small that npr would round to 0 -- must clamp to 1 nonzero/row, not 0 */
    int *rowOff, *colInd; float *vals; long nnz;
    long S = 1024; double density = 0.0001; /* 0.1024 nonzeros/row, rounds to 0 */
    int rc = gemv_gen_csr(S, density, &rowOff, &colInd, &vals, &nnz, 7);
    assert(rc == 0);
    assert(nnz == S); /* exactly 1 nonzero per row */
    free(rowOff); free(colInd); free(vals);
    printf("test_gen_csr_density_floor: PASS\n");
}

static void test_gen_csr_deterministic(void) {
    /* same seed -> same output; benchmark reproducibility depends on this */
    int *rowOff1, *colInd1; float *vals1; long nnz1;
    int *rowOff2, *colInd2; float *vals2; long nnz2;
    long S = 256; double density = 0.1;
    assert(gemv_gen_csr(S, density, &rowOff1, &colInd1, &vals1, &nnz1, 99) == 0);
    assert(gemv_gen_csr(S, density, &rowOff2, &colInd2, &vals2, &nnz2, 99) == 0);
    assert(nnz1 == nnz2);
    assert(memcmp(colInd1, colInd2, (size_t)nnz1 * sizeof(int)) == 0);
    assert(memcmp(vals1, vals2, (size_t)nnz1 * sizeof(float)) == 0);
    free(rowOff1); free(colInd1); free(vals1);
    free(rowOff2); free(colInd2); free(vals2);
    printf("test_gen_csr_deterministic: PASS\n");
}

static void test_mem_footprint_split_vs_replicas(void) {
    long max_mrows;
    /* fp32 in/out (4 bytes each), S=1000, G=4: split -> ceil(1000/4)=250 rows/GPU */
    long split_bytes = gemv_mem_footprint_bytes(4, 4, /*mode_replicas=*/0, 1000, 4, &max_mrows);
    assert(max_mrows == 250);
    long expect_split = (long)((size_t)250 * 1000 * 4 + (size_t)1000 * 4 + (size_t)250 * 4);
    assert(split_bytes == expect_split);

    long replicas_bytes = gemv_mem_footprint_bytes(4, 4, /*mode_replicas=*/1, 1000, 4, &max_mrows);
    assert(max_mrows == 1000);
    long expect_replicas = (long)((size_t)1000 * 1000 * 4 + (size_t)1000 * 4 + (size_t)1000 * 4);
    assert(replicas_bytes == expect_replicas);

    assert(replicas_bytes > split_bytes);
    printf("test_mem_footprint_split_vs_replicas: PASS\n");
}

static void test_mem_footprint_remainder_rows(void) {
    /* S=1001, G=4 -> base=250, rem=1 -> GPU 0 gets the +1 remainder row (251) */
    long max_mrows;
    gemv_mem_footprint_bytes(4, 4, 0, 1001, 4, &max_mrows);
    assert(max_mrows == 251);
    printf("test_mem_footprint_remainder_rows: PASS\n");
}

static void test_peak_gbps_is_a100_sxm4_40gb(void) {
    assert(gemv_peak_gbps() == 1555.0);
    printf("test_peak_gbps_is_a100_sxm4_40gb: PASS\n");
}

static void test_parse_longs_basic(void) {
    long out[8];
    int n = gemv_parse_longs("1024,2048,4096", out, 8);
    assert(n == 3);
    assert(out[0] == 1024 && out[1] == 2048 && out[2] == 4096);
    printf("test_parse_longs_basic: PASS\n");
}

static void test_parse_longs_truncates_at_max(void) {
    long out[2];
    int n = gemv_parse_longs("1,2,3,4,5", out, 2);
    assert(n == 2);
    assert(out[0] == 1 && out[1] == 2);
    printf("test_parse_longs_truncates_at_max: PASS\n");
}

int main(void) {
    test_gen_csr_row_shape();
    test_gen_csr_density_floor();
    test_gen_csr_deterministic();
    test_mem_footprint_split_vs_replicas();
    test_mem_footprint_remainder_rows();
    test_peak_gbps_is_a100_sxm4_40gb();
    test_parse_longs_basic();
    test_parse_longs_truncates_at_max();
    printf("ALL PASS\n");
    return 0;
}
