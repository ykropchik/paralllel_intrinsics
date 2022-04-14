#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

double integrate(double a, double b, unsigned n) {
    double dx = (b - a) / n;
    __m256d Dx = _mm256_set1_pd(4*dx);
    __m256d x = _mm256_set_pd(a + 3*dx, a + 2*dx, a + dx, a);
    __m256d sum = _mm256_setzero_pd();

    for (unsigned i = 0; i < n; i++) {
        sum = _mm256_add_pd(sum, _mm256_mul_pd(x, x));
        x = _mm256_add_pd(x, Dx);
    }

    sum = _mm256_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(_mm_add_pd(_mm256_extractf128_pd(sum, 0), _mm256_extractf128_pd(sum, 1))) * dx;
}

void addMatrixSSE(const double* a, const double* b, std::size_t c, std::size_t r, double* result) {
    auto elementsInRegister = sizeof(__m256d)/sizeof(double);
    auto iterations = c*r/elementsInRegister;

    for(auto i = 0; i < iterations; i++) {
        __m256d aTerm = _mm256_load_pd(a);
        __m256d bTerm = _mm256_load_pd(b);
        __m256d sum = _mm256_add_pd(aTerm, bTerm);
        _mm256_storeu_pd(result, sum);

        a += elementsInRegister;
        b += elementsInRegister;
        result += elementsInRegister;
    }
}

void mulMatrixSSE(const double* a, const double* b, std::size_t cA, std::size_t rA, std::size_t cB, double* result) {
    if (rA & 3) {
        exit(-1);
    }

    auto rB = cA;

    for(auto i = 0; i < rA; i += 4) {
        for(auto j = 0; j < cB; j++) {
            auto tempRes = _mm256_setzero_pd();
            for (int k = 0; k < cA; k++) {
                auto tempA = _mm256_loadu_pd(&a[k*rA+i]);
                auto tempB = _mm256_set1_pd(b[j*rB+k]);
                tempRes = _mm256_add_pd(_mm256_mul_pd(tempA, tempB), tempRes);
            }
            _mm256_storeu_pd(&result[j*rA+i], tempRes);
        }
    }
}

void printRowMatrix(const double* a, unsigned c, unsigned r) {
    for(uint32_t y = 0; y < r; y++) {
        for(uint32_t x = 0; x < c; x++) {
            printf("%11f ", a[y * c + x]);
        }
        printf("\n");
    }
}

void printColumnMatrix(const double* a, unsigned c, unsigned r) {
    for(unsigned y = 0; y < c; y++) {
        for(unsigned x = 0; x < r; x++) {
            printf("%11f ", a[y * r + x]);
        }
        printf("\n");
    }
}

int main() {
    double  intRes = integrate(-1, 1, 1000000);
    // printf("Result: %f\n", intRes);

    uint32_t c = 8;
    uint32_t r = 8;

    auto* a = (double*) malloc(c*r*sizeof(double));
    for (uint32_t i = 0; i < c*r; ++i) {
        a[i] = i + 20;
    }

    auto* b = (double*) malloc(c*r*sizeof(double));
    for (uint32_t i = 0; i < c*r; ++i) {
        b[i] = i + 30;
    }

    auto* resSum = (double*) malloc(c*r*sizeof(double));

//    addMatrixSSE(a, b, c, r, resSum);
//    printf("Sum matrix:\n");
//    print(resSum, c, r);

    unsigned a1C = 2;
    unsigned a1R = 4;
    unsigned b1C = 2;
    unsigned b1R = 2;

    //double* a1 = (double*) malloc(a1C*a1R*sizeof(double));;
    double a1[] = {1,2,3,4,5,6,7,8};
    double b1[] = {1, 3, 2, 4};

    auto* resMul = (double*) malloc(a1R*b1C*sizeof(double));
    mulMatrixSSE((double*) a1, (double*) b1, a1C, a1R, b1C, resMul);
    printf("Mul matrix:\n");
    printColumnMatrix(resMul, a1R, b1C);

    return 0;
}
