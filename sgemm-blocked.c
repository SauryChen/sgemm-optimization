/* AddDot1x4 + AddDot4x4 + intrinsics + pack A */

#include "arm_neon.h"
#include <stdio.h>

const char* sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 72 // defined as the multiple of 4 for AddDot4x4
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* Create macros so that the matrices are stored in column-major order */
# define A(i,j) A[(i) + lda*(j)]
# define B(i,j) B[(i) + lda*(j)]
# define C(i,j) C[(i) + lda*(j)]

/* Create macro so that X(i) equals to the ith element of X, where X is a lda*lda matrix */
# define X(i) X[i*incx]

void AddDot(int, float *, int, float *, float *);
void AddDot1x4(int, float *, int, float *, float *); //otherwise, use 1x4

/* Set the BLOCK_SIZE specific at 72, only use 4x4 if K = BLOCK */
void AddDot4x4(int, float *, int, float *, float *);

void AddDot(int K, float *X, int incx, float *Y, float * sum){
  int k;
  for (k = 0; k < K; ++k){
    * sum += X(k) * Y[k];
  }
}

void PackMatrixA(int K, float* A, int lda, float *a_to){
  for (int j = 0; j < K; ++j){
    float * a_ij_pntr = &A(0,j);
    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);
  }
}

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, float* A, float* B, float* C){
  if(((M % 4) == 0) && ((N % 4) == 0) && ((K % 4) == 0)){

    float packedA[M*K];
    for (int i = 0; i < M; i+=4){
      for (int j = 0; j < N; j+=4){
        if(j == 0){
          PackMatrixA(K, &A(i,0), lda, &packedA[i*K]);
        }
        AddDot4x4(K, &packedA[i*K], lda, &B(0,j), &C(i,j));
      }
    }
  }
  else{
    int jvalue = 0;
    for (int i = 0; i < M; ++i){
      for (int j = 0; j < (N-4); j+=4){
        AddDot1x4(K, &A(i,0), lda, &B(0,j), &C(i,j));
        jvalue = j + 4;
      }
      for (int j = jvalue; j < N; ++j){
        AddDot(K, &A(i,0), lda, &B(0,j), &C(i,j));
      }
    }
  }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_sgemm (int lda, float* A, float* B, float* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE){
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE){
      /* Accumulate block sgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE){
         /* Correct block dimensions if block "goes off edge of" the matrix */
         int M = min (BLOCK_SIZE, lda-i);
         int N = min (BLOCK_SIZE, lda-j);
         int K = min (BLOCK_SIZE, lda-k);
         /* Perform individual block sgemm */
         do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    } 
  }  
}

void AddDot1x4(int K, float *A, int lda, float *B, float *C){
  int p;
  /* C(0,0), C(0,1), C(0,2), C(0,3) and A(0,p)*/
  register float c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;
  c_00_reg = 0.0;
  c_01_reg = 0.0;
  c_02_reg = 0.0;
  c_03_reg = 0.0;


  float *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
  bp0_pntr = &B(0,0);
  bp1_pntr = &B(0,1);
  bp2_pntr = &B(0,2);
  bp3_pntr = &B(0,3);

  for(p = 0; p < K; ++p){
    a_0p_reg = A(0,p);
    c_00_reg += a_0p_reg * *bp0_pntr++;
    c_01_reg += a_0p_reg * *bp1_pntr++;
    c_02_reg += a_0p_reg * *bp2_pntr++;
    c_03_reg += a_0p_reg * *bp3_pntr++;
  }

  C(0,0) += c_00_reg;
  C(0,1) += c_01_reg;
  C(0,2) += c_02_reg;
  C(0,3) += c_03_reg;
}

void AddDot4x4(int K, float *A, int lda, float *B, float *C){
  /* only use AddDot4x4 if K = M = N = BLOCK_SIZE (72) */
  int p;
  
  float32x4_t c_00_30, c_01_31, c_02_32, c_03_33;
  float32x4_t a_0p_3p;

  float32x4_t zero = {0.0, 0.0, 0.0, 0.0};
  c_00_30 = zero; c_01_31 = zero; c_02_32 = zero; c_03_33 = zero;
  
  for(p = 0; p < K; ++p){
    /* both length and width should be multiple of 4 */
    a_0p_3p = vld1q_f32(A); // read a column of A
    A += 4;

    
    float32x4_t b_p0, b_p1, b_p2, b_p3;
    b_p0 = vdupq_n_f32(B(p,0)); // b_x0 = {B(p,0) B(p,0) B(p,0) B(p,0)}
    b_p1 = vdupq_n_f32(B(p,1));
    b_p2 = vdupq_n_f32(B(p,2));
    b_p3 = vdupq_n_f32(B(p,3));
    

    c_00_30 += vmulq_f32(a_0p_3p, b_p0); // c[i] += a[i] * b[i]
    c_01_31 += vmulq_f32(a_0p_3p, b_p1);
    c_02_32 += vmulq_f32(a_0p_3p, b_p2);
    c_03_33 += vmulq_f32(a_0p_3p, b_p3);
  }

  C(0,0) += c_00_30[0]; C(1,0) += c_00_30[1]; C(2,0) += c_00_30[2]; C(3,0) += c_00_30[3];
  C(0,1) += c_01_31[0]; C(1,1) += c_01_31[1]; C(2,1) += c_01_31[2]; C(3,1) += c_01_31[3];
  C(0,2) += c_02_32[0]; C(1,2) += c_02_32[1]; C(2,2) += c_02_32[2]; C(3,2) += c_02_32[3];
  C(0,3) += c_03_33[0]; C(1,3) += c_03_33[1]; C(2,3) += c_03_33[2]; C(3,3) += c_03_33[3];
}