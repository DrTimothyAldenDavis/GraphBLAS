// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
//#include "GB_binary_search.h"
#include "GpuTimer.h"
#include "GB_cuda_buckets.h"
#include "../../rmm_wrap/rmm_wrap.h"
#include <gtest/gtest.h>
#include "test_data.hpp"
#include "problem_spec.hpp"

extern "C" {
    #include "GB.h"
}

#include "../jitFactory.hpp"
#include "dataFactory.hpp"

////Operations for test results on CPU
//template<typename T> T myOP_plus( T a, T b) { return  a + b;}
//template<typename T> T myOP_min ( T a, T b) { return  a < b ? a : b;}
//template<typename T> T myOP_max ( T a, T b) { return  a > b ? a : b;}
//template<typename T> T myOP_first ( T a, T b) { return  a ;}
//template<typename T> T myOP_second ( T a, T b) { return  b ;}
//template<typename T> T myOP_times ( T a, T b) { return  a * b ;}
//
//template<typename T> T (*myOpPTR)(T a, T b);
//template<typename T> T (*ADD_ptr)(T a, T b);
//template<typename T> T (*MUL_ptr)(T a, T b);

//AxB_dot3_phase1 kernels
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory( int64_t , int64_t , int64_t , int64_t ) ;

//AxB_dot3_phase2 kernels
template <typename T_C>
bool test_AxB_dot3_phase2_factory( int , int64_t , int64_t , int64_t, int64_t ) ;

//Fixture to generate valid inputs and hold them for tests
class AxB_dot3_Test : public ::testing::Test
{
   void SetUp() {}

   void TearDown() {}
};

template<typename T, typename I>
void print_array(void *arr, I size, const char *name) {
    std::cout << "Printing " << name << std::endl;
    for(I i = 0; i < size; ++i) {
        std::cout << static_cast<T*>(arr)[i] << ", ";
    }
    std::cout << std::endl << "Done." << std::endl;
}

//------------------------------------------------------------------------------
// test_AxB_phase1_factory: test phase1
//------------------------------------------------------------------------------

// Test generator code, to allow parameterized tests
// Uses jitFactory, dataFactory and GB_jit 
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz, GrB_Monoid monoid, GrB_BinaryOp binop,
                              mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    /********************
     * Launch kernel
     */

    GB_cuda_mxm_factory mysemiringfactory = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mysemiringfactory);

    GpuTimer kernTimer;
    kernTimer.Start();

    int nthrd = p1lF.get_threads_per_block();
    int ntasks = p1lF.get_number_of_blocks(problem_spec.getM());

    // TODO: Verify that RMM is checking and throwing exceptions
    int nanobuckets_size = NBUCKETS * nthrd * ntasks;
    int blockbuckets_size = NBUCKETS * ntasks;

    printf("nanobuckets_size: %d\n", nanobuckets_size);
    printf("blockbuckets_size: %d\n", blockbuckets_size);

    int64_t *Nanobuckets = (int64_t*)rmm_wrap_malloc(nanobuckets_size * sizeof (int64_t));
    int64_t *Blockbucket = (int64_t*)rmm_wrap_malloc(blockbuckets_size * sizeof (int64_t));
//
//    std::cout << "INvoking grid block launch for phase1" << std::endl;
    p1lF.jitGridBlockLaunch(Nanobuckets, Blockbucket,
                            problem_spec.getC(), problem_spec.getM(),
                            problem_spec.getA(), problem_spec.getB());

    CHECK_CUDA(cudaStreamSynchronize(0));
    kernTimer.Stop();
    std::cout<<"returned from phase1 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
    print_array<int64_t>(Nanobuckets, nanobuckets_size, "Nanobuckets");
    print_array<int64_t>(Blockbucket, blockbuckets_size, "Blockbucket");
    std::cout<<"==== phase1 done=============================" <<std::endl;

    int64_t bucket_count = 0;
    for (int i =0; i< NBUCKETS*ntasks; ++i) bucket_count += Blockbucket[i];
    EXPECT_EQ( bucket_count, problem_spec.getCnz()); //check we sum to the right structural counts
//
    rmm_wrap_free(Nanobuckets);
    rmm_wrap_free(Blockbucket);

//
    return true;
}

//------------------------------------------------------------------------------
// test_AxB_phase2_factory: test phase2 and phase2end
//------------------------------------------------------------------------------

template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_AxB_phase2_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz,
                              mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

//    SpGEMM_problem_generator<T_C, T_C, T_C, T_C> G(N, N);
//    int64_t Annz = N*N;
//    int64_t Bnnz = N*N;
//    int64_t Cnz = N;
//    float Cnzpercent = (float) Cnz/(N*N);
//
//    G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW);
//    G.init_B(Bnnz, GxB_FULL, GxB_BY_ROW);
//    G.init_C(Cnzpercent);
//    G.fill_buckets( TB ); // all elements go to testbucket= TB
//    G.loadCj(); // FIXME: Figure out why this is needed here
//
//
//    GrB_Matrix C = G.getC();
//    GrB_Matrix M = G.getM();       // note: values are not accessed

    auto problem_gen = problem_spec.getG();

    problem_gen.loadCj(); // FIXME: Figure out why this is needed here

   GpuTimer kernTimer;
   kernTimer.Start();
   const int64_t mnz = GB_nnz (problem_spec.getM()) ;

   int nthrd = p2lF.get_threads_per_block();
   int ntasks = p2elF.get_number_of_blocks(problem_spec.getM());

    // fabricate data as if it came from phase1:
    int64_t *nanobuckets = (int64_t*)rmm_wrap_malloc(NBUCKETS * nthrd * ntasks * sizeof (int64_t));
    int64_t *blockbucket = (int64_t*)rmm_wrap_malloc(NBUCKETS * ntasks * sizeof (int64_t));
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    int64_t *bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));
    int64_t *offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));

    std::cout << "nthrd: " << nthrd << ", ntasks: " << ntasks << std::endl;
    fillvector_constant(NBUCKETS * nthrd * ntasks, nanobuckets, (int64_t)0);
    fillvector_constant(problem_spec.getCnz(), nanobuckets, (int64_t)1);
    fillvector_constant(NBUCKETS * ntasks, blockbucket, (int64_t)0);
    blockbucket[3] = problem_spec.getCnz();
    fillvector_constant(NBUCKETS, bucketp, (int64_t)0);
    fillvector_constant(NBUCKETS, offset, (int64_t)0);
    fillvector_constant(problem_spec.getCnz(), bucket, (int64_t)0);

    print_array<int64_t>(nanobuckets, NBUCKETS*nthrd*ntasks, "nanobuckets");
    print_array<int64_t>(blockbucket, NBUCKETS*ntasks, "blockbucket");
//
//    // launch phase2 (just with p2ntasks as the # of tasks)
    p2lF.jitGridBlockLaunch(blockbucket, offset, problem_spec.getM());
    CHECK_CUDA(cudaStreamSynchronize(0));
//
//    // do the reduction between phase2 and phase2end
    int64_t s= 0;
    for ( int bucket = 0 ; bucket < NBUCKETS+1; ++bucket)
    {
        bucketp[bucket] = s;
        s+= offset[bucket];
        printf("bucketp[%d] = %ld, offset= %ld\n", bucket, bucketp[bucket],offset[bucket]);
    }

    // launch phase2end: note same # of tasks as phase1
    p2elF.jitGridBlockLaunch( nanobuckets, blockbucket,
                              bucketp, bucket, offset, problem_spec.getC(),
                              problem_spec.getM());
    CHECK_CUDA(cudaStreamSynchronize(0));
//    kernTimer.Stop();
//    std::cout<<"returned from phase2 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//
    print_array<int64_t>(bucketp, NBUCKETS, "bucketp");
    print_array<int64_t>(bucket, mnz, "bucket");
    std::cout<<"phase2 kernel done =================="<<std::endl;

    EXPECT_EQ( bucketp[NBUCKETS], problem_spec.getCnz()); //check we sum to the right structural counts

    rmm_wrap_free(nanobuckets);
    rmm_wrap_free(blockbucket);
    rmm_wrap_free(bucketp);
    rmm_wrap_free(bucket);
    rmm_wrap_free(offset);
   return true;
}

template<typename T>
void make_grb_matrix(GrB_Matrix mat, int64_t n_rows, int64_t n_cols,
                     std::vector<int64_t> &indptr,
                     std::vector<int64_t> &indices, std::vector<T> &data,
                     int gxb_sparsity_control = GxB_SPARSE,
                     int gxb_format = GxB_BY_ROW) {

    GrB_Type type = cuda::jit::to_grb_type<T>();

    GRB_TRY (GrB_Matrix_new (&mat, type, n_rows, n_cols)) ;

    for(int64_t row = 0; row < n_rows; ++row) {
        int64_t start = indptr[row];
        int64_t stop = indptr[row+1];

        for(int64_t offset = start; offset < stop; ++offset) {
            GrB_Index i = (GrB_Index) row;
            GrB_Index j = (GrB_Index) indices[offset];
            T x = data[offset];

            cuda::jit::set_element<T> (mat, x, i, j) ;
        }
    }

    GRB_TRY (GrB_Matrix_wait (mat, GrB_MATERIALIZE)) ;
    GRB_TRY (GB_convert_any_to_non_iso (mat, true, NULL)) ;
    // TODO: Need to specify these
    GRB_TRY (GxB_Matrix_Option_set (mat, GxB_SPARSITY_CONTROL, gxb_sparsity_control)) ;
    GRB_TRY (GxB_Matrix_Option_set(mat, GxB_FORMAT, gxb_format));
    GRB_TRY (GxB_Matrix_fprint (mat, "my mat", GxB_SHORT_VERBOSE, stdout)) ;

    bool iso ;
    GRB_TRY (GxB_Matrix_iso (&iso, mat)) ;
    if (iso)
    {
        printf ("Die! (cannot do iso)\n") ;
        GrB_Matrix_free (&mat) ;
    }

}

template <
    typename T_C, typename T_M, typename T_A,typename T_B,
    typename T_X, typename T_Y, typename T_Z>
bool test_AxB_dot3_full_factory( int TB, int64_t N, int64_t Anz, int64_t Bnz,
                                 GrB_Monoid monoid, GrB_BinaryOp binop,
                                 mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    // FIXME: Allow the adaptive tests in this guy

    //Generate test data and setup for using a jitify kernel with 'bucket' interface
    // The testBucket arg tells the generator which bucket we want to exercise
//    int64_t Annz;
//    int64_t Bnnz;
//
//    switch(TB) {
////      case GB_BUCKET_DNDN:
////          Annz = N * N;
////          Bnnz = N * N;
////          break;
////      case GB_BUCKET_SPDN:
////          Annz = N * N;
////          Bnnz = N * 5;
////          break;
//        case GB_BUCKET_VSSP:
//            Annz = N * 2;
//            Bnnz = N * 10;
//            break;
//        case GB_BUCKET_VSVS:
////      case GB_BUCKET_VSVS_4:
////      case GB_BUCKET_VSVS_16:
////      case GB_BUCKET_VSVS_64:
////      case GB_BUCKET_VSVS_256:
//            Annz = N * 2;
//            Bnnz = N * 4;
//            break;
//        case GB_BUCKET_MERGEPATH:
//            Annz = N * 5;
//            Bnnz = N * 2;
//            break;
//        default:
//            printf("Bucket not yet being tested!\n");
//            exit(1);
//    }
//    int64_t Cnz = N;
//    float Cnzpercent = (float) Cnz/(N*N);
//
//    // FIXME: make this an argument
//    bool Mask_struct = true;
//
//    std::cout << "Getting test data" << std::endl;
//    // FIXME: These need to be set based on the bucket being tested
////    TestData<T_A, T_B, T_C, T_M> data = *make_karate_tricount<T_A, T_B, T_C, T_M>();
//
//    std::cout << "Creating problem gen" << std::endl;
////    N = data.A_indptr.size()-1;
//    SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G(N, N);
//    G.init_C(float(Cnz) / (N * N));
//
////    GrB_Matrix A;
////    GrB_Matrix B;
////    GrB_Matrix C;
////    GrB_Matrix M;
////
////    GrB_Matrix C_actual = G.getC();
//
////    make_grb_matrix<T_A>(A, data.A_indptr, data.A_indices, data.A_data, GxB_SPARSE);
////    make_grb_matrix<T_B>(B, data.B_indptr, data.B_indices, data.B_data, GxB_FULL, GxB_BY_ROW);
////    make_grb_matrix<T_C>(C, data.C_indptr, data.C_indices, data.C_data);
////    make_grb_matrix<T_M>(M, data.M_indptr, data.M_indices, data.M_data);
//
//
////    std::cout << "Filling A" << std::endl;
//    G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW, 543210, 0, 2);
////    std::cout << "Filling B" << std::endl;
//
//    G.init_B(Bnnz, GxB_SPARSE, GxB_BY_ROW, 32, 0, 2);
//
//    /**
//     * For testing, we need to create our output C and configure
//     * it w/ the necessary sparsity.
//     */
//    G.fill_buckets( TB); // all elements go to testbucket= TB
//
//    GrB_Matrix C = G.getC();
//    GrB_Matrix M = G.getM();
//    GrB_Matrix A = G.getA();
//    GrB_Matrix B = G.getB();
//
////    GRB_TRY (GxB_Matrix_fprint (A, "A", GxB_SHORT_VERBOSE, stdout)) ;
////    GRB_TRY (GxB_Matrix_fprint (B, "B", GxB_SHORT_VERBOSE, stdout)) ;
////    GRB_TRY (GxB_Matrix_fprint (M, "M", GxB_SHORT_VERBOSE, stdout)) ;
////    GRB_TRY (GxB_Matrix_fprint (C, "C", GxB_SHORT_VERBOSE, stdout)) ;
////
//    std::cout << "Building mxm factgory" << std::endl;
//    GB_cuda_mxm_factory mymxmfactory = GB_cuda_mxm_factory ( ) ;
//    GrB_Semiring mysemiring;
//    auto grb_info = GrB_Semiring_new(&mysemiring, monoid, binop);
//    GRB_TRY (grb_info) ;
//
//    bool flipxy = false;
//    bool mask_struct = false;
//    bool mask_comp = false;
////    GrB_Matrix C_actual = C;
//
//    bool C_iso = false ;
//    int C_sparsity = GB_sparsity (M) ;
//    GrB_Type ctype = binop->ztype ;
//
//    mymxmfactory.mxm_factory (
//        C_iso, C_sparsity, ctype,
//        M, mask_struct, mask_comp,
//        mysemiring, flipxy,
//        A, B) ;

    bool result = false;

    /**
     * Run Phase 1: Compute nanobuckets and blockbuckets
     */
    const int64_t mnz = GB_nnz (problem_spec.getM()) ;

    int chunk_size = 128;

    int number_of_sms = GB_Global_gpu_sm_get (0);
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));

    CHECK_CUDA(cudaMemset(bucketp, 0, (NBUCKETS+1)*sizeof(int64_t)));

    int64_t *bucket = (int64_t*)rmm_wrap_malloc(problem_spec.getCnz() * sizeof (int64_t));

    /**
     * Run Phase 3: Execute dot3 on all buckets
     */
    for (int b =0; b < NBUCKETS; ++b) {// loop on buckets
        if (b == TB) {
            problem_spec.getG().fill_buckets(b);
            int64_t *Bucket = problem_spec.getG().getBucket();
            int64_t *BucketStart = problem_spec.getG().getBucketStart();

            int64_t b_start = BucketStart [b] ;
            int64_t b_end   = BucketStart [b+1] ;
            int64_t nvecs = b_end - b_start ;

            if (nvecs > 0) std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;

            problem_spec.getG().loadCj();

           GpuTimer kernTimer;
           kernTimer.Start();

            GB_cuda_mxm_factory mysemiringfactory = problem_spec.get_mxm_factory();
           GB_cuda_mxm_phase3(mysemiringfactory, (GB_bucket_code )b,
                              b_start, b_end, bucketp, Bucket, problem_spec.getC(), problem_spec.getM(),
                              problem_spec.getB(), problem_spec.getA());
            CHECK_CUDA(cudaStreamSynchronize(0));

            print_array<int64_t>(bucketp, NBUCKETS+1, "bucketp");

           kernTimer.Stop();

           std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
           GRB_TRY (GxB_Matrix_fprint (problem_spec.getC(), "C GPU", GxB_SHORT_VERBOSE, stdout)) ;

            GrB_Matrix C_actual;
            GrB_Type type = cuda::jit::to_grb_type<T_C>();
            GRB_TRY (GrB_Matrix_new (&C_actual, type, N, N)) ;

            // ensure the GPU is not used
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_NEVER)) ;

            // Use GrB_DESC_S for structural because dot3 mask will never be complemented
            GRB_TRY (GrB_mxm(C_actual, problem_spec.getM(), NULL, problem_spec.get_semiring(), problem_spec.getA(),
                             problem_spec.getB(),
                problem_spec.get_mask_struct() ? GrB_DESC_ST1 : GrB_DESC_T1));
//            GRB_TRY (GrB_mxm(C_actual, M, NULL, mysemiring, A, B,
//                             Mask_struct ? GrB_DESC_S : NULL));

            GRB_TRY (GxB_Matrix_fprint (problem_spec.getM(), "M actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (problem_spec.getA(), "A actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (problem_spec.getB(), "B actual", GxB_SHORT_VERBOSE, stdout));

            GRB_TRY(GrB_Matrix_wait(problem_spec.getC(), GrB_MATERIALIZE));
            GRB_TRY(GrB_Matrix_wait(C_actual, GrB_MATERIALIZE));

            GRB_TRY (GxB_Matrix_fprint (problem_spec.getC(), "C GPU", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (C_actual, "C_actual", GxB_SHORT_VERBOSE, stdout));
            // compare
            double tol = 0 ;
            GrB_Index nvals1 = 0, nvals2 = 0 ;
            GRB_TRY (GrB_Matrix_nvals (&nvals1, problem_spec.getC())) ;
            GRB_TRY (GrB_Matrix_nvals (&nvals2, C_actual)) ;
            if (nvals1 != nvals2) { printf ("Aborting!!!\n") ; abort ( ) ; }
            GrB_Index nrows, ncols ;
            GrB_Matrix_nrows (&nrows, problem_spec.getC()) ;
            GrB_Matrix_ncols (&ncols, problem_spec.getC()) ;

            GrB_Matrix T;

            GRB_TRY (GrB_Matrix_new (&T, GrB_BOOL, nrows, ncols)) ;
            GrB_BinaryOp op = NULL;
            GrB_UnaryOp op_abs = NULL ;
            if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
            else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
            else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
            else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
            else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
            else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
            else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
            else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
            else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
            else if (type == GrB_FP32  )
            {
                op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
                op_abs = GrB_ABS_FP32 ;
            }
            else if (type == GrB_FP64  )
            {
                op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
                op_abs = GrB_ABS_FP64 ;
            }
            else if (type == GxB_FC32  )
            {
                op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
                op_abs = GxB_ABS_FC32 ;
            }
            else if (type == GxB_FC64  )
            {
                op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
                op_abs = GxB_ABS_FC64 ;
            }


            // Diff = C - C_actual
            GrB_Matrix Diff ;
            GRB_TRY (GrB_Matrix_new (&Diff, GrB_FP64, nrows, ncols)) ;
            GRB_TRY (GrB_Matrix_apply (Diff, NULL, NULL, GrB_AINV_FP64, C_actual, NULL)) ;
            GRB_TRY (GrB_Matrix_eWiseAdd_BinaryOp (Diff, NULL, NULL, GrB_PLUS_FP64,
                problem_spec.getC(), Diff, NULL)) ;
            GRB_TRY (GxB_Matrix_fprint (Diff, "Diff actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GrB_Matrix_free (&Diff)) ;

            if (tol == 0)
            {
                // check for perfect equality
                GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, problem_spec.getC(), C_actual,
                    NULL)) ;
                GrB_Index nvals3 = 1 ;
                GRB_TRY (GxB_Matrix_fprint (T, "T actual", GxB_SHORT_VERBOSE, stdout));
                GRB_TRY (GrB_Matrix_nvals (&nvals3, T)) ;
                if (nvals1 != nvals3) { printf ("!!\n") ; abort ( ) ; } 
                bool is_same = false ;
                GRB_TRY (GrB_Matrix_reduce_BOOL (&is_same, NULL, GrB_LAND_MONOID_BOOL,
                    T, NULL)) ;
                if (!is_same) { printf ("!!\n") ; abort ( ) ; } 
                GRB_TRY (GrB_Matrix_free (&T)) ;
            }
            else
            {
                // TODO: check with roundoff
                { printf ("!!\n") ; abort ( ) ; } 
            }

            // re-enable the GPU
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_ALWAYS)) ;
         }
        }

    rmm_wrap_free(bucket);
    rmm_wrap_free(bucketp);

    return result;
}

template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_reduce_factory(unsigned int N, GrB_Monoid monoid, mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    //std::cout<<" alloc'ing data and output"<<std::endl;
    std::vector<int64_t> indptr(N+1);
    std::vector<int64_t> index(N);
    std::vector<T_C> d_data(N);

    indptr[N] = N;
    fillvector_linear<int64_t>((int)N, indptr.data(), (int64_t)0);
    fillvector_constant<int64_t>((int)N, index.data(), (int64_t)1);
    fillvector_linear<T_C> ( N, d_data.data());

    GrB_Type t = cuda::jit::to_grb_type<T_C>();

    GrB_Matrix A;
    make_grb_matrix(problem_spec.getA(), N, N, indptr, index, d_data, GxB_SPARSE, GxB_BY_ROW);
    CHECK_CUDA(cudaStreamSynchronize(0));

    GRB_TRY (GrB_Matrix_wait (problem_spec.getA(), GrB_MATERIALIZE)) ;
    GRB_TRY (GxB_Matrix_fprint (problem_spec.getA(), "A", GxB_SHORT_VERBOSE, stdout));

    T_C actual;
    GB_cuda_reduce( problem_spec.getA(), &actual, monoid );

    GrB_Vector v;
    GrB_Vector_new(&v, t, N);

    // Just sum in place for now (since we are assuming sum)
    int sum = 0;
    for(int i = 0; i < N; ++i) {
        sum+= d_data[i];
        cuda::jit::vector_set_element<T_C>(v, i, d_data[i]);
    }
    printf("Sum: %d\n", sum);

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_NEVER)) ;

    printf("Invoking grb reduce\n");
    T_C expected;
    GRB_TRY(cuda::jit::vector_reduce(&expected, v, monoid));
    CHECK_CUDA(cudaStreamSynchronize(0));
    printf("Done.\n");

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_ALWAYS)) ;
    if(expected != actual) {
        std::cout << "results do not match: reduced=" << expected << ", actual=" << actual << std::endl;
        exit(1);
    } else {
        std::cout << "Results matched!" << std::endl;
    }

    return expected == actual;
}

