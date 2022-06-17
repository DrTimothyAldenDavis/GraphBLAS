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
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
// test_AxB_phase1_factory: test phase1
//------------------------------------------------------------------------------

// Test generator code, to allow parameterized tests
// Uses jitFactory, dataFactory and GB_jit 
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    /********************
     * Launch kernel
     */
    problem_spec.set_sparsity_control(problem_spec.getA(), GxB_SPARSE, GxB_BY_ROW);
    problem_spec.set_sparsity_control(problem_spec.getB(), GxB_SPARSE, GxB_BY_ROW);

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
    EXPECT_EQ( bucket_count, problem_spec.getCnnz()); //check we sum to the right structural counts
//
    rmm_wrap_free(Nanobuckets);
    rmm_wrap_free(Blockbucket);

    std::cout << "end phase1 test ------------" << std::endl;

    fflush(stdout);
    return true;
}

//------------------------------------------------------------------------------
// test_AxB_phase2_factory: test phase2 and phase2end
//------------------------------------------------------------------------------

template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_AxB_phase2_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{

    int gpuID;
    cudaGetDevice( &gpuID);

    std::cout<< "found device "<<gpuID<<std::endl;

    auto mymxm = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mymxm);
    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

    problem_spec.set_sparsity_control(problem_spec.getA(), GxB_SPARSE, GxB_BY_ROW);
    problem_spec.set_sparsity_control(problem_spec.getB(), GxB_SPARSE, GxB_BY_ROW);

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
    fillvector_constant(NBUCKETS, bucketp, (int64_t)0);
    fillvector_constant(NBUCKETS, offset, (int64_t)0);
    fillvector_constant(problem_spec.getCnnz(), bucket, (int64_t)0);

    std::cout << "Running phase1 kernel" << std::endl;
    p1lF.jitGridBlockLaunch(nanobuckets, blockbucket,
                            problem_spec.getC(), problem_spec.getM(),
                            problem_spec.getA(), problem_spec.getB());


    CHECK_CUDA(cudaStreamSynchronize(0));

    std::cout << "Done." << std::endl;
    print_array<int64_t>(nanobuckets, NBUCKETS*nthrd*ntasks, "nanobuckets");
    print_array<int64_t>(blockbucket, NBUCKETS*ntasks, "blockbucket");

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

    EXPECT_EQ( bucketp[NBUCKETS], problem_spec.getCnnz()); //check we sum to the right structural counts

    rmm_wrap_free(nanobuckets);
    rmm_wrap_free(blockbucket);
    rmm_wrap_free(bucketp);
    rmm_wrap_free(bucket);
    rmm_wrap_free(offset);

//    problem_gen.revertCj();
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
bool test_AxB_dot3_full_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    // FIXME: Allow the adaptive tests in this guy
    std::cout << "phase 3 test ======================" << std::endl;

    bool result = false;

    int64_t N = problem_spec.getN();
    /**
     * Run Phase 1, phase 2 and phase2end: Compute nanobuckets and blockbuckets
     */

    auto mymxm = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mymxm);
    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

    problem_spec.set_sparsity_control(problem_spec.getA(), GxB_SPARSE, GxB_BY_ROW);
    problem_spec.set_sparsity_control(problem_spec.getB(), GxB_SPARSE, GxB_BY_ROW);

    const int64_t mnz = GB_nnz (problem_spec.getM()) ;
    const int64_t cnz = GB_nnz (problem_spec.getC()) ;
    const int64_t cvlen = problem_spec.getC()->vlen ;
    const int64_t cvdim = problem_spec.getC()->vdim ;
    const int64_t cnvec = problem_spec.getC()->nvec ;

    bool C_iso = false ;
    int C_sparsity = GB_sparsity (problem_spec.getM()) ;
    int M_sparsity = GB_sparsity (problem_spec.getM()) ;
    GrB_Type ctype = problem_spec.getBinaryOp()->ztype ;
    GrB_Matrix C_actual;

    std::cout << "Creating new bix: " << cnz << std::endl;
    GrB_Info info = GB_new_bix (&C_actual, // sparse or hyper (from M), existing header
                       ctype, cvlen, cvdim, GB_Ap_malloc, true,
                       M_sparsity, false, problem_spec.getM()->hyper_switch, cnvec,
                       cnz+1,  // add one to cnz for GB_cumsum of Cwork
                       true, C_iso, NULL) ;//Context) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        return (info) ;
    }

    std::cout << "Done creating new bix" << std::endl;

    int nthrd = p2lF.get_threads_per_block();
    int ntasks = p2elF.get_number_of_blocks(problem_spec.getM());

    // fabricate data as if it came from phase1:
    int64_t *nanobuckets = (int64_t*)rmm_wrap_malloc(NBUCKETS * nthrd * ntasks * sizeof (int64_t));
    int64_t *blockbucket = (int64_t*)rmm_wrap_malloc(NBUCKETS * ntasks * sizeof (int64_t));
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    int64_t *bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));
    int64_t *offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));

    std::cout << "nthrd: " << nthrd << ", ntasks: " << ntasks << std::endl;
    fillvector_constant(NBUCKETS, bucketp, (int64_t)0);
    fillvector_constant(NBUCKETS, offset, (int64_t)0);
    fillvector_constant(problem_spec.getCnnz(), bucket, (int64_t)0);

    std::cout << "Running phase1 kernel" << std::endl;
    p1lF.jitGridBlockLaunch(nanobuckets, blockbucket,
                            C_actual, problem_spec.getM(),
                            problem_spec.getA(), problem_spec.getB());


    CHECK_CUDA(cudaStreamSynchronize(0));

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

    std::cout << "Launching phase2end" << std::endl;


    // launch phase2end: note same # of tasks as phase1
    p2elF.jitGridBlockLaunch( nanobuckets, blockbucket,
                              bucketp, bucket, offset, C_actual,
                              problem_spec.getM());
    CHECK_CUDA(cudaStreamSynchronize(0));
    std::cout << "Done running phase2end" << std::endl;

    /**
     * Run Phase 3: Execute dot3 on all buckets
     */
    for (int b =0; b < NBUCKETS; ++b) {// loop on buckets
            int64_t b_start = bucketp[b];
            int64_t b_end = bucketp[b+1];
            int64_t nvecs = b_end - b_start;

            std::cout<< "bucket "<<b<<" has "<<nvecs<<" dots to do"<<std::endl;

           GpuTimer kernTimer;
           kernTimer.Start();

           fflush(stdout);

           GB_cuda_mxm_factory mysemiringfactory = problem_spec.get_mxm_factory();
           GB_cuda_mxm_phase3(mysemiringfactory, (GB_bucket_code )b,
                              b_start, b_end, bucketp, bucket, C_actual, problem_spec.getM(),
                              problem_spec.getB(), problem_spec.getA());
            CHECK_CUDA(cudaStreamSynchronize(0));

           kernTimer.Stop();


            fflush(stdout);

           std::cout<<"returned from kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
           GRB_TRY (GxB_Matrix_fprint (C_actual, "C GPU", GxB_SHORT_VERBOSE, stdout)) ;

            GrB_Matrix C_expected;
            GrB_Type type = cuda::jit::to_grb_type<T_C>();
            GRB_TRY (GrB_Matrix_new (&C_expected, type, N, N)) ;

            // ensure the GPU is not used
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_CONTROL, GxB_GPU_NEVER)) ;

            // Use GrB_DESC_S for structural because dot3 mask will never be complemented
            GRB_TRY (GrB_mxm(C_expected, problem_spec.getM(), NULL, problem_spec.get_semiring(), problem_spec.getA(),
                             problem_spec.getB(), problem_spec.get_mask_struct() ? GrB_DESC_ST1 : GrB_DESC_T1));
//            GRB_TRY (GrB_mxm(C_actual, M, NULL, mysemiring, A, B,
//                             Mask_struct ? GrB_DESC_S : NULL));

            GRB_TRY (GxB_Matrix_fprint (problem_spec.getM(), "M actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (problem_spec.getA(), "A actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (problem_spec.getB(), "B actual", GxB_SHORT_VERBOSE, stdout));

            GRB_TRY(GrB_Matrix_wait(C_actual, GrB_MATERIALIZE));
            GRB_TRY(GrB_Matrix_wait(C_expected, GrB_MATERIALIZE));

            GRB_TRY (GxB_Matrix_fprint (C_actual, "C GPU", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GxB_Matrix_fprint (C_expected, "C_actual", GxB_SHORT_VERBOSE, stdout));
            // compare
            double tol = 0 ;
            GrB_Index nvals1 = 0, nvals2 = 0 ;
            GRB_TRY (GrB_Matrix_nvals (&nvals1, C_actual)) ;
            GRB_TRY (GrB_Matrix_nvals (&nvals2, C_expected)) ;
            if (nvals1 != nvals2) { printf ("Aborting!!!\n") ; abort ( ) ; }
            GrB_Index nrows, ncols ;
            GrB_Matrix_nrows (&nrows, C_actual) ;
            GrB_Matrix_ncols (&ncols, C_actual) ;

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
                C_actual, Diff, NULL)) ;
            GRB_TRY (GxB_Matrix_fprint (Diff, "Diff actual", GxB_SHORT_VERBOSE, stdout));
            GRB_TRY (GrB_Matrix_free (&Diff)) ;

            if (tol == 0)
            {
                // check for perfect equality
                GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, C_actual, C_expected,
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

    rmm_wrap_free(bucket);
    rmm_wrap_free(bucketp);

    std::cout << "phase 3 test complete ======================" << std::endl;
    return result;
}

template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_reduce_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    // TODO: This test doesn't really fit the `mxm` category
    GrB_Monoid monoid = problem_spec.getMonoid();
    int64_t N = problem_spec.getN();
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

