//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/GB_cuda_builder.cpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "builder/GB_cuda_builder.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    GB_Matrix_free (Thandle) ;                  \
    GB_cuda_stream_pool_release (&stream) ;     \
}

GrB_Info GB_cuda_builder            // build a matrix from tuples
(
    GrB_Matrix *Thandle,            // matrix to build, dynamic header
    const GrB_Type ttype,           // type of output matrix T
    const int64_t vlen,             // length of each vector of T
    const int64_t vdim,             // number of vectors in T
    const bool is_csc,              // true if T is CSC, false if CSR
    const bool is_matrix,           // true if T a GrB_Matrix, false if vector
    const GB_void *restrict I,      // original indices, size nvals
    const GB_void *restrict J,      // original indices, size nvals
    const GB_void *restrict X,      // array of values of tuples, size nvals,
                                    // or size 1 if X is iso
    const bool X_iso,               // true if X is iso
    const int64_t nvals,            // number of tuples
    GrB_BinaryOp dup,               // binary function to assemble duplicates,
                                    // if NULL use the SECOND operator to
                                    // keep the most recent duplicate.
    const GrB_Type xtype,           // the type of X
    bool do_burble,                 // if true, then burble is allowed
    bool I_is_32,       // true if I is 32 bit, false if 64
    bool J_is_32,       // true if J is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32       // true if T->i is built as 32 bit, false if 64
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    ASSERT (Thandle != NULL) ;
    ASSERT (I != NULL) ;
    ASSERT (X != NULL) ;
    ASSERT (ttype != NULL) ;
    ASSERT (xtype != NULL) ;

    //--------------------------------------------------------------------------
    // construct the SECOND operator if dup is NULL
    //--------------------------------------------------------------------------

    struct GB_BinaryOp_opaque dup_header ;
    if (dup == NULL)
    { 
        // z = SECOND (x,y) where all three types are the same as ttype
        // T(i,j) = (ttype) Sx(k) will be done for all tuples.  If dup is
        // SECOND_UDT, dup->binop_function will be NULL; this is OK.
        dup = GB_binop_second (ttype, &dup_header) ;
        ASSERT (dup != NULL && GB_op_is_second (dup, ttype)) ;
    }

    //--------------------------------------------------------------------------
    // get CUDA stream and geometry
    //--------------------------------------------------------------------------

    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    // determine the geometry of the CUDA kernel launches
    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (nvals, GB_CUDA_BUILDER_CHUNKSIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    //--------------------------------------------------------------------------
    // build T from the (I,J,X) tuples
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_builder_jit (Thandle, ttype, vlen, vdim, is_csc, is_matrix,
        I, J, X, X_iso, nvals, dup, xtype, I_is_32, J_is_32,
        Tp_is_32, Tj_is_32, Ti_is_32, stream, gridsz)) ;

    //--------------------------------------------------------------------------
    // release the stream
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;

    //--------------------------------------------------------------------------
    // handle the iso case
    //--------------------------------------------------------------------------

    if (X_iso)
    {
        // copy the iso-value entry of X [0] into T->x [0]
        ASSERT (xtype == ttype) ;
        GrB_Matrix T = (*Thandle) ;
        memcpy (T->x, X, xtype->size) ;
    }

    // return the result
    return (GrB_SUCCESS) ;
}

