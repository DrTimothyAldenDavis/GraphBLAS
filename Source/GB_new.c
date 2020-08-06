//------------------------------------------------------------------------------
// GB_new: create a new GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Creates a new matrix but does not allocate space for A->b, A->i, and A->x.
// See GB_create instead.

// If the Ap_option is GB_Ap_calloc, the A->p and A->h are allocated and
// initialized, and A->magic is set to GB_MAGIC to denote a valid matrix.
// Otherwise, the matrix has not yet been completelyinitialized, and A->magic
// is set to GB_MAGIC2 to denote this.  This case only occurs internally in
// GraphBLAS.  The internal function that calls GB_new must then allocate or
// initialize A->p itself, and then set A->magic = GB_MAGIC when it does so.

// To allocate a full or bitmap matrix, sparsity_structure is GB_FULL (which is
// 2) or GB_BITMAP (which is 3).  The Ap_option is ignored.  For a full or
// bitmap matrix, only the header is allocated, if NULL on input.

// Only GrB_SUCCESS and GrB_OUT_OF_MEMORY can be returned by this function.

// The GrB_Matrix object holds both a sparse vector and a sparse matrix.  A
// vector is represented as an vlen-by-1 matrix, but it is sometimes treated
// differently in various methods.  Vectors are never transposed via a
// descriptor, for example.

// The matrix may be created in an existing header, which case *Ahandle is
// non-NULL on input.  If an out-of-memory condition occurs, (*Ahandle) is
// returned as NULL, and the existing header is freed as well, if non-NULL on
// input.

// To see where these options are used in SuiteSparse:GraphBLAS:
// grep "allocate a new header"
// which shows all uses of GB_new and GB_create

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_new                 // create matrix, except for indices & values
(
    GrB_Matrix *Ahandle,        // handle of matrix to create
    const GrB_Type type,        // matrix type
    const int64_t vlen,         // length of each vector
    const int64_t vdim,         // number of vectors
    const GB_Ap_code Ap_option, // allocate A->p and A->h, or leave NULL
    const bool is_csc,          // true if CSC, false if CSR
    const int sparsity_structure,   // 1:hyper, 0:nonhyper, -1:auto,
                                    // 2:full, or 3:bitmap
    const float hyper_switch,   // A->hyper_switch, unless auto
    const int64_t plen,         // size of A->p and A->h, if A hypersparse.
                                // Ignored if A is not hypersparse.
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ahandle != NULL) ;
    ASSERT_TYPE_OK (type, "type for GB_new", GB0) ;
    ASSERT (vlen >= 0 && vlen <= GxB_INDEX_MAX)
    ASSERT (vdim >= 0 && vdim <= GxB_INDEX_MAX) ;

    //--------------------------------------------------------------------------
    // allocate the matrix header, if not already allocated on input
    //--------------------------------------------------------------------------

    bool allocated_header = false ;
    if ((*Ahandle) == NULL)
    {
        (*Ahandle) = GB_CALLOC (1, struct GB_Matrix_opaque) ;
        if (*Ahandle == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        allocated_header = true ;
    }

    GrB_Matrix A = *Ahandle ;

    //--------------------------------------------------------------------------
    // initialize the matrix header
    //--------------------------------------------------------------------------

    // basic information
    A->magic = GB_MAGIC2 ;                 // object is not yet valid
    A->type = type ;

    // CSR/CSC format
    A->is_csc = is_csc ;

    // hypersparsity
    bool A_is_hyper ;
    bool A_is_full_or_bitmap = false ;
    A->hyper_switch = hyper_switch ;
    if (sparsity_structure == GB_FORCE_HYPER)
    { 
        A_is_hyper = true ;             // force A to be hypersparse
    }
    else if (sparsity_structure == GB_FORCE_NONHYPER)
    { 
        A_is_hyper = false ;            // force A to be sparse
    }
    else if (sparsity_structure == GB_FULL || sparsity_structure == GB_BITMAP)
    { 
        A_is_full_or_bitmap = true ;    // force A to be full or bitmap
        A_is_hyper = false ;
    }
    else // GB_AUTO_HYPER
    { 
        // auto selection:  non-hypersparse if one vector or less, or
        // if the global hyper_switch is negative.  This is only used by
        // GrB_Matrix_new, and in a special case in GB_mask.
        // Never select A to be full, since A is created with no entries.
        ASSERT (sparsity_structure == GB_AUTO_HYPER) ;
        float hyper_switch = GB_Global_hyper_switch_get ( ) ;
        A->hyper_switch = hyper_switch ;
        A_is_hyper = !(vdim <= 1 || 0 > hyper_switch) ;
    }

    // matrix dimensions
    A->vlen = vlen ;
    A->vdim = vdim ;

    // content that is freed or reset in GB_ph_free
    if (A_is_full_or_bitmap)
    {  
        // A is full or bitmap
        A->plen = -1 ;
        A->nvec = vdim ;
        // all vectors present, unless matrix has a zero dimension 
        A->nvec_nonempty = (vlen > 0) ? vdim : 0 ;
    }
    else if (A_is_hyper)
    { 
        // A is hypersparse
        A->plen = GB_IMIN (plen, vdim) ;
        A->nvec = 0 ;           // no vectors present
        A->nvec_nonempty = 0 ;      // all vectors are empty
    }
    else
    { 
        // A is sparse
        A->plen = vdim ;
        A->nvec = vdim ;        // all vectors present in the data structure
                                // (but all are currently empty)
        A->nvec_nonempty = 0 ;      // all vectors are empty
    }

    A->p = NULL ;
    A->h = NULL ;
    A->p_shallow = false ;
    A->h_shallow = false ;
    A->mkl = NULL ;             // no analysis from MKL yet

    A->logger = NULL ;          // no error logged yet

    // content that is freed or reset in GB_bix_free
    A->b = NULL ;
    A->i = NULL ;
    A->x = NULL ;
    A->nzmax = 0 ;              // GB_NNZ(A) checks nzmax==0 before Ap[nvec]
    A->nvals = 0 ;              // for bitmapped matrices only
    A->b_shallow = false ;
    A->i_shallow = false ;
    A->x_shallow = false ;
    A->nzombies = 0 ;
    A->jumbled = false ;
    A->Pending = NULL ;

    //--------------------------------------------------------------------------
    // Allocate A->p and A->h if requested
    //--------------------------------------------------------------------------

    bool ok ;
    if (A_is_full_or_bitmap || Ap_option == GB_Ap_null)
    { 
        // A is not initialized yet; A->p and A->h are both NULL.
        // sparse case: GB_NNZ(A) must check A->nzmax == 0 since A->p is not
        // allocated.
        // full case: A->x not yet allocated.  A->nzmax still zero
        // bitmap case: A->b, A->x not yet allocated.  A->nzmax still zero
        A->magic = GB_MAGIC2 ;
        A->p = NULL ;
        A->h = NULL ;
        ok = true ;
    }
    else if (Ap_option == GB_Ap_calloc)
    {
        // Sets the vector pointers to zero, which defines all vectors as empty
        A->magic = GB_MAGIC ;
        A->p = GB_CALLOC (A->plen+1, int64_t) ;
        ok = (A->p != NULL) ;
        if (A_is_hyper)
        { 
            // since nvec is zero, there is never any need to initialize A->h
            A->h = GB_MALLOC (A->plen, int64_t) ;
            ok = ok && (A->h != NULL) ;
        }
    }
    else // Ap_option == GB_Ap_malloc
    {
        // This is faster but can only be used internally by GraphBLAS since
        // the matrix is allocated but not yet completely initialized.  The
        // caller must set A->p [0..plen] and then set A->magic to GB_MAGIC,
        // before returning the matrix to the user application.  GB_NNZ(A) must
        // check A->nzmax == 0 since A->p [A->nvec] is undefined.
        A->magic = GB_MAGIC2 ;
        A->p = GB_MALLOC (A->plen+1, int64_t) ;
        ok = (A->p != NULL) ;
        if (A_is_hyper)
        { 
            A->h = GB_MALLOC (A->plen, int64_t) ;
            ok = ok && (A->h != NULL) ;
        }
    }

    if (!ok)
    {
        // out of memory
        if (allocated_header)
        { 
            // only free the header if it was allocated here
            GB_Matrix_free (Ahandle) ;
        }
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The vector pointers A->p are initialized only if Ap_calloc is true
    if (A->magic == GB_MAGIC)
    { 
        ASSERT_MATRIX_OK (A, "new matrix from GB_new", GB0) ;
    }
    return (GrB_SUCCESS) ;
}

