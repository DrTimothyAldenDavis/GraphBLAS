//------------------------------------------------------------------------------
// GB_AxB_dot5: compute C=A'*B, C<M>=A'*B or C<!M>=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_dot5 computes C in bitmap form.

// LAGraph_bfs_parent:
//      M present, complemented, valued, and dense
//      C: a vector, not in place
//      semiring: ANY_SECONDI1_INT32
//      C_replace: true
//      accum: not present

// OK: BITMAP (in progress)

#include "GB_mxm.h"
#include "GB_binop.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

GrB_Info GB_AxB_dot5                // A'*B, dot product method
(
    GrB_Matrix *Chandle,            // output matrix (if not done in place)
    GrB_Matrix C_in_place,          // input/output matrix, if done in place
    const GrB_Matrix M,             // mask matrix for C<M>=A'*B or C<!M>=A'*B
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // accum operator for C+=A'*B
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // prototype
    //--------------------------------------------------------------------------

    GrB_Info info ;

    // A is sparse (not hypersparse, bitmap, or full)
    if (!GB_IS_SPARSE (A)) return (GrB_NO_VALUE) ;

    // B is bitmap, and a single vector (not hypersparse, sparse, or full)
    if (!GB_IS_BITMAP (B)) return (GrB_NO_VALUE) ;
    if (B->vdim != 1) return (GrB_NO_VALUE) ;

    // semiring for BFS
    if (semiring != GxB_ANY_SECONDI1_INT32) return (GrB_NO_VALUE) ;

    // mask present, complemented, valued, dense, and GrB_INT32
    if (M == NULL) return (GrB_NO_VALUE) ;
    if (!GB_IS_FULL (M)) return (GrB_NO_VALUE) ;
    if (!Mask_comp) return (GrB_NO_VALUE) ;
    if (Mask_struct) return (GrB_NO_VALUE) ;
    if (M->type != GrB_INT32) return (GrB_NO_VALUE) ;

    // no accum
    if (accum != NULL) return (GrB_NO_VALUE) ;

    // no flipxy
    if (flipxy) return (GrB_NO_VALUE) ;

    // not in place (for now)
    if (C_in_place) return (GrB_NO_VALUE) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;

    ASSERT_MATRIX_OK_OR_NULL (C_in_place, "C_in_place for dot5 A'*B", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for dot5 A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot5 A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot5 A'*B", GB0) ;

    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_JUMBLED (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for numeric A'*B", GB0) ;
    ASSERT (A->vlen == B->vlen) ;

    // C is bitmap:
    int64_t cnvec = B->vdim ;
    int64_t cvlen = A->vdim ;
    int64_t cvdim = B->vdim ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_NNZ (A) ;
    int64_t bnz = GB_NNZ (B) ;
    int64_t cnz ;
    if (!GB_Index_multiply ((GrB_Index *) (&cnz), cvlen, cvdim))
    { 
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + bnz + cnz, chunk, nthreads_max) ;

    int64_t bnvec = B->nvec ;

    int naslice, nbslice ;
    if (nthreads == 1)
    { 
        naslice = 1 ;
        nbslice = 1 ;
    }
    else if (bnvec == 1)
    {
        // C and B are single vectors
        naslice = nthreads * 256 ;
        nbslice = 1 ;
    }
    else
    { 
        // TODO: this case is not handled yet
        return (GrB_NO_VALUE) ;
    }

    int ntasks = naslice * nbslice ;

    GBURBLE ("(dot5) ") ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_AxB_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate workspace and slice A and B
    //--------------------------------------------------------------------------


#if 0
    // A and B can have any sparsity: full, sparse, or hypersparse.
    // C is always created as sparse or hypersparse.

    if (!GB_pslice (&A_slice, A->p, A->nvec, naslice))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    if (!GB_pslice (&B_slice, B->p, B->nvec, nbslice))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }
#endif

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;

    // TODO: C->b could be malloc'd
    info = GB_create (Chandle, ctype, cvlen, cvdim, GB_Ap_calloc, true,
        GB_BITMAP, B->hyper_switch, cnvec, cnz, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    GrB_Matrix C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // get C, M, A, B
    //--------------------------------------------------------------------------

    #define GB_CTYPE int32_t
    #define GB_BTYPE GB_void    // not accessed
    #define GB_ATYPE GB_void    // not accessed

    int64_t  *GB_RESTRICT Cp = C->p ;
    int64_t  *GB_RESTRICT Ci = C->i ;
    int8_t   *GB_RESTRICT Cb = C->b ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    // int64_t bnvec = B->nvec ;
    int64_t bvlen = B->vlen ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    int64_t anvec = A->nvec ;
    int64_t avlen = A->vlen ;

    const int64_t *GB_RESTRICT Mp = M->p ;
    const int64_t *GB_RESTRICT Mh = M->h ;
    const int64_t *GB_RESTRICT Mi = M->i ;
    // assumes M is int32:
    // const GB_void *GB_RESTRICT Mx ;
    const int32_t *GB_RESTRICT Mx ;
    Mx = (int32_t *) (Mask_struct ? NULL : (M->x)) ;
    size_t msize = M->type->size ;
    const int64_t mnvec = M->nvec ;
    const int64_t mvlen = M->vlen ;
    bool M_is_hyper = GB_IS_HYPER (M) ;

    // TODO: if M is present and sparse and thus not used in place, scatter it
    // into the Cb bitmap with -1's and the rest 0's.  Then when done, walk
    // through the mask M again and set any -1's to 0.  This works for both
    // M and !M.  If M, then C(i,j) can be computed only if Cb [p] == -1,
    // and the result is Cb [p] = 0 or 1, depending on whether the entry
    // exists.  If !M, then C(i,j) can be computed only if Cb [p] == 0,
    // and the result is Cb [p] = 0 or 1, depending on whether the entry
    // exists.

    // If there is an accumulator, the existence of C(i,j) on input must be
    // preserved.  Use, say:
    //
    //  0:  C(i,j) does not exist on input, and M(i,j)=0
    //  1:  C(i,j) exists on input, and M(i,j)=0
    //  -1: C(i,j) does not exist on input, and M(i,j)=1
    //  2:  C(i,j) exists on input, and M(i,j)=1

    //--------------------------------------------------------------------------
    // GxB_ANY_FIRSTJ1_INT32 definitions:
    //--------------------------------------------------------------------------

        // for DOT5 (like DOT3)
        #define GB_CIJ_DECLARE(cij) \
            int32_t cij

        // for the ANY_FIRSTJ1_INT32 semiring:
        #define GB_GETA(aki, Ax, pA) ;
        #define GB_GETB(bkj, Bx, pB) ;
        #define GB_MULT(z,x,y,i,k,j) \
            z = (k+1)
        #define GB_MULTADD(z,x,y,i,k,j) \
            z = (k+1)
        #define GB_IDENTITY \
            0

        #define GB_PUTC(cij,p) Cx [p] = cij

        #define GB_IS_ANY_MONOID 1

    //--------------------------------------------------------------------------
    // C = A'*B, computing each entry with a dot product, via builtin semiring
    //--------------------------------------------------------------------------

    // A is sparse or hypersparse, M is dense and used in place, B is bitmap,
    // no accum, M is complemented and not structural.  Semiring is
    // GxB_ANY_FIRSTJ1_INT32.

    int64_t cnvals = 0 ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (int tid = 0 ; tid < ntasks ; tid++)
    { 
        // assume B is a single vector:
        int a_tid = tid ;
        int b_tid = 0 ;
        // int a_tid = tid / nbslice ;
        // int b_tid = tid % nbslice ;

        // determine the part of A and B to work on
        int64_t kA_start, kA_end, kB_start, kB_end ;
        GB_PARTITION (kA_start, kA_end, anvec, a_tid, naslice) ;
        // assume B is a single vector:
        // GB_PARTITION (kB_start, kB_end, bnvec, b_tid, nbslice) ;
        kB_start = 0 ;
        kB_end = 1 ;

        // compute C (kA_start:kA_start-1, kB_start:kB_start-1)
        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {
            // for B bitmap, sparse, or full, do this:
            int64_t j = kB ;
            // for all kinds of matrices B, do this instead:
            // int64_t j = GBH (Bh, kB) ;

            // since B is bitmap (also works for full):
            int64_t pB_start = kB * bvlen ;
            // this also works for all matrices, including bitmap:
            // int64_t pB_start = GBP (Bp, kB, bvlen) ;
            // int64_t pB_end   = GBP (Bp, kB+1, bvlen) ;

            // if A and/or B are hypersparse, then some rows and columns of
            // C will not be computed at all.  They must be set to Cb[p]=0,
            // unless they are preserved by the mask or accum.

            // since C is always bitmap:
            // pC_start = the start of C(:,j)
            int64_t pC_start = j * cvlen ;

            for (int64_t kA = kA_start ; kA < kA_end ; kA++)
            { 

                // if A is hypersparse, or for all matrices, do:
                // int64_t i = GBH (Ah, kA) ;
                // for A sparse:
                int64_t i = kA ;

                //--------------------------------------------------------------
                // compute C(i,j)<M(i,j) = A(:,i)'*B(:,j)
                //--------------------------------------------------------------

                // M is dense and used in-place (see M_dense_in_place in
                // GB_AxB_saxpy3)
                // pC = the location of C(i,j) in the bitmap
                int64_t pC = pC_start + i ;     // C is bitmap
                bool mij ;
                // since M is dense and used in place, it has the same
                // dimensions as C.
                mij = (Mx [pC] != 0) ;          // GB_mcast (Mx, pC, msize)
                if (!mij)
                { 

                    //----------------------------------------------------------
                    // C(i,j) = A(:,i)'*B(:,j)
                    //----------------------------------------------------------

                    // assumes A is sparse or hypersparse:
                    int64_t pA = Ap [kA] ; // GBP (Ap, kA, avlen) ;
                    int64_t pA_end = Ap [kA+1] ; // GBP (Ap, kA+1, avlen) ;

                    // GB_AxB_dot_cij starts here, if it did bitmaps:

                    GB_CIJ_DECLARE (cij) ;

                    // A(:,i) is sparse and B(:,j) is bitmap
                    #if !GB_IS_ANY_MONOID
                    bool cij_exists = false ;
                    cij = GB_IDENTITY ;
                    #endif
                    for (int64_t p = pA ; p < pA_end ; p++)
                    { 
                        // next index of A(:,k)
                        int64_t k = Ai [p] ;                // ok: A is sparse
                        // for any matrix A, do this instead:
                        // int64_t k = GBI (Ai, p, avlen) ;

                        // to handle A bitmap, do this:
                        // if (!GBB (Ab, p)) continue ;

                        // check existence of B(k,j): assumes B bitmap:
                        if (!Bb [pB_start+k]) continue ;
                        // to handle B bitmap or full, do:
                        // if (!GBB (Bb, pB_start+k)) continue ;

                        // see GB_DOT in Template/GB_AxB_dot_cij.c:
                        // cij += A(k,i) * B(k,j)
                        GB_GETA (aki, Ax, pA) ;             // aki = A(k,i)
                        GB_GETB (bkj, Bx, pB_start+k) ;     // bkj = B(k,j)
                        GB_MULTADD (cij, aki, bkj, i, k, j) ;
                        #if GB_IS_ANY_MONOID
                        // for the ANY monoid: always terminal:
                        cnvals++ ;              // one more entry in the bitmap
                        Cb [pC] = 1 ;           // assumes Cb is calloc'ed
                        GB_PUTC (cij, pC) ;     // Cx [pC] = cij
                        break ;
                        #else
                        cij_exists = true ;
                        // test terminal condition here
                        #endif
                    }

                    #if !GB_IS_ANY_MONOID
                    if (cij_exists)
                    { 
                        cnvals++ ;              // one more entry in the bitmap
                        Cb [pC] = 1 ;           // assumes Cb is calloc'ed
                        GB_PUTC (cij, pC) ;     // Cx [pC] = cij
                    }
                    ASSERT (Cb [pC] == cij_exists) ;
                    #endif

                }
                else
                { 

                    //----------------------------------------------------------
                    // the mask prevents C(i,j) from existing
                    //----------------------------------------------------------

                    // commented out because C->b is calloc'd above
                    // Cb [pC] = 0 ;
                    ASSERT (Cb [pC] == 0) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    C->magic = GB_MAGIC ;

    ASSERT_MATRIX_OK (C, "dot5: C = A'*B output", GB0) ;
    ASSERT (*Chandle == C) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    return (GrB_SUCCESS) ;
}

