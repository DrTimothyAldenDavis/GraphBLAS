//------------------------------------------------------------------------------
// GB_AxB_flopcount:  compute flops for C<M>=A*B or C=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// On input, A and B are two matrices for C<M>=A*B or C=A*B.  The flop count
// for each B(:,j) is computed, and returned as a cumulative sum.  This
// function is CSR/CSC agnostic, but for simplicity of this description, assume
// A and B are both CSC matrices, so that ncols(A) == nrows(B).  For both CSR
// and CSC, A->vdim == B->vlen holds.  A and/or B may be hypersparse, in any
// combination.

// The complemented mask is not handled, so the flops for C<!M>=A*B is not
// computed.

// If present, Bflops has size (B->nvec)+1, for both standard and hypersparse
// B.  Let n = B->vdim be the column dimension of B (that is, B is m-by-n).

// If B is a standard CSC matrix then Bflops has size n+1 == B->nvec+1, and on
// output, Bflops [j] is the # of flops required to compute C (:, 0:j-1).  B->h
// is NULL, and is implicitly the vector 0:(n-1).

// If B is hypersparse, then let Bh = B->h.  Its size is B->nvec, and j = Bh
// [kk] is the (kk)th column in the data structure for B.  C will also be
// hypersparse, and only C(:,Bh) will be computed (C may have fewer non-empty
// columns than B).  On output, Bflops [kk] is the number of needed flops to
// compute C (:, Bh [0:kk-1]).

// In both cases, Bflops [0] = 0, and Bflops [B->nvec] = total number of flops.
// The size of Bflops is B->nvec+1 so that it has the same size as B->p.  The
// first entry of B->p and Bflops are both zero.  This allows B to be sliced
// either by # of entries in B (by slicing B->p) or by the flop count required
// (by slicing Bflops).

// This algorithm does not look at the values of M, A, or B, just their
// patterns.  If the mask is present, it is assumed to not be complemented.
// The flop count of C=A*B or C<M>=A*B is computed for a saxpy-based method;
// the work for A'*B for the dot product method is not computed.

// The algorithm scans all nonzeros in B.  It only scans at most the min and
// max (first and last) row indices in A and M (if M is present).  If A and M
// are not hypersparse, the time taken is O(nnz(B)+n).  If all matrices are
// hypersparse, the time is O(nnz(B)*log(h)) where h = max # of vectors present
// in A and M.  In pseudo-MATLAB, and assuming B is in standard (not
// hypersparse) form:

/*
    [m n] = size (B) ;
    Bflops = zeros (1,n+1) ;        % (set to zero in the caller)
    for each column j in B:
        if (B (:,j) is empty) continue ;
        if (M is present and M (:,j) is empty) continue ;
        im_first = min row index of nonzeros in M(:,j)
        im_last  = max row index of nonzeros in M(:,j)
        for each k where B (k,j) is nonzero:
            aknz = nnz (A (:,k))
            if (aknz == 0) continue ;
            alo = min row index of nonzeros in A(:,k)
            ahi = max row index of nonzeros in A(:,k)
            if (M is present)
                if (intersection (alo:ahi, im_first:im_last) empty) continue
            end
            % numerical phase will compute: C(:,j)<M(:,j)> += A(:,k)*B(k,j),
            % which takes aknz flops, so:
            Bflops (j) += aknz
            Bflops_per_entry (k,j) = aknz
        end
    end
*/ 

// If Bflops is NULL, then this function is being called by a single thread.
// Bflops is not computed.  Instead, the total_flops are computed, and the
// function returns just the result of the test (total_flops <= floplimit).
// total_flops is not returned, just the true/false result of the test.  This
// allows the function to return early, once the total_flops exceeds the
// threshold.

#include "GB.h"

bool GB_AxB_flopcount           // compute flops for C<M>=A*B or C=A*B
(
    int64_t *Bflops,            // size B->nvec+1 and all zero, if present
    int64_t *Bflops_per_entry,  // size nnz(B)+1 and all zero, if present
    const GrB_Matrix M,         // optional mask matrix
    const GrB_Matrix A,
    const GrB_Matrix B,
    int64_t floplimit,          // maximum flops to compute if Bflops NULL
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_OK_OR_NULL (GB_check (M, "M for flop count A*B", GB0)) ;
    ASSERT_OK (GB_check (A, "A for flop count A*B", GB0)) ;
    ASSERT_OK (GB_check (B, "B for flop count A*B", GB0)) ;
    ASSERT (!GB_PENDING (M)) ; ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (A->vdim == B->vlen) ;

    int64_t bnz = GB_NNZ (B) ;
    int64_t bnvec = B->nvec ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    bool check_quick_return = (Bflops == NULL) && (Bflops_per_entry == NULL) ;

    #ifndef NDEBUG
    if (check_quick_return)
    {
        // a single thread is testing the condition (total_flops <= floplimit)
        ASSERT (Context == NULL) ;
        ASSERT (nthreads == 1) ;
    }
    if (Bflops != NULL)
    {
        // Bflops is set to zero in the calller
        for (int64_t kk = 0 ; kk <= bnvec ; kk++)
        {
            ASSERT (Bflops [kk] == 0) ;
        }
    }
    if (Bflops_per_entry != NULL)
    {
        // Bflops_per_entry is set to zero in the calller
        for (int64_t pB = 0 ; pB <= bnz ; pB++)
        {
            ASSERT (Bflops_per_entry [pB] == 0) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // get the mask, if present
    //--------------------------------------------------------------------------

    const int64_t *restrict Mh = NULL ;
    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mi = NULL ;
    int64_t mnvec = 0 ;
    bool M_is_hyper = GB_IS_HYPER (M) ;
    if (M != NULL)
    { 
        Mh = M->h ;
        Mp = M->p ;
        Mi = M->i ;
        mnvec = M->nvec ;
    }

    //--------------------------------------------------------------------------
    // get A and B
    //--------------------------------------------------------------------------

    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;
    int64_t anvec = A->nvec ;
    bool A_is_hyper = GB_IS_HYPER (A) ;

    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bi = B->i ;
    bool B_is_hyper = GB_IS_HYPER (B) ;

    //--------------------------------------------------------------------------
    // compute flop counts for C<M> = A*B
    //--------------------------------------------------------------------------

    int64_t total_flops = 0 ;
    bool quick_return = false ;

    #pragma omp parallel for num_threads(nthreads) \
        reduction(+:total_flops)                   \
        firstprivate(quick_return)
    for (int64_t kk = 0 ; kk < bnvec ; kk++)
    {

        // The (kk)th iteration of this loop computes Bflops [kk], if not NULL.
        // All iterations are completely independent.

        if (quick_return)
        { 
            // TODO: if quick_return is true, then Bflops is NULL and the
            // computations are being done by a single thread, and the thread
            // can terminate.  But a break statement cannot be used here.
            continue ;
        }

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        int64_t j = (B_is_hyper) ? Bh [kk] : kk ;
        int64_t pB     = Bp [kk] ;
        int64_t pB_end = Bp [kk+1] ;

        // C(:,j) is empty if B(:,j) is empty
        int64_t bjnz = pB_end - pB ;
        if (bjnz == 0) continue ;

        //----------------------------------------------------------------------
        // see if M(:,j) is present and non-empty
        //----------------------------------------------------------------------

        int64_t im_first = -1, im_last = -1 ;
        if (M != NULL)
        { 
            // TODO: can reuse mpleft from the last binary search of M->h, to
            // speed up the search when M is hypersparse.  This is just a
            // heuristic, and resetting mpleft to zero here would work too
            // (just more of M->h would be searched; the results would be the
            // same), as in:
            int64_t mpleft = 0 ;     // this works too
            // To reuse mpleft from its prior iteration, each thread needs its
            // own threadprivate mpleft, to use in all its iterations
            int64_t mpright = mnvec - 1 ;
            int64_t pM, pM_end ;
            GB_lookup (M_is_hyper, Mh, Mp, &mpleft, mpright, j, &pM, &pM_end) ;
            int64_t mjnz = pM_end - pM ;
            // C(:,j) is empty if M(:,j) is empty
            if (mjnz == 0) continue ;
            // M(:,j) has at least one entry; get 1st and last index in M(:,j)
            im_first = Mi [pM] ;
            im_last  = Mi [pM_end-1] ;
        }

        //----------------------------------------------------------------------
        // trim Ah on right
        //----------------------------------------------------------------------

        // Ah [0..A->nvec-1] holds the set of non-empty vectors of A, but only
        // vectors k corresponding to nonzero entries B(k,j) are accessed for
        // this vector B(:,j).  If nnz (B(:,j)) > 2, prune the search space on
        // the right, so the remaining calls to GB_lookup will only need to
        // search Ah [pleft...pright-1].  pright does not change.  pleft is
        // advanced as B(:,j) is traversed, since the indices in B(:,j) are
        // sorted in ascending order.

        int64_t pleft = 0 ;
        int64_t pright = anvec-1 ;
        if (A_is_hyper && bjnz > 2)
        { 
            // trim Ah [0..pright] to remove any entries past the last B(:,j)
            GB_bracket_right (Bi [pB_end-1], Ah, 0, &pright) ;
        }

        //----------------------------------------------------------------------
        // count the flops to compute C(:,j)<M(:,j)> = A*B(:,j)
        //----------------------------------------------------------------------

        int64_t bjflops = 0 ;

        // TODO: the following for loop could also be parallel.  pleft would
        // need to be set to zero before each call to GB_lookup, or private for
        // each thread.  It is currently updated after each search to reduce
        // the work in subsequent binary searches.  The break statement would
        // need to be removed.  Doing the following loop in parallel would be
        // important if B is a single dense vector, for example.  In that case,
        // the outer loop is a single iteration, with bnvec == 1.

        // reduction (+:bjflops, +:total_flops)
        for ( ; pB < pB_end ; pB++)
        {
            // B(k,j) is nonzero
            int64_t k = Bi [pB] ;

            // find A(:,k), reusing pleft since Bi [...] is sorted
            int64_t pA, pA_end ;
            GB_lookup (A_is_hyper, Ah, Ap, &pleft, pright, k, &pA, &pA_end) ;

            // skip if A(:,k) empty
            int64_t aknz = pA_end - pA ;
            if (aknz == 0) continue ;

            // skip if intersection of A(:,k) and M(:,j) is empty
            if (M != NULL)
            { 
                // A(:,k) is non-empty; get the first and last index of A(:,k)
                int64_t alo = Ai [pA] ;
                int64_t ahi = Ai [pA_end-1] ;
                if (ahi < im_first || alo > im_last) continue ;
            }

            // increment by flops for the single entry B(k,j)
            // C(:,j)<M(:,j)> += A(:,k)*B(k,j).
            bjflops += aknz ;

            if (Bflops_per_entry != NULL)
            { 
                // flops for the single entry, B(k,j)
                Bflops_per_entry [pB] = aknz ;
            }

            // check for a quick return
            if (check_quick_return)
            {
                // the work is being done by a single thread
                ASSERT (nthreads == 1) ;
                total_flops += aknz ;
                if (total_flops > floplimit)
                { 
                    // quick return:  (total_flops <= floplimit) is false.
                    // total_flops is not returned since it is only partially
                    // computed.  However, it does not exceed the floplimit
                    // threshold, so the result is false.
                    quick_return = true ;
                    break ;
                }
            }
        }

        if (Bflops != NULL)
        { 
            Bflops [kk] = bjflops ;
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Bflops and Bflops_per_entry
    //--------------------------------------------------------------------------

    if (Bflops != NULL)
    { 
        // Bflops = cumsum ([0 Bflops]) ;
        ASSERT (Bflops [bnvec] == 0) ;
        GB_cumsum (Bflops, bnvec, NULL, nthreads) ;
        // Bflops [bnvec] is now the total flop count
        // printf ("flop count %g (per col)\n", (double) Bflops [bnvec]) ;
        total_flops = Bflops [bnvec] ;
    }

    if (Bflops_per_entry != NULL)
    { 
        // Bflops_per_entry = cumsum ([0 Bflops_per_entry]) ;
        ASSERT (Bflops_per_entry [bnz] == 0) ;
        GB_cumsum (Bflops_per_entry, bnz, NULL, nthreads) ;
        // Bflops_per_entry [bnz] is now the total flop count
        // printf ("flop count %g (per entry)\n",
        // (double) Bflops_per_entry [bnz]);
        total_flops = Bflops_per_entry [bnz] ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (total_flops <= floplimit) ;
}

