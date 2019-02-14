//------------------------------------------------------------------------------
// GB_AxB_parallel: C<M>=A*B, C<M>=A'*B, C=A*B, or C=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Parallel matrix-matrix multiply, A*B or A'*B, with optional mask M.  This
// method is used by GrB_mxm, GrB_vxm, and GrB_mxv.  For both of the latter two
// methods, B on input will be an nrows-by-1 column vxector.
//
// If do_adotb is true, then A'*B is being computed.  In this case, A has
// not been transposed yet (and will not be).  A and B must have the same
// vector length, vlen (as if both A and B are CSC matrices with the same
// number of rows, for example).  The GB_AxB_sequential method when applied
// to A and B (or a slice of B) will operate on A' without forming it.
//
// If do_adotb is false, then A*B is being computed, and the vector dimension
// of A must be identical to the vector length of B (as if both A and B are
// CSC matrices, and the number of columns of A is the same as the number of
// rows of B).
//
// The output matrix C = *Chandle has not been allocated, so C is NULL on
// input.  The mask M is optional.  If present, it is not complemented.
//
// The semiring defines C=A*B, and must be valid.  flipxy modifies how the
// semiring multiply operator is applied.  If false, then fmult(aik,bkj)
// is computed.  If true, then the operands are swapped, and fmult(bkj,aij)
// is done instead.
//
// AxB_method selects the method to use:
//      GxB_DEFAULT:        the method is selected automatically
//      GxB_AxB_GUSTAVSON:  Gustavson's method for A*B
//      GxB_AxB_HEAP:       heap method for A*B
//      GxB_AxB_DOT:        dot method for A'*B
//      GxB_AxB_HASH:       hash method for A*B (TODO)
//
// AxB_method_used reports the method actually chosen.  This is for
// informational purposes only, so if a parallel C=A*B splits the work into
// multiple submatrix multiplications, and uses different methods on each
// submatrix, then AxB_method_used is the method chosen by thread zero.
//
// Context: the GB_Context containing the # of threads to use, a string of the
// user-callable function that is calling this function (GrB_mxm, GrB_mxv, or
// GxB_vxm) and detailed error reports.
//
// C, M, A, and B are CSR/CSC agnostic.  For this discussion, suppose they
// are CSC, with vlen = # of rows, and vdim = # of columns.

// PARALLEL: A' or B is sliced, and the slices of C are concatenated.

// FUTURE: The result of this function is the T matrix in GB_mxm.  It may be
// transplanted directly into the user's matrix, or it may be passed through
// GB_accum_mask.  See GB_mxm.  For the latter case, it would be better to
// delay the contatenation of the output matrix T = [C0 ... C(t-t)].
// GB_accum_mask could do the accum/mask using the sliced T matrix, to
// update the user's C matrix (which is not sliced), and then T is freed.

#include "GB.h"

GrB_Info GB_AxB_parallel            // parallel matrix-matrix multiply
(
    GrB_Matrix *Chandle,            // output matrix, NULL on input
    GrB_Matrix M,                   // optional mask matrix
    const bool Mask_comp,           // if true, use ~M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const bool do_adotb,            // if true, do A'*B via dot products
    const GrB_Desc_Value AxB_method,// for auto vs user selection of methods
//  const GrB_Desc_Value AxB_slice, // how to slice B or A'
    GrB_Desc_Value *AxB_method_used,// method selected
    bool *mask_applied,             // if true, mask was applied
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Chandle != NULL) ;          // C = (*Chandle) is NULL
    ASSERT (*Chandle == NULL) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for parallel A*B", GB0)) ;
    ASSERT_OK (GB_check (A, "A for parallel A*B", GB0)) ;
    ASSERT_OK (GB_check (B, "B for parallel A*B", GB0)) ;
    ASSERT (!GB_PENDING (M)) ; ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT_OK (GB_check (semiring, "semiring for parallel A*B", GB0)) ;
    ASSERT (AxB_method_used != NULL) ;

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    nthreads = 1 ;      // TODO: hack for now

    // nthreads = GB_IMAX (4, B->vdim) ;
    // if (do_adotb) nthreads = 1 ;

    //==========================================================================
    // sequential C<M>=A*B, C<M>=A'*B, C=A*B, or C=A'*B
    //==========================================================================

    #define GB_FREE_ALL ;

    // if (nthreads == 1)
    {

        // do the entire computation with a single thread

        // select the method
        // printf ("select the method %d\n", AxB_method) ;
        int64_t bjnz_max ;
        GB_AxB_select (A, B, semiring, do_adotb, AxB_method,
            AxB_method_used, &bjnz_max) ;

        // printf ("selected method %d\n", *AxB_method_used) ;

        // acquire a Sauna if Gustavson's method is being used
        int Sauna_id = -2 ;
        if (*AxB_method_used == GxB_AxB_GUSTAVSON)
        {
            // printf ("get sauna\n") ;
            GB_OK (GB_Sauna_acquire (1, &Sauna_id, AxB_method_used, Context)) ;
            // printf ("got sauna %d\n", Sauna_id) ;
        }

        // C<M>=A*B or A'*B
        GrB_Info thread_info = GB_AxB_sequential (Chandle, M, Mask_comp,
            A, B, semiring, flipxy, *AxB_method_used, bjnz_max, mask_applied,
            Sauna_id) ;

        // printf ("info from GB_AxB_sequential is %d\n", thread_info) ;

        // release the Sauna for Gustavson's method
        if (*AxB_method_used == GxB_AxB_GUSTAVSON)
        {
            // printf ("release sauna\n") ;
            GB_OK (GB_Sauna_release (1, &Sauna_id)) ;
            // printf ("released sauna\n") ;
        }

        info = thread_info ;
        // printf ("thread_info now is %d\n", info) ;

        return ((info == GrB_OUT_OF_MEMORY) ? GB_OUT_OF_MEMORY : info) ;
    }

    //==========================================================================
    // parallel C<M>=A*B, C<M>=A'*B, C=A*B, or C=A'*B
    //==========================================================================

    // TODO C=A*B below is only partially drafted.

#if 0


    //--------------------------------------------------------------------------
    // select the method for slicing B or A'
    //--------------------------------------------------------------------------

    int64_t anvec = A->nvec ;
    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    int64_t bnvec = B->nvec ;
    int64_t bvdim = B->vdim ;
    int64_t bvlen = B->vlen ;

    bool check_for_dense_mask = true ;

    bool slice_A ;      // true if slicing A', false if slicing B

    if (do_adotb)
    {
        
        //----------------------------------------------------------------------
        // select slicing method for A'*B
        //----------------------------------------------------------------------

        // C<M>=A'*B:  either slice the columns of B or rows of A'.  The dot
        // product method will be used.  Slicing the columns of B is the same
        // as for A*B, except that the flopcounts for each column of A*B are
        // not simple to compute.  The *FINE methods would likely be overkill
        // for the dot product.  Thus, the available methods are:

        // GxB_SLICE_ATROW:      each slice of A' has same # of rows
        // GxB_SLICE_ATNZ:       each slice of A' has same # of nonzeros
        // GxB_SLICE_BCOL:       each slice of B has same # of columnns
        // GxB_SLICE_BNZ:        each slice of B has same # of nonzeros

        // Note that the GxB_SLICE_BCOL and GxB_SLICE_BNZ methods can be used
        // for both A'*B and A*B.

        if (AxB_slice < GxB_SLICE_ATROW || AxB_slice > GxB_SLICE_BNZ)
        {
            // invalid method, so use default rule instead
            AxB_slice = GxB_DEFAULT ;
        }

        if (AxB_slice == GxB_DEFAULT)
        {
            // select the method automatically for A'*B
            AxB_slice = GxB_SLICE_ATNZ ;
        }

        // can slice A or B when computing A'*B
        slice_A = (AxB_slice == GxB_SLICE_ATROW) ||
                  (AxB_slice == GxB_SLICE_ATNZ) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // select slicing method for A*B
        //----------------------------------------------------------------------

        // C<M>=A*B is being computed, with C, A, and B all in the same format
        // (the comments here assume they are all CSC, but the methods are
        // CSR/CSC agnostic).  When computing A*B, the matrix A cannot be
        // sliced easily, but the flop counts for each column B(:,j) or even
        // each entry B(k,j) can be computed.

        // GxB_SLICE_BCOL:       each slice of B has same # of columnns
        // GxB_SLICE_BNZ:        each slice of B has same # of nonzeros
        // GxB_SLICE_BFLOPS:     each slice of B has same # of flops

        // TODO: GxB_SLICE_B*FINE not yet implemented
        // GxB_SLICE_BNZFINE:    like BNZ but split inside columns of B
        // GxB_SLICE_BFLOPSFINE: like BFLOPS but split inside columns of B

        if (AxB_slice < GxB_SLICE_BCOL || GxB_SLICE > GxB_SLICE_BFLOPS)
        {
            // invalid method, so use default rule instead
            AxB_slice = GxB_DEFAULT ;
        }

        if (AxB_slice == GxB_DEFAULT)
        {
            // select the method automatically for A*B
            AxB_slice = GxB_SLICE_BFLOPS ;
        }

        // can only slice B when doing A*B
        slice_A = false ;
    }

    //--------------------------------------------------------------------------
    // find where to slice and compute C=A'*B or A*B accordingly.
    //--------------------------------------------------------------------------

    int64_t Slice [nthreads+1] ;
    GrB_Matrix *Cslice = NULL ;
    GrB_Matrix *Bslice = NULL ;
    GrB_Matrix *Aslice = NULL ;

    #undef  GB_FREE_ALL
    #define GB_FREE_ALL                             \
    {                                               \
        GB_FREE_SLICE (&Cslice, nthreads) ;         \
        GB_FREE_SLICE (&Bslice, nthreads) ;         \
        GB_FREE_SLICE (&Aslice, nthreads) ;         \
    }

    GB_CALLOC_MEMORY (Cslice, nthreads+1, sizeof (GrB_Matrix)) ;
    if (Cslice == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GB_OUT_OF_MEMORY) ;
    }

    if (slice_A)
    {
        
        //----------------------------------------------------------------------
        // slice A' for C=A'*B
        //----------------------------------------------------------------------

        // thread t will do rows Slice [t] to Slice [t+1]-1 of A'
        Slice [nthreads] = anvec ;

        if (AxB_slice == GxB_SLICE_ATROW)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_ATROW:  slice A' by row
            //------------------------------------------------------------------

            // GxB_SLICE_ATROW:  slice A' so that each slice has a balanced # of
            // rows per thread.  Each thread gets roughly anvec/nthreads rows.

            for (int t = 1 ; t < nthreads ; t++)
            {
                Slice [t] = ((t * (double) anvec) / (double) nthreads) ;
            }

        }
        else if (AxB_slice == GxB_SLICE_ANZ)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_ANZ: slice A' by nz
            //------------------------------------------------------------------

            // GxB_SLICE_ANZ: slice A' so that each slice has a balanced # of
            // entries per thread.  Each thread gets enough rows of A so that
            // it has roughly anz/nthreads entries in its slice.  Do not split
            // any row of A'.  Cheap, and likely better than GxB_SLICE_ATROW
            // for most matrices.  This method will work well for GrB_mxv when
            // A is stored by row and v is dense.

            int64_t anz = GB_NNZ (A) ;
            int64_t pleft = 0 ;
            for (int t = 1 ; t < nthreads ; t++)
            {
                // binary search to find k so that Ap [k] == (t * anz) /
                // nthreads.  The exact value will not typically not be found;
                // just pick what the binary search comes up with.
                int64_t nz = ((t * (double) anz) / (double) nthreads) ;
                int64_t pright = anvec ;
                GB_BINARY_TRIM_SEARCH (nz, Ap, pleft, pright) ;
                Slice [t] = pleft ;
            }

        }

        //----------------------------------------------------------------------
        // construct each slice of A'
        //----------------------------------------------------------------------

        GB_OK (GB_slice (A, nthreads, Slice, &Aslice, Context)) :

        //----------------------------------------------------------------------
        // compute each slice of C = A'*B, with optional mask M
        //----------------------------------------------------------------------

        // only GxB_AxB_dot can be used here
        (*AxB_method_used) = GxB_AxB_DOT ;

        // for all threads in parallel, with no synchronization
        bool ok = true ;
        bool panic = false ;

        // reduction(&&:ok,||:panic)
        for (int t = 0 ; t < nthreads ; t++)
        {
            GrB_Info thread_info = GB_AxB_dot (&(Cslice [t]), M, &(Aslice [t]),
                B, semiring, flipxy) ;

            // collect all thread-specific info 
            ok    = ok    && (thread_info == GrB_SUCCESS) ;
            panic = panic || (thread_info == GrB_PANIC) ;
        }

        if (!ok)
        {
            // out of memory, or panic if a critical section fails
            return (panic ? GrB_PANIC : GB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // concatenate the slices of C
        //----------------------------------------------------------------------

        // C = [Cslice(0) ; Cslice(1) ; ... ; Cslice(nthreads-1)] concatenated
        // vertically.  Each slice Cslice(t) has the same dimensions as C, but
        // each one contains entries that appear in a unique and contiguous
        // subset of the rows of C.  C is stored by column.

        /*
        GB_OK (GB_vcat_slice (Chandle, nthreads, Cslice)) ;
        */

    }
    else
    {

        //----------------------------------------------------------------------
        // slice B for A*B or A'*B
        //----------------------------------------------------------------------

        // thread t will do columns Slice [t] to Slice [t+1]-1
        Slice [nthreads] = bnvec ;

        if (AxB_slice == GxB_SLICE_BCOL)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_BCOL: slice B by column
            //------------------------------------------------------------------

            // GxB_SLICE_BCOL:  slice B so that each slice has a balanced # of
            // columns per thread.  Each thread gets roughly bnvec/nthreads
            // columns.  Do not split any column.  Very cheap but not likely to
            // perform well unless the matrices A and B have uniform nonzero
            // pattern.

            for (int t = 1 ; t < nthreads ; t++)
            {
                Slice [t] = ((t * (double) bnvec) / (double) nthreads) ;
            }

        }
        else if (AxB_slice == GxB_SLICE_BNZ)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_BCOL: slice B by nz
            //------------------------------------------------------------------

            // GxB_SLICE_BNZ: slice B so that each slice has a balanced # of
            // entries per thread.  Each thread gets enough columns of B so
            // that it has roughly bnz/nthreads entries in its slice.  Do not
            // split any column of B.  Cheap, and likely better than
            // GxB_SLICE_BCOL for most matrices.

            int64_t bnz = GB_NNZ (B) ;
            int64_t pleft = 0 ;
            for (int t = 1 ; t < nthreads ; t++)
            {
                // binary search to find k so that Bp [k] == (t * bnz) /
                // nthreads.  The exact value will not typically not be found;
                // just pick what the binary search comes up with.
                int64_t nz = ((t * (double) bnz) / (double) nthreads) ;
                int64_t pright = bnvec ;
                GB_BINARY_TRIM_SEARCH (nz, Bp, pleft, pright) ;
                Slice [t] = pleft ;
            }

        }
        else if (AxB_slice == GxB_SLICE_BFLOPS)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_BFLOPS: slice B by flops
            //------------------------------------------------------------------

            // GxB_SLICE_BFLOPS: slice B so that each slice has a balanced
            // amount of flops, to compute its slice of C.  Each thread gets
            // enough columns of B so that it has roughly total_flops /
            // nthreads work to do.  This should give a very good load balance,
            // but at the cost of a more expensive symbolic analysis, taking
            // O(bnz) time.  The analysis is itself fully parallel, however.
            // This can only be done when computing A*B, not A'*B.

            int64_t *Bflops ;
            GB_MALLOC_MEMORY (Bflops, bnvec+1, sizeof (int64_t)) ;
            if (Bflops == NULL)
            {
                // out of memory
                FREE_WORKSPACE ;
                return (GB_OUT_OF_MEMORY) ;
            }
            GB_AxB_flopcount (Bflops, (Mask_comp) ? NULL : M, A, B, Context) ;
            int64_t total_flops = Bflops [bnvec] ;
            for (int t = 1 ; t < nthreads ; t++)
            {
                // binary search to find k so that Bp [k] == (t * bnz) /
                // nthreads.  The exact value will not typically not be found;
                // just pick what the binary search comes up with.
                int64_t f = ((t * (double) total_flops) / (double) nthreads) ;
                int64_t pright = bnvec ;
                GB_BINARY_TRIM_SEARCH (f, Bflops, pleft, pright) ;
                Slice [t] = pleft ;
            }
            int64_t total_flops = Bflops [bnvec] ;
            if (total_flops < GB_NNZ (M))
            {
                // Discard the mask, it is too costly to use
                M = NULL ;
            }
            GB_FREE_MEMORY (Bflops, bnvec+1, sizeof (int64_t)) ;

        }

        #if 0
        else if (AxB_slice == GxB_SLICE_BNZFINE)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_BNZFINE:  slice B by nz (split columns of B)
            //------------------------------------------------------------------

            // GxB_SLICE_BNZFINE:  slice B so that each slice has nearly
            // exactly balanced number of entries.  Individual columns of B may
            // need to be sliced inside, and shared between different threads,
            // so the last column of Bslice[t] and the first column of
            // Bslice[t+1] might be the same column.  If shared between just
            // two threads, then some first low-numbered row indices would be
            // owned by thread t, and the higher-numbered ones would be owned
            // by thread t+1.  The resulting Cslices must be summed, not
            // concatenated.  This method will work well when B is a single
            // column, which will be important for sparse matrix times sparse
            // vector (as in BFS): vxm when the matrix is CSR, or mxv when the
            // matrix is CSC (in both cases where the matrix is not transposed
            // with the descriptor).  In both cases, v is the matrix B, here.

            // TODO not yet implemented

        }
        else if (AxB_slice == GxB_SLICE_BFLOPSFINE)
        {

            //------------------------------------------------------------------
            // GxB_SLICE_BFLOPSFINE:  slice B by flops (split columns of B)
            //------------------------------------------------------------------

            // GxB_SLICE_BFLOPSFINE:  slice B so that each slice has nearly
            // exactly balanced amount of flops to compute its slice of C.
            // Each thread gets exactly the number of entries so that it does
            // total_flops/nthreads work (rounded to the nearest number of
            // entries in B).  Otherwise, the same as GxB_SLICE_BNZFINE.  This
            // method may be overkill.  GxB_SLICE_BNZFINE might be enough.

            // TODO not yet implemented (perhaps never)

        }
        #endif

        //----------------------------------------------------------------------
        // construct each slice of B
        //----------------------------------------------------------------------

        GB_OK (GB_slice (B, nthreads, Slice, &Bslice, Context)) :

        ASSERT (0) ;

        //----------------------------------------------------------------------
        // select the method for each slice
        //----------------------------------------------------------------------

        GrB_Desc_Value AxB_methods_used [nthreads] ;
        int bjnz_max [nthreads] ;
        int Sauna_ids [nthreads] ;

        // for all threads in parallel, no dependencies except the reduction
        bool any_Gustavson = false ;
        // reduction (||:any_Gustavson)
        for (int t = 0 ; t < nthreads ; t++)
        {
            GB_AxB_select (A, &(Bslice [t]), semiring, false,
                &(AxB_methods_used [t]), &(bjnz_max [t])) ;
            any_Gustavson = any_Gustavson ||
                (AxB_methods_used [t] == GB_AxB_GUSTAVSON) ;
        }

        (*AxB_method_used) = AxB_methods_used [0] ;

        //----------------------------------------------------------------------
        // acquire the Saunas for each thread that needs it
        //----------------------------------------------------------------------

        if (any_Gustavson)
        {
            // at least one thread needs a Sauna
            GB_OK (GB_Sauna_acquire (nthreads, Sauna_ids, AxB_methods_used,
                Context)) ;
        }
        else
        {
            // no thread needs a Sauna
            for (int t = 0 ; t < nthreads ; t++)
            {
                Sauna_ids [t] = -2 ;
            }
        }

        //----------------------------------------------------------------------
        // compute each slice of C = A*B or A'*B, with optional mask M
        //----------------------------------------------------------------------

        // for all threads in parallel, with no synchronization except
        // for these boolean reductions:
        bool ok = true ;
        bool panic = false ;
        bool allmask = true ;

        // reduction(&&:ok,||:panic,&&:allmask)
        for (int t = 0 ; t < nthreads ; t++)
        {
            bool thread_mask_applied = false ;
            GrB_Info thread_info = GB_AxB_sequential (&(Cslice [t]), M,
                Mask_comp, A, Bslice [t], semiring, flipxy,
                AxB_methods_used [t], bjnz_max [t], &thread_mask_applied,
                Sauna_ids [t]) ;

            // collect all thread-specific info 
            ok      = ok      && (thread_info == GrB_SUCCESS) ;
            panic   = panic   || (thread_info == GrB_PANIC) ;
            allmask = allmask && (thread_mask_applied) ;
        }

        // panic if a critical section fails
        if (panic) return (GrB_PANIC) ;

        // if all threads applied the mask to their slices, then GB_accum_mask
        // does not need to apply it to the concatenated C in GB_AxB_meta.  If
        // just some of them did, then GB_accum_mask needs to apply the mask
        // again.
        (*mask_applied) = allmask ;

        //----------------------------------------------------------------------
        // release the Saunas, if any
        //----------------------------------------------------------------------

        if (any_Gustavson)
        {
            // at least one thread used a Sauna
            GB_OK (GB_Sauna_release (nthreads, Sauna_ids)) ;
        }

        // check the return info from all the threads
        if (!ok)
        {
            // out of memory
            return (GB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // concatenate the slices of C
        //----------------------------------------------------------------------

        // C = [Cslice(0) Cslice(1) ... Cslice(nthreads-1)] concatenatied
        // horizontally.  Each slice Cslice(t) has the same dimensions as C,
        // but each one contains entries that appear in a unique and contiguous
        // subset of the columns of C.  C is stored by column.

        /*
        GB_OK (GB_hcat_slice (Chandle, nthreads, Cslice, Context)) ;
        */

        // TODO
        ASSERT (0) ;

    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
#endif
}

