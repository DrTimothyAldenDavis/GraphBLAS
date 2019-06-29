//------------------------------------------------------------------------------
// GB_subassigner: C(I,J)<#M> = accum (C(I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Submatrix assignment: C(I,J)<M> = A, or accum (C (I,J), A), no transpose

// All assignment operations rely on this function, including the GrB_*_assign
// operations in the spec, and the GxB_*_subassign operations that are a
// SuiteSparse:GraphBLAS extension to the spec:

// GrB_Matrix_assign,
// GrB_Matrix_assign_TYPE,
// GrB_Vector_assign,
// GrB_Vector_assign_TYPE,
// GrB_Row_assign,
// GrB_Col_assign

// GxB_Matrix_subassign,
// GxB_Matrix_subassign_TYPE,
// GxB_Vector_subassign,
// GxB_Vector_subassign_TYPE,
// GxB_Row_subassign,
// GxB_Col_subassign

// This function handles the accumulator, and the mask M, and the C_replace
// option itself, without relying on GB_accum_mask or GB_mask.  The mask M has
// the same size as C(I,J) and A.  M(0,0) governs how A(0,0) is assigned
// into C(I[0],J[0]).  This is how GxB_subassign operates.  For GrB_assign, the
// mask M in this function is the SubMask, constructed via SubMask=M(I,J).

// No transposed case is handled.  This function is also agnostic about the
// CSR/CSC format of C, A, and M.  The A matrix must have A->vlen == nI and
// A->vdim == nJ (except for scalar expansion, in which case A is NULL).  The
// mask M must be the same size as A, if present.

// Any or all of the C, M, and/or A matrices may be hypersparse or standard
// non-hypersparse.

// C is operated on in-place and thus cannot be aliased with the inputs A or M.

// Since the pattern of C does not change here, C->p, C->h, C->nvec, and
// C->nvec_nonempty are constant.  C->x and C->i can be modified, but only one
// entry at a time.  No entries are shifted.  C->x can be modified, and C->i
// can be changed by turning an entry into a zombie, or by bringing a zombie
// back to life, but no entry in C->i moves in position.

#define GB_FREE_WORK                                    \
{                                                       \
    GB_MATRIX_FREE (&S) ;                               \
    GB_MATRIX_FREE (&A2) ;                              \
    GB_MATRIX_FREE (&M2) ;                              \
    GB_FREE_MEMORY (I2,  ni, sizeof (GrB_Index)) ;      \
    GB_FREE_MEMORY (I2k, ni, sizeof (GrB_Index)) ;      \
    GB_FREE_MEMORY (J2,  nj, sizeof (GrB_Index)) ;      \
    GB_FREE_MEMORY (J2k, nj, sizeof (GrB_Index)) ;      \
}

#include "GB_subassign.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                     \
{                                                       \
    GB_PHIX_FREE (C) ;                                  \
    GB_FREE_WORK ;                                      \
}

GrB_Info GB_subassigner             // C(I,J)<#M> = A or accum (C (I,J), A)
(
    GrB_Matrix C,                   // input/output matrix for results
    bool C_replace,                 // C matrix descriptor
    const GrB_Matrix M_input,       // optional mask for C(I,J), unused if NULL
    const bool Mask_comp,           // mask descriptor
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),A)
    const GrB_Matrix A_input,       // input matrix (NULL for scalar expansion)
    const GrB_Index *I_input,       // list of indices
    const int64_t   ni_input,       // number of indices
    const GrB_Index *J_input,       // list of vector indices
    const int64_t   nj_input,       // number of column indices
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GB_Type_code scalar_code, // type code of scalar to expand
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix S = NULL ;
    GrB_Matrix A2 = NULL ;
    GrB_Matrix M2 = NULL ;
    GrB_Index *restrict I2  = NULL ;
    GrB_Index *restrict I2k = NULL ;
    GrB_Index *restrict J2  = NULL ;
    GrB_Index *restrict J2k = NULL ;

    GrB_Matrix A = A_input ;
    GrB_Matrix M = M_input ;
    GrB_Index *I = I_input ;
    GrB_Index *J = J_input ;
    int64_t ni = ni_input ;
    int64_t nj = nj_input ;

    ASSERT (C != NULL) ;
    ASSERT (!GB_aliased (C, M)) ;
    ASSERT (!GB_aliased (C, A)) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // subassign tolerates both zombies and pending tuples in C, but not M or A
    GB_WAIT (M) ;
    GB_WAIT (A) ;

    //--------------------------------------------------------------------------
    // check empty mask conditions
    //--------------------------------------------------------------------------

    if (M == NULL)
    {
        // the mask is empty
        if (Mask_comp)
        {
            // an empty mask is complemented
            if (!C_replace)
            { 
                // No work to do.  This the same as the GB_RETURN_IF_QUICK_MASK
                // case in other GraphBLAS functions, except here only the
                // sub-case of C_replace=false is handled.  The C_replace=true
                // sub-case needs to delete all entries in C(I,J), which is
                // handled below in GB_subassign_method0.
                return (GrB_SUCCESS) ;
            }
        }
        else
        { 
            // The mask is empty and not complemented.  In this case, C_replace
            // is effectively false.  Disable it, since it can force pending
            // tuples to be assembled.  In the comments below "C_replace
            // effectively false" means that either C_replace is false on
            // input, or the mask is empty and not complemented and thus
            // C_replace is set to false here.
            C_replace = false ;
        }
    }

    // C_replace now has its effective value: can only be true if true on
    // input and if the mask is present, or empty and complemented.  C_replace
    // is false if it is false on input, or if the mask is empty and not
    // complemented.

    ASSERT (GB_IMPLIES (M == NULL && !Mask_comp, C_replace == false)) ;

    //--------------------------------------------------------------------------
    // get the C matrix
    //--------------------------------------------------------------------------

    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    int64_t ccode = C->type->code ;

    // the matrix C may have pending tuples and/or zombies
    ASSERT (GB_PENDING_OK (C)) ; ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (scalar_code <= GB_UDT_code) ;

    //--------------------------------------------------------------------------
    // determine the length and kind of I and J, and check their properties
    //--------------------------------------------------------------------------

    int64_t nI, nJ, Icolon [3], Jcolon [3] ;
    int Ikind, Jkind ;
    GB_ijlength (I, ni, cvlen, &nI, &Ikind, Icolon) ;
    GB_ijlength (J, nj, cvdim, &nJ, &Jkind, Jcolon) ;

    // If the descriptor says that A must be transposed, it has already been
    // transposed in the caller.  Thus C(I,J), A, and M (if present) all
    // have the same size: length(I)-by-length(J)

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;
    // printf ("=========================================== I\n") ;
    GB_OK (GB_ijproperties (I, ni, nI, cvlen, &Ikind, Icolon,
                &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Context)) ;
    // printf ("=========================================== J\n") ;
    GB_OK (GB_ijproperties (J, nj, nJ, cvdim, &Jkind, Jcolon,
                &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Context)) ;

    //--------------------------------------------------------------------------
    // sort I and J and remove duplicates, if needed
    //--------------------------------------------------------------------------

    // If I or J are explicit lists, and either of are unsorted or are sorted
    // but have duplicate entries, then both I and J are sorted and their
    // duplicates are removed.  A and M are adjusted accordingly.  Removing
    // duplicates decreases the length of I and J.

    bool I_jumbled = (I_unsorted || I_has_dupl) ;
    bool J_jumbled = (J_unsorted || J_has_dupl) ;
    bool presort = I_jumbled || J_jumbled ;

    // This pre-sort of I and J is required for the parallel subassign.
    // Otherwise, multiple threads may attempt to modify the same part of C.
    // This could cause a race condition, if one thread flags a zombie at the
    // same time another thread is using that index in a binary search.  If the
    // 2nd thread finds either zombie/not-zombie, this is fine, but the
    // modification would have to be atomic.  Atomic read/write is slow, so to
    // avoid the use of atomics, the index lists I and J are sorted and all
    // duplicates are removed.

    // A side benefit of this pre-sort is that it ensures that the results of
    // GrB_assign and GxB_subassign are fully defined if I and J have
    // duplicates.  The definition of this pre-sort is given in the M-file
    // below.

    /*
        function C = subassign (C, I, J, A)
        % submatrix assignment with pre-sort of I and J; and remove duplicates

        % delete duplicates from I, keeping the last one seen
        [I2 I2k] = sort (I) ;
        Idupl = [(I2 (1:end-1) == I2 (2:end)), false] ;
        I2  = I2  (~Idupl) ;
        I2k = I2k (~Idupl) ;
        assert (isequal (I2, unique (I)))

        % delete duplicates from J, keeping the last one seen
        [J2 J2k] = sort (J) ;
        Jdupl = [(J2 (1:end-1) == J2 (2:end)), false] ;
        J2  = J2  (~Jdupl) ;
        J2k = J2k (~Jdupl) ;
        assert (isequal (J2, unique (J)))

        % do the submatrix assignment, with no duplicates in I2 or J2
        C (I2,J2) = A (I2k,J2k) ;
    */

    // With this subassign script, the result returned by GB_subassigner
    // matches the behavior in MATLAB, so the following holds:

    /*
        C2 = C ;
        C2 (I,J) = A ;
        C3 = subassign (C, I, J, A) ;
        assert (isequal (C2, C3)) ;
    */

    // That is, the pre-sort of I, J, and A has no effect on the final C, in
    // MATLAB.

    // The pre-sort itself takes additional work and memory space, but it may
    // actually improve the performance of GB_subassigner, since it makes
    // the data access of C more regular, even in the sequential case.

    if (presort)
    {

        ASSERT (Ikind == GB_LIST || Jkind == GB_LIST) ;

        if (I_jumbled)
        { 
            // sort I and remove duplicates
            // printf ("sort I and remove duplicates::::\n") ;
            ASSERT (Ikind == GB_LIST) ;
            GB_OK (GB_ijsort (I, &ni, &I2, &I2k, Context)) ;
            I = I2 ;
            // Recheck the length and properties of the new I.  This may
            // convert I to GB_ALL or GB_RANGE, after I has been sorted.
            GB_ijlength (I, ni, cvlen, &nI, &Ikind, Icolon) ;
            GB_OK (GB_ijproperties (I, ni, nI, cvlen, &Ikind, Icolon,
                &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Context)) ;
            ASSERT (! (I_unsorted || I_has_dupl)) ;
        }

        if (J_jumbled)
        { 
            // sort J and remove duplicates
            // printf ("sort J and remove duplicates::::\n") ;
            ASSERT (Jkind == GB_LIST) ;
            GB_OK (GB_ijsort (J, &nj, &J2, &J2k, Context)) ;
            J = J2 ;
            // Recheck the length and properties of the new J.  This may
            // convert J to GB_ALL or GB_RANGE, after J has been sorted.
            GB_ijlength (J, nj, cvdim, &nJ, &Jkind, Jcolon) ;
            GB_OK (GB_ijproperties (J, nj, nJ, cvdim, &Jkind, Jcolon,
                &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Context)) ;
            ASSERT (! (J_unsorted || J_has_dupl)) ;
        }

        if (!scalar_expansion)
        { 
            // A2 = A (I2k, J2k)
            // printf ("A2 = A (I2k, J2k):\n") ;
            GB_OK (GB_subref (&A2, A->is_csc, A,
                I_jumbled ? I2k : GrB_ALL, ni,
                J_jumbled ? J2k : GrB_ALL, nj, false, true, Context)) ;
            A = A2 ;
        }

        if (M != NULL)
        { 
            // M2 = M (I2k, J2k)
            // printf ("M2 = M (I2k, J2k):\n") ;
            GB_OK (GB_subref (&M2, M->is_csc, M,
                I_jumbled ? I2k : GrB_ALL, ni,
                J_jumbled ? J2k : GrB_ALL, nj, false, true, Context)) ;
            M = M2 ;
        }

        GB_FREE_MEMORY (I2k, ni, sizeof (GrB_Index)) ;
        GB_FREE_MEMORY (J2k, nj, sizeof (GrB_Index)) ;
    }

    // I and J are now sorted, with no duplicate entries.  They are either
    // GB_ALL, GB_RANGE, or GB_STRIDE, which are intrinsically sorted with no
    // duplicates, or they are explicit GB_LISTs with sorted entries and no
    // duplicates.

    ASSERT (! (I_unsorted || I_has_dupl)) ;
    ASSERT (! (J_unsorted || J_has_dupl)) ;

    //--------------------------------------------------------------------------
    // determine the type and nnz of A (from a scalar or matrix)
    //--------------------------------------------------------------------------

    // also determines if A is dense.  The scalar is always dense.

    // mn = nI * nJ; valid only if mn_ok is true.
    GrB_Index mn ;
    bool mn_ok = GB_Index_multiply (&mn, nI, nJ) ;
    bool is_dense ;     // true if A is dense (or scalar expansion)
    int64_t anz ;       // nnz(A), or mn for scalar expansion
    bool anz_ok ;       // true if anz is OK
    GrB_Type atype ;    // the type of A or the scalar

    if (scalar_expansion)
    { 
        // The input is a scalar; the matrix A is not present.  Scalar
        // expansion results in an implicit dense matrix A whose type is
        // defined by the scalar_code.
        ASSERT (A == NULL) ;
        ASSERT (scalar != NULL) ;
        anz = mn ;
        anz_ok = mn_ok ;
        is_dense = true ;
        // a run-time or compile-time user-defined scalar is assumed to have
        // the same type as C->type which is also user-defined (or else it
        // would not be compatible).  Compatibility has already been checked in
        // the caller.  The type of scalar for built-in types is determined by
        // scalar_code, instead, since it can differ from C (in which case it
        // is typecasted into C->type).  User-defined scalars cannot be
        // typecasted.
        atype = GB_code_type (scalar_code, C->type) ;
        ASSERT_OK (GB_check (atype, "atype for scalar expansion", GB0)) ;
    }
    else
    { 
        // A is an nI-by-nJ matrix, with no pending computations
        ASSERT_OK (GB_check (A, "A for subassign kernel", GB0)) ;
        ASSERT (nI == A->vlen && nJ == A->vdim) ;
        ASSERT (!GB_PENDING (A)) ;   ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT (scalar == NULL) ;
        anz = GB_NNZ (A) ;
        anz_ok = true ;
        is_dense = (mn_ok && anz == (int64_t) mn) ;
        atype = A->type ;
    }

    //--------------------------------------------------------------------------
    // check the size of the mask
    //--------------------------------------------------------------------------

    // For subassignment, the mask must be |I|-by-|J|

    if (M != NULL)
    { 
        // M can have no pending tuples nor zombies
        ASSERT_OK (GB_check (M, "M for subassign kernel", GB0)) ;
        ASSERT (!GB_PENDING (M)) ;  ASSERT (!GB_ZOMBIES (M)) ;
        ASSERT (nI == M->vlen && nJ == M->vdim) ;
    }

    //--------------------------------------------------------------------------
    // check compatibilty of prior pending tuples
    //--------------------------------------------------------------------------

    // The action: ( delete ), described below, can only delete a live
    // entry in the pattern.  It cannot delete a pending tuple; pending tuples
    // cannot become zombies.  Thus, if this call to GxB_subassign has the
    // potential for creating zombies, all prior pending tuples must be
    // assembled now.  They thus become live entries in the pattern of C, so
    // that this GxB_subassign can (potentially) turn them into zombies via
    // action: ( delete ).

    // If accum is NULL, the operation is C(I,J) = A, or C(I,J)<M> = A.
    // If A has any implicit zeros at all, or if M is present, then
    // the action: ( delete ) is possible.  This action is taken when an entry
    // is found in C but not A.  It is thus not possible to check A in advance
    // if an entry in C must be deleted.  If an entry does not appear in C but
    // appears as a pending tuple, deleting it would require a scan of all the
    // pending tuples in C.  This is costly, and simply assembling all pending
    // tuples first is faster.

    // The action: ( insert ), described below, adds additional pending tuples.
    // All pending tuples will be assembled sometime later on, using a single
    // pending operator, and thus the current accum operator must match the
    // prior pending operator.  If the operators do not match, then all prior
    // pending tuples must be assembled now, so that this GxB_subassign can
    // (potentially) insert new pending tuples whose pending operator is accum.

    // These tests are conservative because it is possible that this
    // GxB_subassign will not need to use action: ( insert ).

    // In the discussion below, let SECOND_Ctype denote the SECOND operator
    // z=f(x,y) whose ztype, xtype, and ytype matches the type of C.

    bool wait = false ;

    if (C->Pending == NULL)
    { 

        //----------------------------------------------------------------------
        // no pending tuples currently exist
        //----------------------------------------------------------------------

        // If any new pending tuples are added, their pending operator is
        // accum, or the implicit SECOND_Ctype operator if accum is NULL.
        // The type of any pending tuples will become C->type.
        // Prior zombies have no effect on this decision.

        wait = false ;

    }
    else
    {

        //----------------------------------------------------------------------
        // prior pending tuples exist: check if action: ( delete ) can occur
        //----------------------------------------------------------------------

        // action: ( delete ) can only operate on entries in the pattern by
        // turning them into zombies.  It cannot delete prior pending tuples.
        // Thus all prior pending tuples must be assembled first if
        // action: ( delete ) can occur.

        if (C_replace)
        { 
            // C_replace must use the action: ( delete )
            wait = true ;
        }
        else if (accum == NULL)
        {
            // This GxB_subassign can potentially use action: ( delete ), and
            // thus prior pending tuples must be assembled first.  However, if
            // A is completely dense and if there is no mask M, then C(I,J)=A
            // cannot delete any entries from C.

            if (M == NULL && is_dense)
            { 
                // A is a dense matrix, so entries cannot be deleted
                wait = false ;
            }
            else
            { 
                // A is sparse or M is present.
                // In this case, action: ( delete ) might occur
                wait = true ;
            }
        }

        //----------------------------------------------------------------------
        // check if pending operator is compatible
        //----------------------------------------------------------------------

        if (!wait)
        {

            // ( delete ) will not occur, but new pending tuples may be added
            // via the action: ( insert ).  Check if the accum operator is the
            // same as the prior pending operator and ensure the types are
            // the same.

            ASSERT (C->Pending != NULL) ;
            ASSERT (C->Pending->type != NULL) ;

            if (atype != C->Pending->type)
            { 
                // entries in A are copied directly into the list of pending
                // tuples for C, with no typecasting.  The type of the prior
                // pending tuples must match the type of A.  Since the types
                // do not match, prior updates must be assembled first.
                wait = true ;
            }
            else if
            (
                // the types match, now check the pending operator
                ! (
                    // the operators are the same
                    (accum == C->Pending->op)
                    // or both operators are SECOND_Ctype, implicit or explicit
                    || (GB_op_is_second (accum, C->type) &&
                        GB_op_is_second (C->Pending->op, C->type))
                  )
            )
            { 
                wait = true ;
            }
        }
    }

    if (wait)
    { 
        // Prior computations are not compatible with this assignment, so all
        // prior work must be finished.  This potentially costly.

        // delete any lingering zombies and assemble any pending tuples
        ASSERT_OK (GB_check (C, "C before wait", GB0)) ;
        GB_OK (GB_wait (C, Context)) ;
        ASSERT_OK (GB_check (C, "C after wait", GB0)) ;
    }

    ASSERT_OK_OR_NULL (GB_check (accum, "accum for assign", GB0)) ;

    //--------------------------------------------------------------------------
    // keep track of the current accum operator
    //--------------------------------------------------------------------------

    // If accum is NULL and pending tuples are added, they will be assembled
    // sometime later (not here) using the implied SECOND_Ctype operator.  This
    // GxB_subassign operation corresponds to C(I,J)=A or C(I,J)<M>=A.
    // Subsequent calls to GrB_setElement, and subsequent calls to
    // GxB_subassign with an explict SECOND_Ctype operator, may create
    // additional pending tuples and add them to the list without requiring
    // that they be assembled first.

    // If accum is non-NULL, then all prior pending tuples have the same
    // pending operator as this accum.  If that prior operator was the implicit
    // SECOND_Ctype and those pending tuples still exist, then this accum
    // operator is the explicit SECOND_ctype operator.  The implicit
    // SECOND_Ctype operator is replaced with the current accum, which is the
    // explicit SECOND_Ctype operator.

    if (C->Pending != NULL)
    {
        C->Pending->op = accum ;
    }

    //--------------------------------------------------------------------------
    // check for quicker method if accum is present and C_replace is false
    //--------------------------------------------------------------------------

    // Before allocating S, see if there is a faster method
    // that does not require S to be created.

    bool S_Extraction = true ;

    bool C_Mask_scalar = (scalar_expansion && !C_replace &&
        M != NULL && !Mask_comp) ;

    int64_t cnz = GB_NNZ (C) ;      // includes zombies but not pending tuples

    int64_t nzMask = (M == NULL) ? 0 : GB_NNZ (M) ;

    if (Mask_comp && M == NULL)
    {
        // use Method 0: C(I,J) = empty
        S_Extraction = true ;
    }
    else if (C_Mask_scalar)
    { 
        // use Method 1 or 2: C(I,J)<M> = scalar or += scalar; C_replace false
        S_Extraction = false ;
    }
    else if (accum != NULL && !C_replace)
    {
        // If accum is present and C_replace is false, then only entries in A
        // need to be examined.  Not all entries in C(I,J) and M need to be
        // examined.  As a result, computing S=C(I,J) can dominate the time and
        // memory required for the S_Extraction method.  If S_Extraction is
        // set false, then Method 3, 4, 5, or 6 will be used.
        if (nI == 1 || nJ == 1 || cnz == 0)
        { 
            // No need to form S if it has just a single row or column.  If C
            // is empty so is S, so don't bother computing it.  Do not use
            // S; uses Methods 3, 4, 5, or 6 instead.
            S_Extraction = false ;
        }
        else if (anz_ok && cnz + nzMask > anz)
        {
            // If C and M are very dense, then do not extract S
            S_Extraction = GB_subassign_select (C, nzMask, anz,
                J, nJ, Jkind, Jcolon, Context)  ;
        }
    }

    //--------------------------------------------------------------------------
    // extract the pattern: S = C(I,J) for S_Extraction method, and quick mask
    //--------------------------------------------------------------------------

    // S is a sparse int64_t matrix.  Its "values" are not numerical, but
    // indices into C.  For example, suppose 100 = I [5] and 200 = J [7].  Then
    // S(5,7) is the entry C(I(5),J(7)), and the value of S(5,7) is the
    // position in C that holds that particular entry C(100,200):
    // pC = S->x [...] gives the location of the value C->x [pC] and row index
    // 100 = C->i [pC], and pC will be between C->p [200] ... C->p [200+1]-1
    // if C is non-hypersparse.  If C is hyperparse then pC will be still
    // reside inside the vector jC, in the range C->p [k] ... C->p [k+1]-1,
    // if jC is the kth non-empty vector in the hyperlist of C.

    if (S_Extraction)
    { 

        //----------------------------------------------------------------------
        // extract symbolic structure S=C(I,J)
        //----------------------------------------------------------------------

        // TODO: the properties of I and J are already known, and thus do not
        // need to be recomputed by GB_subref.

        // S and C have the same CSR/CSC format.  S is always returned sorted,
        // in the same hypersparse form as C (unless S is empty, in which case
        // it is always returned as hypersparse). This also checks I and J.

// double t = omp_get_wtime ( ) ;
        GB_OK (GB_subref (&S, C->is_csc, C, I, ni, J, nj, true, true, Context));
// t = omp_get_wtime ( ) - t ; printf ("\nsubref %g sec\n", t) ;

        ASSERT_OK (GB_check (C, "C for subref extraction", GB0)) ;
        ASSERT_OK (GB_check (S, "S for subref extraction", GB0)) ;

        #ifdef GB_DEBUG
        GB_GET_S ;
        // this body of code explains what S contains.
        // S is nI-by-nJ where nI = length (I) and nJ = length (J)
        GBI_for_each_vector (S)
        {
            // prepare to iterate over the entries of vector S(:,jnew)
            GBI_jth_iteration (jnew, pS_start, pS_end) ;
            // S (inew,jnew) corresponds to C (iC, jC) ;
            // jC = J [j] ; or J is a colon expression
            int64_t jC = GB_ijlist (J, jnew, Jkind, Jcolon) ;
            for (int64_t pS = pS_start ; pS < pS_end ; pS++)
            {
                // S (inew,jnew) is a pointer back into C (I(inew), J(jnew))
                int64_t inew = Si [pS] ;
                ASSERT (inew >= 0 && inew < nI) ;
                // iC = I [iA] ; or I is a colon expression
                int64_t iC = GB_ijlist (I, inew, Ikind, Icolon) ;
                int64_t p = Sx [pS] ;
                ASSERT (p >= 0 && p < GB_NNZ (C)) ;
                int64_t pC_start, pC_end, pleft = 0, pright = C->nvec-1 ;
                bool found = GB_lookup (C->is_hyper, C->h, C->p,
                    &pleft, pright, jC, &pC_start, &pC_end) ;
                ASSERT (found) ;
                // If iC == I [inew] and jC == J [jnew], (or the equivaleent
                // for GB_ALL, GB_RANGE, GB_STRIDE) then A(inew,jnew) will be
                // assigned to C(iC,jC), and p = S(inew,jnew) gives the pointer
                // into C to where the entry (C(iC,jC) appears in C:
                ASSERT (pC_start <= p && p < pC_end) ;
                ASSERT (iC == GB_UNFLIP (C->i [p])) ;
            }
        }
        #endif
    }

    //==========================================================================
    // submatrix assignment C(I,J)<M> = accum (C(I,J),A): meta-algorithm
    //==========================================================================

    // There are up to 64 combinations of options, but not required to be
    // implemented, because they are either identical to another method
    // (C_replace is effectively false if M=NULL and Mask_comp=false), or they
    // are not used (the last option, whether or not S is constructed, is
    // determined here; it is not a user input).  The first 5 options are
    // determined by the input.  The table below has been collapsed to remove
    // combinations that are not used, or equivalent to other entries in the
    // table.  Only 26 unique combinations of the 64 combinations are needed.

    //      M           present or NULL
    //      Mask_comp   true or false
    //      C_replace   true or false
    //      accum       present or NULL
    //      A           scalar (x) or matrix (A)
    //      S           constructed or not 

    // C(I,J)<(M,comp,repl)> ( = , += ) (A, scalar), (with or without S);
    // I and J can be anything for any of these methods (":", colon, or list).

    // See the "No work to do..." comment above:
    // If M is not present, Mask_comp true, C_replace false: no work to do.
    // If M is not present, Mask_comp true, C_replace true: use method0
    // If M is not present, Mask_comp false:  C_replace is now false.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   -   -   -   S        7: C(I,J) = x, with S
        //  -   -   -   -   A   S        9: C(I,J) = A, with S
        //  -   -   -   +   -   -        3: C(I,J) += x
        //  -   -   -   +   -   S        8: C(I,J) += x, with S
        //  -   -   -   +   A   -        5: C(I,J) += A
        //  -   -   -   +   A   S       10: C(I,J) += A, with S

        //  -   -   r                   C_replace true on input but now false:
        //                              use methods 7, 9, 3, 8, 5, 10 above.

        //  -   c   -                   no work to do; already returned

        //  -   c   r           S        0: C(I,J) = empty, with S

        //  M   -   -   -   -   -        1: C(I,J)<M> = x
        //  M   -   -   -   A   -       15: C(I,J)<M> = A, no S (TODO)
        //  M   -   -   -   A   S      13d: C(I,J)<M> = A, with S
        //  M   -   -   +   -   -        2: C(I,J)<M> += x
        //  M   -   -   +   A   -       6b: C(I,J)<M> += A
        //  M   -   -   +   A   S      14d: C(I,J)<M> += A, with S

        //  M   -   r   -   -   S      11c: C(I,J)<M,repl> = x, with S
        //  M   -   r   -   A   S      13c: C(I,J)<M,repl> = A, with S
        //  M   -   r   +   -   S      12c: C(I,J)<M,repl> += x, with S
        //  M   -   r   +   A   S      14c: C(I,J)<M,repl> += A, with S

        //  M   c   -   -   -   S      11b: C(I,J)<!M> = x, with S
        //  M   c   -   -   A   S      13b: C(I,J)<!M> = A, with S
        //  M   c   -   +   -   -        4: C(I,J)<!M> += x
        //  M   c   -   +   -   S      12b: C(I,J)<!M> += x, with S
        //  M   c   -   +   A   -       6a: C(I,J)<!M> += A
        //  M   c   -   +   A   S      14b: C(I,J)<!M> += A, with S

        //  M   c   r   -   -   S      11a: C(I,J)<!M,repl> = x, with S
        //  M   c   r   -   A   S      13a: C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   -   S      12a: C(I,J)<!M,repl> += x, with S
        //  M   c   r   +   A   S      14a: C(I,J)<!M,repl> += A, with S

// 15: C(I,J)<M> = A, no S (TODO, write this: scan M, and binary search C and A
// 13d  // Time: TODO SUBOPTIMAL.  C(I,J)<M> = A ; using S: do only M.*(A+S)
// 14d  // Time: TODO SUBOPTIMAL.  C(I,J)<M> += A ; using S: do only M.*A.
// 6b   // Time: TODO SUBOPTIMAL.  C(I,J)<M> += A ; no S: do only M.*A.
// use methods 15 and 6b if nnz(M) << nnz(A); methods 13d and 14d otherwise

    // The following cases (on input) can be handled by two methods: with or
    // without S.  For all these cases, C_replace is false and accum is
    // present.  The choice between these pairs of methods is made via a
    // heuristic that attempts to pick the fastest method of the two options.

        // Methods 3 and 8:
        //  -   -   -   +   -   -        3: C(I,J) += x
        //  -   -   -   +   -   S        8: C(I,J) += x, with S

        // Methods 5 and 10:
        //  -   -   -   +   A   -        5: C(I,J) += A
        //  -   -   -   +   A   S       10: C(I,J) += A, with S

        // Methods 6b and 14d:
        //  M   -   -   +   A   -       6b: C(I,J)<M> += A
        //  M   -   -   +   A   S      14d: C(I,J)<M> += A, with S

        // Methods 4 and 12b:
        //  M   c   -   +   -   -        4: C(I,J)<!M> += x
        //  M   c   -   +   -   S      12b: C(I,J)<!M> += x, with S

        // Methods 6a and 14b:
        //  M   c   -   +   A   -       6a: C(I,J)<!M> += A
        //  M   c   -   +   A   S      14b: C(I,J)<!M> += A, with S

// double t = omp_get_wtime ( ) ;

    if (M == NULL && Mask_comp)
    { 

        //----------------------------------------------------------------------
        // C(I,J) = empty
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   c   r           S        0: C(I,J) = empty, with S

        // If the mask is not present but complemented, and C_replace is true,
        // then all entries in C(I,J) must be deleted by turning them into
        // zombies.  If C_replace is false then there is no work to do; this
        // case has already been handled; see "No work to do..." above.
        // S = C(I,J) is required, and has just been computed above.

        ASSERT (C_replace) ;
        ASSERT (S != NULL) ;

        // Method 0: C(I,J) = empty ; using S
        GB_OK (GB_subassign_method0 (C,
            I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
            S, Context)) ;

        // For the remaining methods, C_replace can now only be true if the
        // mask M is present.  If the mask M is not present, then C_replace is
        // now effectively false.

    }
    else if (C_Mask_scalar)
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = scalar or +=scalar, !C_replace, !Mask_comp
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   -   -        1: C(I,J)<M> = x
        //  M   -   -   +   -   -        2: C(I,J)<M> += x

        // These two methods iterate across all entries in the mask M, and for
        // each place where the M(i,j) is true, they update C(I(i),J(j)).  Not
        // all of C needs to be examined.  The accum operator may be present
        // (Method 2), or absent (Method 1).  No entries can be deleted from C.

        ASSERT (scalar_expansion) ;         // A is a scalar
        ASSERT (M != NULL && !Mask_comp) ;  // mask M present, not compl.
        ASSERT (!C_replace) ;               // C_replace is false
        ASSERT (S == NULL) ;                // S is not used

        if (accum == NULL)
        { 
            // Method 1: C(I,J)<M> = scalar ; no S
            GB_OK (GB_subassign_method1 (C,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, scalar, atype, Context)) ;
        }
        else
        { 
            // Method 2: C(I,J)<M> += scalar ; no S
            GB_OK (GB_subassign_method2 (C,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, accum, scalar, atype, Context)) ;
        }

    }
    else if (S == NULL)
    {

        //----------------------------------------------------------------------
        // assignment without S, accum present, C_replace false
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   +   -   -        3: C(I,J) += x
        //  -   -   -   +   A   -        5: C(I,J) += A
        //  M   -   -   +   A   -       6b: C(I,J)<M> += A
        //  M   c   -   +   -   -        4: C(I,J)<!M> += x
        //  M   c   -   +   A   -       6a: C(I,J)<!M> += A

        // These 4 methods can only be used if accum is present and C_replace
        // is false.  They iterate over the entries in A and ignore any part
        // of C outside of the pattern of A.  They handle any case of the mask:
        // present or NULL, and complemented or not complemented.  A can be a
        // matrix or a scalar.  No entries in C are deleted.

        ASSERT (accum != NULL) ;
        ASSERT (!C_replace) ;

        if (scalar_expansion)
        {
            if (M == NULL)
            { 
                // Method 3: C(I,J) += scalar ; no S
                GB_OK (GB_subassign_method3 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    accum, scalar, atype, Context)) ;
            }
            else
            { 
                // Method 4: C(I,J)<!M> += scalar ; no S
                // Note that Method 2 already handles C(I,J)<M> += scalar.
                ASSERT (Mask_comp) ;
                GB_OK (GB_subassign_method4 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, scalar, atype, Context)) ;
            }
        }
        else
        {
            if (M == NULL)
            { 
                // Method 5: C(I,J) += A ; no S
                GB_OK (GB_subassign_method5 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    accum, A, Context)) ;
            }
            else
            {
                if (Mask_comp)
                { 
                    // Method 6a: C(I,J)<!M> += A ; no S
                    GB_OK (GB_subassign_method6a (C,
                        I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                        M, accum, A, Context)) ;
                }
                else
                { 
                    // Method 6b: C(I,J)<M> += A ; no S
                    GB_OK (GB_subassign_method6b (C,
                        I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                        M, accum, A, Context)) ;
                }
            }
        }

    }
    else if (M == NULL)
    {

        //----------------------------------------------------------------------
        // assignment using S_Extraction method, no mask M
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S        7: C(I,J) = x, with S
        //  -   -   -   -   A   S        9: C(I,J) = A, with S
        //  -   -   -   +   -   S        8: C(I,J) += x, with S
        //  -   -   -   +   A   S       10: C(I,J) += A, with S

        // These four methods handle all cases of the submatrix assignment,
        // when the mask is not present (and not complemented).  C_replace may
        // be either true or false on input, but is now effectively false.  The
        // accum operator may or may not be present.

        // 6 cases to consider

        // [ C A 1 ]    C_A_1: C present, A present
        // [ X A 1 ]    C_A_1: C zombie, A present
        // [ . A 1 ]    D_A_1: C not present, A present
        // [ C . 1 ]    C_D_1: C present, A not present
        // [ X . 1 ]    C_D_1: C zombie, A not present
        // [ . . 1 ]           not encountered, nothing to do

        ASSERT (!Mask_comp) ;
        ASSERT (!C_replace) ;

        if (scalar_expansion)
        {
            if (accum == NULL)
            { 
                // Method 7: C(I,J) = scalar ; using S
                GB_OK (GB_subassign_method7 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    scalar, atype, S, Context)) ;
            }
            else
            { 
                // Method 8: C(I,J) += scalar ; using S
                GB_OK (GB_subassign_method8 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    accum, scalar, atype, S, Context)) ;
            }
        }
        else
        {
            if (accum == NULL)
            { 
                // Method 9: C(I,J) = A ; using S
                GB_OK (GB_subassign_method9 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    A, S, Context)) ;
            }
            else
            { 
                // Method 10: C(I,J) += A ; using S
                GB_OK (GB_subassign_method10 (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    accum, A, S, Context)) ;
            }
        }

    }
    else if (scalar_expansion)
    {

        //----------------------------------------------------------------------
        // C(I,J)<#M> = scalar or += scalar ; using S
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   r   -   -   S      11c: C(I,J)<M,repl> = x, with S
        //  M   -   r   +   -   S      12c: C(I,J)<M,repl> += x, with S
        //  M   c   -   -   -   S      11b: C(I,J)<!M> = x, with S
        //  M   c   -   +   -   S      12b: C(I,J)<!M> += x, with S
        //  M   c   r   -   -   S      11a: C(I,J)<!M,repl> = x, with S
        //  M   c   r   +   -   S      12a: C(I,J)<!M,repl> += x, with S

        // These methods handle all cases of the submatrix assignment,
        // when the mask is present (either complemented or not complemented).
        // C_replace may be either true or false.  The accum operator may or
        // may not be present.

        // 6 cases to consider
        // [ C A 1 ]    C_A_1: C present, A present, M present and = 1
        // [ X A 1 ]    C_A_1: C zombie, A present, M present and = 1
        // [ . A 1 ]    D_A_1: C not present, A present, M present and = 1
        // [ C A 0 ]    C_A_0: C present, A present, M not present or zero
        // [ X A 0 ]    C_A_0: C zombie, A present, M not present or zero
        // [ . A 0 ]           C not present, A present, M 0; nothing to do

        // The C(I,J)<M> = scalar and += scalar case is handled by
        // Methods 1 and 2.  Thus, either C_replace is true, or Mask_comp
        // is true, or both.
        ASSERT (!C_Mask_scalar) ;
        ASSERT (C_replace || Mask_comp) ;

        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            { 
                // Method 11a: C(I,J)<!M,repl> = scalar ; using S
                GB_OK (GB_subassign_method11a (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, scalar, atype, S, Context)) ;
            }
            else if (Mask_comp)
            { 
                // Method 11b: C(I,J)<!M> = scalar ; using S
                GB_OK (GB_subassign_method11b (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, scalar, atype, S, Context)) ;
            }
            else // if (C_replace)
            { 
                // Method 11c: C(I,J)<M,repl> = scalar ; using S
                ASSERT (C_replace) ;
                GB_OK (GB_subassign_method11c (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, scalar, atype, S, Context)) ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            { 
                // Method 12a: C(I,J)<!M,repl> += scalar ; using S
                GB_OK (GB_subassign_method12a (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, scalar, atype, S, Context)) ;
            }
            else if (Mask_comp)
            { 
                // Method 12b: C(I,J)<!M> += scalar ; using S
                GB_OK (GB_subassign_method12b (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, scalar, atype, S, Context)) ;
            }
            else // if (C_replace)
            { 
                // Method 12c: C(I,J)<M,repl> += scalar ; using S
                ASSERT (C_replace) ;
                GB_OK (GB_subassign_method12c (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, scalar, atype, S, Context)) ;
            }
        }

    }
    else
    {

        //------------------------------------------------------------------
        // C(I,J)<#M> = A or += A ; using S
        //------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   A   S      13d: C(I,J)<M> = A, with S
        //  M   -   -   +   A   S      14d: C(I,J)<M> += A, with S
        //  M   -   r   -   A   S      13c: C(I,J)<M,repl> = A, with S
        //  M   -   r   +   A   S      14c: C(I,J)<M,repl> += A, with S
        //  M   c   -   -   A   S      13b: C(I,J)<!M> = A, with S
        //  M   c   -   +   A   S      14b: C(I,J)<!M> += A, with S
        //  M   c   r   -   A   S      13a: C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   A   S      14a: C(I,J)<!M,repl> += A, with S

        // 12 cases to consider
        // [ C A 1 ]    C_A_1: C present, A present, M present and = 1
        // [ X A 1 ]    C_A_1: C zombie, A present, M present and = 1
        // [ . A 1 ]    D_A_1: C not present, A present, M present and = 1
        // [ C . 1 ]    C_D_1: C present, A not present, M present and = 1
        // [ X . 1 ]    C_D_1: C zombie, A not present, M present and = 1
        // [ . . 1 ]           only M=1 present, nothing to do
        // [ C A 0 ]    C_A_0: C present, A present, M not present or zero
        // [ X A 0 ]    C_A_0: C zombie, A present, M not present or zero
        // [ . A 0 ]           C not present, A present, M 0; nothing to do
        // [ C . 0 ]    C_D_0: C present, A not present, M 0
        // [ X . 0 ]    C_D_0: C zombie, A not present, M 0
        // [ . . 0 ]           M not present or zero, nothing to do

        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            { 
                // Method 13a: C(I,J)<!M,repl> = A ; using S
                GB_OK (GB_subassign_method13a (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, A, S, Context)) ;
            }
            else if (Mask_comp)
            { 
                // Method 13b: C(I,J)<!M> = A ; using S
                GB_OK (GB_subassign_method13b (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, A, S, Context)) ;
            }
            else if (C_replace)
            { 
                // Method 13c: C(I,J)<M,repl> = A ; using S
                GB_OK (GB_subassign_method13c (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, A, S, Context)) ;
            }
            else
            { 
                // Method 13d: C(I,J)<M> = A ; using S
                GB_OK (GB_subassign_method13d (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, A, S, Context)) ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            { 
                // Method 14a: C(I,J)<!M,repl> += A ; using S
                GB_OK (GB_subassign_method14a (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, A, S, Context)) ;
            }
            else if (Mask_comp)
            { 
                // Method 14b: C(I,J)<!M> += A ; using S
                GB_OK (GB_subassign_method14b (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, A, S, Context)) ;
            }
            else if (C_replace)
            { 
                // Method 14c: C(I,J)<M,repl> += A ; using S
                GB_OK (GB_subassign_method14c (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, A, S, Context)) ;
            }
            else
            { 
                // Method 14d: C(I,J)<M> += A ; using S
                GB_OK (GB_subassign_method14d (C,
                    I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                    M, accum, A, S, Context)) ;
            }
        }
    }

// t = omp_get_wtime ( ) - t ; printf ("method %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;

    //--------------------------------------------------------------------------
    // insert C in the queue if it has work to do and isn't already queued
    //--------------------------------------------------------------------------

    if (C->nzombies == 0 && C->Pending == NULL)
    { 
        // C may be in the queue from a prior assignment, but this assignemt
        // can bring zombies back to life, and the zombie count can go to zero.
        // In that case, C must be removed from the queue.  The removal does
        // nothing if C is already not in the queue.

        // FUTURE:: this might cause thrashing if lots of assigns or
        // setElements are done in parallel.  Instead, leave the matrix in the
        // queue, and allow matrices to be in the queue even if they have no
        // unfinished computations.  See also GB_setElement.

        GB_CRITICAL (GB_queue_remove (C)) ;
    }
    else
    { 
        // If C has any zombies or pending tuples, it must be in the queue.
        // The queue insert does nothing if C is already in the queue.
        GB_CRITICAL (GB_queue_insert (C)) ;
    }

    //--------------------------------------------------------------------------
    // finalize C and return result
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (C, "C(I,J) result", GB0)) ;
    return (GB_block (C, Context)) ;
}

