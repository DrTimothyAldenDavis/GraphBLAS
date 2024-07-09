//------------------------------------------------------------------------------
// GB_subassigner_method: determine method for GB_subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_subassign.h"

int GB_subassigner_method           // return method to use in GB_subassigner
(
    // outputs
    bool *C_iso_out,                // true if C is iso on output
    GB_void *cout,                  // iso value of C on output
    // inputs
    const GrB_Matrix C,             // input/output matrix for results
    const bool C_replace,           // C matrix descriptor
    const GrB_Matrix M,             // optional mask for C(I,J), unused if NULL
    const bool Mask_comp,           // mask descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),A)
    const GrB_Matrix A,             // input matrix (NULL for scalar expansion)
    const int Ikind,                // I: just the kind
    const int Jkind,                // J: kind, length, and colon
    const int64_t nJ,
    const int64_t Jcolon [3],
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,
    const GrB_Type scalar_type      // type of the scalar, or NULL
)
{

    //--------------------------------------------------------------------------
    // get all properties of C, M, and A required by this function
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // empty_mask: mask not present and complemented.  This condition has
    // already handled by GB_assign_prep.
    bool empty_mask = (M == NULL) && Mask_comp ;
    ASSERT (!empty_mask) ;
    #endif

    // no_mask: mask not present and not complemented
    bool no_mask = (M == NULL) && !Mask_comp ;
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_bitmap = GB_IS_BITMAP (M) ;

    bool A_is_bitmap = GB_IS_BITMAP (A) ;
    bool A_is_full = GB_IS_FULL (A) ;
    bool A_is_sparse = GB_IS_SPARSE (A) ;
    int64_t anz = GB_nnz (A) ;

    // these properties of C are not affected by wait(C):
    bool C_is_M = (C == M) ;
    GrB_Type ctype = C->type ;

    // these properties of C can change after wait(C):
    bool C_is_empty = (GB_nnz (C) == 0 && !GB_PENDING (C) && !GB_ZOMBIES (C)) ;
    bool C_is_bitmap = GB_IS_BITMAP (C) ;
    bool C_is_full = GB_IS_FULL (C) ;

    //--------------------------------------------------------------------------
    // determine the method to use
    //--------------------------------------------------------------------------

    // whole_C_matrix is true if all of C(:,:) is being assigned to
    bool whole_C_matrix = (Ikind == GB_ALL) && (Jkind == GB_ALL) ;

    bool C_splat_scalar = false ;   // C(:,:) = x
    bool C_splat_matrix = false ;   // C(:,:) = A

    if (whole_C_matrix && no_mask && (accum == NULL))
    {
        // C(:,:) = x or A:  whole matrix assignment with no mask
        if (scalar_expansion)
        {   GB_cov[2936]++ ;
// covered (2936): 44414
            // Method 21: C(:,:) = x
            C_splat_scalar = true ;
        }
        else
        {   GB_cov[2937]++ ;
// covered (2937): 25095
            // Method 24: C(:,:) = A
            C_splat_matrix = true ;
        }
    }

    // check if C is full
    bool C_dense_update = false ;
    if (C_is_full && whole_C_matrix && no_mask && (accum != NULL)
            && (ctype == accum->ztype) && (ctype == accum->xtype))
    {   GB_cov[2938]++ ;
// covered (2938): 79784
        // C(:,:) += x or A, where C is full, no typecasting of C
        C_dense_update = true ;
    }

    // GB_assign_prep has already disabled C_replace if no mask present
    ASSERT (GB_IMPLIES (no_mask, !C_replace)) ;

    // if C is empty, C_replace is effectively false and already disabled
    ASSERT (GB_IMPLIES (C_is_empty, !C_replace)) ;

    // simple_mask: C(I,J)<M> = ... ; or C(I,J)<M> += ...
    bool simple_mask = (!C_replace && M != NULL && !Mask_comp) ;

    // C_Mask_scalar: C(I,J)<M> = scalar or += scalar
    bool C_Mask_scalar = (scalar_expansion && simple_mask) ;

    // C_Mask_matrix:  C(I,J)<M> = A or += A
    bool C_Mask_matrix = (!scalar_expansion && simple_mask) ;

    //==========================================================================
    // submatrix assignment C(I,J)<M> = accum (C(I,J),A): meta-algorithm
    //==========================================================================

    // There are many combinations of options, but not all must be implemented,
    // because they are either identical to another method (C_replace is
    // effectively false if M=NULL and Mask_comp=false), or they are not used.
    // The last option, whether or not S is constructed, is determined here; it
    // is not a user input.  The table below has been pruned to remove
    // combinations that are not used, or equivalent to other entries in the
    // table.

    // Primary options:
    //
    //      M           present or NULL
    //      Mask_comp   true or false
    //      Mask_struct structural or valued mask
    //      C_replace   true or false
    //      accum       present or NULL
    //      A           scalar (x) or matrix (A)
    //      S           constructed or not 

    // Other options handle special cases such as aliasing of input matrices
    // (methods 06d and 05f), or other properties (methods 21 to 26).

    // C(I,J)<(M,comp,repl)> ( = , += ) (A, scalar), (with or without S);
    // I and J can be anything for most of these methods (":", colon, or list).

    // See the "No work to do..." comment above:
    // If M is not present, Mask_comp true, C_replace false: no work to do.
    // If M is not present, Mask_comp true, C_replace true: use Method 00
    // If M is not present, Mask_comp false:  C_replace is now false.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   -   -       21:  C = x, no S, C anything
        //  -   -   x   -   A   -       24:  C = A, no S, C and A anything
        //  -   -   -   +   -   -       22:  C += x, no S, C full
        //  -   -   -   +   A   -       23:  C += A, no S, C full

        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   -   A   -       26:  C(:,j) = A, append column, no S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S
        //  -   -   r                        uses methods 01, 02, 03, 04
        //  -   c   -                        no work to do
        //  -   c   r           S       00:  C(I,J)<!,repl> = empty, with S

        //  M   -   -   -   -   -       05d: C<M> = x, no S, C full
        //  M   -   -   -   -   -       05e: C<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C full
        //  M   -   -   -   A   -       25:  C<M,s> = A, A full, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S
        //  M   -   -   +   A   -       08n: C(I,J)<M> += A, no S
        //  M   -   -   +   A   -       08s: C(I,J)<M> += A, with S
        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   -   A   S       10:  C(I,J)<M,repl> = A, with S
        //  M   -   r   +   -   S       11:  C(I,J)<M,repl> += x, with S
        //  M   -   r   +   A   S       12:  C(I,J)<M,repl> += A, with S

        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   -   A   S       14:  C(I,J)<!M> = A, with S
        //  M   c   -   +   -   S       15:  C(I,J)<!M> += x, with S
        //  M   c   -   +   A   S       16:  C(I,J)<!M> += A, with S
        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   -   A   S       18:  C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   -   S       19:  C(I,J)<!M,repl> += x, with S
        //  M   c   r   +   A   S       20:  C(I,J)<!M,repl> += A, with S

        //----------------------------------------------------------------------
        // FUTURE::: 8 simpler cases when I and J are ":" (S not needed):
        //----------------------------------------------------------------------

        //  M   -   -   -   A   -       06x: C(:,:)<M> = A
        //  M   -   -   +   A   -       08x: C(:,:)<M> += A
        //  M   -   r   -   A   -       10x: C(:,:)<M,repl> = A
        //  M   -   r   +   A   -       12x: C(:,:)<M,repl> += A
        //  M   c   -   -   A   -       14x: C(:,:)<!M> = A
        //  M   c   -   +   A   -       16x: C(:,:)<!M> += A
        //  M   c   r   -   A   -       18x: C(:,:)<!M,repl> = A
        //  M   c   r   +   A   -       20x: C(:,:)<!M,repl> += A

        //----------------------------------------------------------------------
        // FUTURE::: C<C,s> += x   C == M, update all values, C_replace ignored
        // FUTURE::: C<C,s> = A    C == M, A full, C_replace ignored
        //----------------------------------------------------------------------

    // For the single case C(I,J)<M>=A, two methods can be used: 06n and 06s.

    int subassign_method = -1 ;
    bool S_Extraction ;

    if (C_splat_scalar)
    {   GB_cov[2939]++ ;
// covered (2939): 44414

        //----------------------------------------------------------------------
        // C = x where x is a scalar; C becomes full
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   -   -       21:  C = x, no S, C anything

        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum == NULL) ;            // accum is not present
        ASSERT (!C_replace) ;               // C_replace is effectively false
        ASSERT (scalar_expansion) ;         // x is a scalar

        // Method 21: C = x where x is a scalar; C becomes full
        S_Extraction = false ;              // S is not used
        subassign_method = GB_SUBASSIGN_METHOD_21 ;

    }
    else if (C_splat_matrix)
    {   GB_cov[2940]++ ;
// covered (2940): 25095

        //----------------------------------------------------------------------
        // C = A
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   A   -       24:  C = A, no S, C and A anything

        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum == NULL) ;            // accum is not present
        ASSERT (!C_replace) ;               // C_replace is effectively false
        ASSERT (!scalar_expansion) ;        // A is a matrix

        // Method 24: C = A
        S_Extraction = false ;              // S is not used
        subassign_method = GB_SUBASSIGN_METHOD_24 ;

    }
    else if (C_dense_update)
    {

        //----------------------------------------------------------------------
        // C += A or x where C is full
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   +   -   -       22:  C += x, no S, C full
        //  -   -   -   +   A   -       23:  C += A, no S, C full

        ASSERT (C_is_full) ;                // C is full
        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum != NULL) ;            // accum is present
        ASSERT (!C_replace) ;               // C_replace is false

        S_Extraction = false ;              // S is not used
        if (scalar_expansion)
        {   GB_cov[2941]++ ;
// covered (2941): 4590
            // Method 22: C(:,:) += x where C is full
            subassign_method = GB_SUBASSIGN_METHOD_22 ;
        }
        else
        {   GB_cov[2942]++ ;
// covered (2942): 75194
            // Method 23: C(:,:) += A where C is full
            subassign_method = GB_SUBASSIGN_METHOD_23 ;
        }

    }
    else if (C_Mask_scalar)
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = scalar or +=scalar
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   -   -       05d: C(:,:)<M> = x, no S, C full
        //  M   -   -   -   -   -       05e: C(:,:)<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C(:,:)<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S

        ASSERT (scalar_expansion) ;         // A is a scalar
        ASSERT (M != NULL && !Mask_comp) ;  // mask M present, not compl.
        ASSERT (!C_replace) ;               // C_replace is false

        S_Extraction = false ;              // S is not used
        if (accum == NULL)
        {
            if (C_is_M && whole_C_matrix && Mask_struct)
            {   GB_cov[2943]++ ;
// covered (2943): 5080
                // Method 05f: C(:,:)<C,s> = scalar ; no S ; C == M ; M struct
                subassign_method = GB_SUBASSIGN_METHOD_05f ;
            }
            else if (C_is_empty && whole_C_matrix && Mask_struct)
            {   GB_cov[2944]++ ;
// covered (2944): 599
                // Method 05e: C(:,:)<M,s> = scalar ; no S; C empty, M struct
                subassign_method = GB_SUBASSIGN_METHOD_05e ;
            }
            else if (C_is_full && whole_C_matrix)
            {   GB_cov[2945]++ ;
// covered (2945): 235
                // Method 05d: C(:,:)<M> = scalar ; no S; C is full
                // C becomes full.
                subassign_method = GB_SUBASSIGN_METHOD_05d ;
            }
            else
            {   GB_cov[2946]++ ;
// covered (2946): 6019
                // Method 05: C(I,J)<M> = scalar ; no S
                subassign_method = GB_SUBASSIGN_METHOD_05 ;
            }
        }
        else
        {   GB_cov[2947]++ ;
// covered (2947): 6686
            // Method 07: C(I,J)<M> += scalar ; no S
            subassign_method = GB_SUBASSIGN_METHOD_07 ;
        }

    }
    else if (C_Mask_matrix)
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = A or += A
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   +   A   -       08n:  C(I,J)<M> += A, no S
        //  M   -   -   +   A   -       08s:  C(I,J)<M> += A, with S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C full
        //  M   -   x   -   A   -       25:  C<M,s> = A, A full, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S

        ASSERT (!scalar_expansion) ;        // A is a matrix
        ASSERT (M != NULL && !Mask_comp) ;  // mask M present, not compl.
        ASSERT (!C_replace) ;

        if (accum != NULL)
        {
            // Method 08n: C(I,J)<M> += A, no S.  Cannot use M or A as bitmap.
            // Method 08s: C(I,J)<M> += A, with S.  Can use M or A as bitmap.
            // if S_Extraction is true, Method 08s is used (with S).
            // Method 08n is not used if any matrix is bitmap.
            // If C is bitmap, GB_bitmap_assign_M_accum is used instead.
            S_Extraction = M_is_bitmap || A_is_bitmap ;
            if (S_Extraction)
            {   GB_cov[2948]++ ;
// covered (2948): 81948
                // Method 08s: C(I,J)<M> += A ; with S
                subassign_method = GB_SUBASSIGN_METHOD_08s ;
            }
            else
            {   GB_cov[2949]++ ;
// covered (2949): 59885
                // Method 08n: C(I,J)<M> += A ; no S
                // No matrix can be bitmap.
                subassign_method = GB_SUBASSIGN_METHOD_08n ;
            }
        }
        else
        {
            // Methods 06d, 25, 06s, or 06n: no accumulator
            if ((C_is_full || C_is_bitmap) && whole_C_matrix && M_is_A)
            {   GB_cov[2950]++ ;
// covered (2950): 4852
                // Method 06d: C(:,:)<A> = A ; no S, C full or bitmap
                S_Extraction = false ;
                subassign_method = GB_SUBASSIGN_METHOD_06d ;
                ASSERT ((C_is_full || C_is_bitmap) && whole_C_matrix && M_is_A);
            }
            else if (C_is_empty && whole_C_matrix && Mask_struct &&
                (A_is_full || A_is_bitmap))
            {   GB_cov[2951]++ ;
// covered (2951): 1036
                // Method 25: C<M,s> = A, where M is structural, A is full or
                // bitmap, and C starts out empty.  The pattern of C will be
                // the same as M, and the subassign method is extremely simple.
                // S is not used.
                S_Extraction = false ;
                subassign_method = GB_SUBASSIGN_METHOD_25 ;
            }
            else
            {   GB_cov[2952]++ ;
// covered (2952): 418096
                // C(I,J)<M> = A ;  use 06s (with S) or 06n (without S)
                // method 06s (with S) is faster when nnz (A) < nnz (M).
                // Method 06n (no S) or Method 06s (with S):
                // Method 06n is not used if M or A are bitmap.  If M and A are
                // aliased and Method 06d is not used, then 06s is used instead
                // of 06n since M==A implies nnz(A) == nnz(M).
                S_Extraction = anz < GB_nnz (M) || M_is_bitmap || A_is_bitmap ;
                if (!S_Extraction)
                {   GB_cov[2953]++ ;
// covered (2953): 51428
                    // Method 06n: C(I,J)<M> = A ; no S
                    // If M or A are bitmap, this method is not used;
                    // 06s is used instead.
                    subassign_method = GB_SUBASSIGN_METHOD_06n ;
                }
                else
                {   GB_cov[2954]++ ;
// covered (2954): 366668
                    // Method 06s: C(I,J)<M> = A ; using S
                    subassign_method = GB_SUBASSIGN_METHOD_06s ;
                }
            }
        }

    }
    else if (M == NULL)
    {

        //----------------------------------------------------------------------
        // assignment primarily using S_Extraction method, no mask M
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   -   A   -       26:  C(:,j) = A, append column, no S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S

        ASSERT (!Mask_comp) ;
        ASSERT (!C_replace) ;

        if (scalar_expansion)
        {
            S_Extraction = true ;               // S is used
            if (accum == NULL)
            {   GB_cov[2955]++ ;
// covered (2955): 6197
                // Method 01: C(I,J) = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_01 ;
            }
            else
            {   GB_cov[2956]++ ;
// covered (2956): 10130
                // Method 03: C(I,J) += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_03 ;
            }
        }
        else
        {
            if (accum == NULL)
            {   GB_cov[2957]++ ;
// covered (2957): 42215

//              printf ("Ikind %d\n", Ikind) ;
//              printf ("C hyper %d\n", GB_IS_HYPERSPARSE (C)) ;
//              printf ("A sparse %d\n", GB_IS_SPARSE (C)) ;
//              printf ("Jkind %d %d\n", Jkind, GB_LIST) ;
//              printf ("nJ %ld\n", nJ) ;
//              printf ("C iso %d\n", C->iso) ;
//              printf ("A iso %d\n", A->iso) ;

                if (Ikind == GB_ALL && GB_IS_HYPERSPARSE (C) && GB_IS_SPARSE (A)
                    && (Jkind == GB_RANGE)
                    && (nJ == 1)        // FUTURE: allow jlo:jhi
                    && (Jcolon [0] ==
                        ((C->nvec == 0) ? 0 : (C->h [C->nvec-1] + 1)))
                    && (C->type == A->type)
                    && !(A->iso)        // FUTURE: allow A to be iso
                    && !(C->iso))       // FUTURE: allow C to be iso

                {
                    // Method 26: C(:,j) = A ; append a single column.  No S.
                    // C must be hypersparse, and the last column currently in
                    // the hyperlist of C must be j-1.  A must be sparse.  No
                    // typecasting.  Method 26 is a special case of Method 02.
                    // FUTURE: extend to C(:,jlo:jhi) = A, and iso cases
//                  printf ("got method 26\n") ;
                    S_Extraction = false ;      // S not used
                    subassign_method = GB_SUBASSIGN_METHOD_26 ;
                }
                else
                {
                    // Method 02: C(I,J) = A ; using S
//                  printf ("punt to method 02\n") ;
                    S_Extraction = true ;       // S is used
                    subassign_method = GB_SUBASSIGN_METHOD_02 ;
                }
            }
            else
            {   GB_cov[2958]++ ;
// covered (2958): 151308
                // Method 04: C(I,J) += A ; using S
                S_Extraction = true ;           // S is used
                subassign_method = GB_SUBASSIGN_METHOD_04 ;
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
        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   +   -   S       11:  C(I,J)<M,repl> += x, with S
        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   +   -   S       15:  C(I,J)<!M> += x, with S
        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   +   -   S       19:  C(I,J)<!M,repl> += x, with S

        ASSERT (!C_Mask_scalar) ;
        ASSERT (C_replace || Mask_comp) ;

        S_Extraction = true ;               // S is used
        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            {   GB_cov[2959]++ ;
// covered (2959): 1907
                // Method 17: C(I,J)<!M,repl> = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_17 ;
            }
            else if (Mask_comp)
            {   GB_cov[2960]++ ;
// covered (2960): 6172
                // Method 13: C(I,J)<!M> = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_13 ;
            }
            else // if (C_replace)
            {   GB_cov[2961]++ ;
// covered (2961): 6527
                // Method 09: C(I,J)<M,repl> = scalar ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_09 ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            {   GB_cov[2962]++ ;
// covered (2962): 6624
                // Method 19: C(I,J)<!M,repl> += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_19 ;
            }
            else if (Mask_comp)
            {   GB_cov[2963]++ ;
// covered (2963): 6569
                // Method 15: C(I,J)<!M> += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_15 ;
            }
            else // if (C_replace)
            {   GB_cov[2964]++ ;
// covered (2964): 2159
                // Method 11: C(I,J)<M,repl> += scalar ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_11 ;
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
        //  M   -   r   -   A   S       10:  C(I,J)<M,repl> = A, with S
        //  M   -   r   +   A   S       12:  C(I,J)<M,repl> += A, with S
        //  M   c   -   -   A   S       14:  C(I,J)<!M> = A, with S
        //  M   c   -   +   A   S       16:  C(I,J)<!M> += A, with S
        //  M   c   r   -   A   S       18:  C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   A   S       20:  C(I,J)<!M,repl> += A, with S

        ASSERT (Mask_comp || C_replace) ;

        S_Extraction = true ;               // S is used
        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            {   GB_cov[2965]++ ;
// covered (2965): 27143
                // Method 18: C(I,J)<!M,repl> = A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_18 ;
            }
            else if (Mask_comp)
            {   GB_cov[2966]++ ;
// covered (2966): 53026
                // Method 14: C(I,J)<!M> = A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_14 ;
            }
            else // if (C_replace)
            {   GB_cov[2967]++ ;
// covered (2967): 41450
                // Method 10: C(I,J)<M,repl> = A ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_10 ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            {   GB_cov[2968]++ ;
// covered (2968): 8613
                // Method 20: C(I,J)<!M,repl> += A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_20 ;
            }
            else if (Mask_comp)
            {   GB_cov[2969]++ ;
// covered (2969): 10705
                subassign_method = GB_SUBASSIGN_METHOD_16 ;
            }
            else // if (C_replace)
            {   GB_cov[2970]++ ;
// covered (2970): 23855
                // Method 12: C(I,J)<M,repl> += A ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_12 ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // determine the iso property of C on output
    //--------------------------------------------------------------------------

    // For scalar expansion, or if A is iso on input, then C might be iso on
    // output.  Otherwise, C is always non-iso on output.  Skip this if cout or
    // C_iso_out are NULL, since that means they have already been computed.

    bool iso_check = (cout != NULL && C_iso_out != NULL) ;
    if (iso_check)
    {

        bool A_iso = scalar_expansion           // all scalars are iso
            || (A != NULL && A->iso)            // or A is iso
            || (anz == 1 && !A_is_bitmap) ;     // or A is effectively iso
        if (A_iso &&
            subassign_method != GB_SUBASSIGN_METHOD_26  /* FUTURE */)
        {

            //------------------------------------------------------------------
            // cout = tentative iso value of C on output
            //------------------------------------------------------------------

            GB_Type_code ccode = ctype->code ;
            size_t       csize = ctype->size ;
            GrB_Type atype = (A == NULL) ? scalar_type : A->type ;
            GB_Type_code acode = atype->code ;
            size_t       asize = atype->size ;

            // cout = (ctype) (scalar or A->x)
            GB_cast_scalar (cout, ccode, (scalar_expansion) ? scalar : A->x,
                acode, asize) ;
            bool c_ok = false ;
            if (C_is_empty)
            {   GB_cov[2971]++ ;
// covered (2971): 47553
                // C is empty on input; note that C->iso might also be true,
                // but this is ignored.
                c_ok = true ;
            }
            else if (C->iso)
            {   GB_cov[2972]++ ;
// covered (2972): 323
                // C is iso on input; compare cout and C->x
                c_ok = (memcmp (cout, C->x, csize) == 0) ;
            }

            //------------------------------------------------------------------
            // apply the accum, if present, and compare its result with cout
            //------------------------------------------------------------------

            bool accum_ok = false ;
            if (c_ok && accum != NULL && C->iso)
            {
                if (C_is_empty)
                {   GB_cov[2973]++ ;
// covered (2973): 76
                    // If C is empty, the accum is not applied.
                    accum_ok = true ;
                }
                else
                {   GB_cov[2974]++ ;
// covered (2974): 17
                    // C is iso and not empty; check the result of accum
                    GxB_binary_function faccum = accum->binop_function ;

                    size_t xsize = accum->xtype->size ;
                    size_t ysize = accum->ytype->size ;
                    size_t zsize = accum->ztype->size ;

                    GB_Type_code xcode = accum->xtype->code ;
                    GB_Type_code ycode = accum->ytype->code ;
                    GB_Type_code zcode = accum->ztype->code ;

                    // x = (xtype) C->x
                    GB_void x [GB_VLA(xsize)] ;
                    GB_cast_scalar (x, xcode, C->x, ccode, csize) ;

                    // y = (ytype) (scalar or A->x)
                    GB_void y [GB_VLA(ysize)] ;
                    GB_cast_scalar (y, ycode,
                        (scalar_expansion) ? scalar : A->x, acode, asize) ;

                    // z = x + y
                    GB_void z [GB_VLA(zsize)] ;
                    faccum (z, x, y) ;

                    // c = (ctype) z
                    GB_void c [GB_VLA(csize)] ;
                    GB_cast_scalar (c, ccode, z, zcode, zsize) ;

                    // compare c and cout
                    accum_ok = (memcmp (cout, c, csize) == 0) ;
                }
            }

            switch (subassign_method)
            {

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == scalar
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_01  : GB_cov[2975]++ ;    // C(I,J) = scalar
// covered (2975): 6197
                case GB_SUBASSIGN_METHOD_05  : GB_cov[2976]++ ;    // C(I,J)<M> = scalar
// covered (2976): 10398
                case GB_SUBASSIGN_METHOD_13  : GB_cov[2977]++ ;    // C(I,J)<!M> = scalar
// covered (2977): 16084
                case GB_SUBASSIGN_METHOD_05d  : GB_cov[2978]++ ;   // C(:,:)<M> = scalar ; C full
// covered (2978): 16319
                case GB_SUBASSIGN_METHOD_09  : GB_cov[2979]++ ;    // C(I,J)<M,replace> = scalar
// covered (2979): 22274
                case GB_SUBASSIGN_METHOD_17  : GB_cov[2980]++ ;    // C(I,J)<!M,replace> = scalar
// covered (2980): 23475
                    (*C_iso_out) = c_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == a
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_02  : GB_cov[2981]++ ;    // C(I,J) = A
// covered (2981): 8227
//              FUTURE: handle iso case for method 26
//              case GB_SUBASSIGN_METHOD_26  : GB_cov[2982]++ ;    // C(:,j) = A, append column
// NOT COVERED (2982):
                case GB_SUBASSIGN_METHOD_06s  : GB_cov[2983]++ ;   // C(I,J)<M> = A ; with S
// covered (2983): 73940
                case GB_SUBASSIGN_METHOD_14  : GB_cov[2984]++ ;    // C(I,J)<!M> = A
// covered (2984): 76930
                case GB_SUBASSIGN_METHOD_10  : GB_cov[2985]++ ;    // C(I,J)<M,replace> = A
// covered (2985): 78584
                case GB_SUBASSIGN_METHOD_18  : GB_cov[2986]++ ;    // C(I,J)<!M,replace> = A
// covered (2986): 79760
                case GB_SUBASSIGN_METHOD_06d  : GB_cov[2987]++ ;   // C(:,:)<A> = A ; C is full
// covered (2987): 80707
                case GB_SUBASSIGN_METHOD_06n  : GB_cov[2988]++ ;   // C(I,J)<M> = A ; no S
// covered (2988): 84496
                    (*C_iso_out) = c_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is always iso, regardless of C_in
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_21  : GB_cov[2989]++ ;    // C(:,:) = scalar
// covered (2989): 44414
                case GB_SUBASSIGN_METHOD_05e  : GB_cov[2990]++ ;   // C(:,:)<M,struct>=x ; C empty
// covered (2990): 45013
                case GB_SUBASSIGN_METHOD_05f  : GB_cov[2991]++ ;   // C(:,:)<C,struct>=scalar
// covered (2991): 50093
                    (*C_iso_out) = true ;       // scalars are always iso
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if A is iso, regardless of C_in
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_24  : GB_cov[2992]++ ;    // C = A
// covered (2992): 2007
                case GB_SUBASSIGN_METHOD_25  : GB_cov[2993]++ ;    // C(:,:)<M,str> = A ; C empty
// covered (2993): 2124
                    (*C_iso_out) = true ;       // A is iso (see above)
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == cin+scalar
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_03  : GB_cov[2994]++ ;    // C(I,J) += scalar
// covered (2994): 10130
                case GB_SUBASSIGN_METHOD_07  : GB_cov[2995]++ ;    // C(I,J)<M> += scalar
// covered (2995): 14252
                case GB_SUBASSIGN_METHOD_15  : GB_cov[2996]++ ;    // C(I,J)<!M> += scalar
// covered (2996): 20147
                case GB_SUBASSIGN_METHOD_22  : GB_cov[2997]++ ;    // C += scalar ; C is full
// covered (2997): 24737
                case GB_SUBASSIGN_METHOD_11  : GB_cov[2998]++ ;    // C(I,J)<M,replace> += scalar
// covered (2998): 26098
                case GB_SUBASSIGN_METHOD_19  : GB_cov[2999]++ ;    // C(I,J)<!M,replace> += scalar
// covered (2999): 32054
                    (*C_iso_out) = accum_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in and A iso and cin==cin+a
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_12  : GB_cov[3000]++ ;    // C(I,J)<M,replace> += A
// covered (3000): 2174
                case GB_SUBASSIGN_METHOD_20  : GB_cov[3001]++ ;    // C(I,J)<!M,replace> += A
// covered (3001): 3816
                case GB_SUBASSIGN_METHOD_04  : GB_cov[3002]++ ;    // C(I,J) += A
// covered (3002): 19271
                case GB_SUBASSIGN_METHOD_08s  : GB_cov[3003]++ ;   // C(I,J)<M> += A, with S
// covered (3003): 26375
                case GB_SUBASSIGN_METHOD_16  : GB_cov[3004]++ ;    // C(I,J)<!M> += A 
// covered (3004): 28313
                case GB_SUBASSIGN_METHOD_23  : GB_cov[3005]++ ;    // C += A ; C is full
// covered (3005): 40679
                case GB_SUBASSIGN_METHOD_08n  : GB_cov[3006]++ ;   // C(I,J)<M> += A, no S
// covered (3006): 48939
                    (*C_iso_out) = accum_ok ;
                    break ;

                default :;
            }
        }
        else
        {   GB_cov[3007]++ ;
// covered (3007): 833669
            // A is non-iso, so C is non-iso on output, and cout is not
            // computed
            (*C_iso_out) = false ;
        }
    }

    //--------------------------------------------------------------------------
    // determine if the subassign method can handle this case for bitmaps
    //--------------------------------------------------------------------------

    #define GB_USE_BITMAP_IF(condition) \
        if (condition) subassign_method = GB_SUBASSIGN_METHOD_BITMAP ;

    switch (subassign_method)
    {

        //----------------------------------------------------------------------
        // scalar assignent methods
        //----------------------------------------------------------------------

        case GB_SUBASSIGN_METHOD_01  : GB_cov[3008]++ ;    // C(I,J) = scalar
// covered (3008): 6197
        case GB_SUBASSIGN_METHOD_03  : GB_cov[3009]++ ;    // C(I,J) += scalar
// covered (3009): 16327
        case GB_SUBASSIGN_METHOD_05  : GB_cov[3010]++ ;    // C(I,J)<M> = scalar
// covered (3010): 22346
        case GB_SUBASSIGN_METHOD_07  : GB_cov[3011]++ ;    // C(I,J)<M> += scalar
// covered (3011): 29032
        case GB_SUBASSIGN_METHOD_13  : GB_cov[3012]++ ;    // C(I,J)<!M> = scalar
// covered (3012): 35204
        case GB_SUBASSIGN_METHOD_15  : GB_cov[3013]++ ;    // C(I,J)<!M> += scalar
// covered (3013): 41773
        case GB_SUBASSIGN_METHOD_21  : GB_cov[3014]++ ;    // C(:,:) = scalar
// covered (3014): 86187
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_05d  : GB_cov[3015]++ ;   // C(:,:)<M> = scalar ; C is full
// covered (3015): 235
        case GB_SUBASSIGN_METHOD_05e  : GB_cov[3016]++ ;   // C(:,:)<M,struct> = scalar ; C empty
// covered (3016): 834
        case GB_SUBASSIGN_METHOD_05f  : GB_cov[3017]++ ;   // C(:,:)<C,struct> = scalar
// covered (3017): 5914
        case GB_SUBASSIGN_METHOD_22  : GB_cov[3018]++ ;    // C += scalar ; C is full
// covered (3018): 10504
            // C and M can have any sparsity pattern, including bitmap
            break ;

        case GB_SUBASSIGN_METHOD_09  : GB_cov[3019]++ ;    // C(I,J)<M,replace> = scalar
// covered (3019): 6527
        case GB_SUBASSIGN_METHOD_11  : GB_cov[3020]++ ;    // C(I,J)<M,replace> += scalar
// covered (3020): 8686
        case GB_SUBASSIGN_METHOD_17  : GB_cov[3021]++ ;    // C(I,J)<!M,replace> = scalar
// covered (3021): 10593
        case GB_SUBASSIGN_METHOD_19  : GB_cov[3022]++ ;    // C(I,J)<!M,replace> += scalar
// covered (3022): 17217
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            break ;

        //----------------------------------------------------------------------
        // matrix assignent methods
        //----------------------------------------------------------------------

        // GB_accum_mask may use any of these methods, with I and J as GB_ALL.

        case GB_SUBASSIGN_METHOD_02  : GB_cov[3023]++ ;    // C(I,J) = A
// covered (3023): 42200
        case GB_SUBASSIGN_METHOD_06s  : GB_cov[3024]++ ;   // C(I,J)<M> = A ; with S
// covered (3024): 408868
        case GB_SUBASSIGN_METHOD_14  : GB_cov[3025]++ ;    // C(I,J)<!M> = A
// covered (3025): 461894
        case GB_SUBASSIGN_METHOD_10  : GB_cov[3026]++ ;    // C(I,J)<M,replace> = A
// covered (3026): 503344
        case GB_SUBASSIGN_METHOD_18  : GB_cov[3027]++ ;    // C(I,J)<!M,replace> = A
// covered (3027): 530487
        case GB_SUBASSIGN_METHOD_12  : GB_cov[3028]++ ;    // C(I,J)<M,replace> += A
// covered (3028): 554342
        case GB_SUBASSIGN_METHOD_20  : GB_cov[3029]++ ;    // C(I,J)<!M,replace> += A
// covered (3029): 562955
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            break ;

        case GB_SUBASSIGN_METHOD_04  : GB_cov[3030]++ ;    // C(I,J) += A
// covered (3030): 151308
        case GB_SUBASSIGN_METHOD_08s  : GB_cov[3031]++ ;   // C(I,J)<M> += A, with S
// covered (3031): 233256
        case GB_SUBASSIGN_METHOD_16  : GB_cov[3032]++ ;    // C(I,J)<!M> += A 
// covered (3032): 243961
        case GB_SUBASSIGN_METHOD_24  : GB_cov[3033]++ ;    // C = A
// covered (3033): 269056
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_06d  : GB_cov[3034]++ ;   // C(:,:)<A> = A ; C is full
// covered (3034): 4852
        case GB_SUBASSIGN_METHOD_23  : GB_cov[3035]++ ;    // C += A ; C is full
// covered (3035): 80046
            // C, M, and A can have any sparsity structure, including bitmap
            break ;

        case GB_SUBASSIGN_METHOD_25  : GB_cov[3036]++ ;    // C(:,:)<M,struct> = A ; C empty
// covered (3036): 1036
            // C, M, and A can have any sparsity structure, including bitmap,
            // but if M is bitmap or full, use bitmap assignment instead.
            GB_USE_BITMAP_IF (M_is_bitmap || GB_IS_FULL (M)) ;
            break ;

        case GB_SUBASSIGN_METHOD_06n  : GB_cov[3037]++ ;   // C(I,J)<M> = A ; no S
// covered (3037): 51428
            // If M or A are bitmap, Method 06s is used instead of 06n.
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            ASSERT (!M_is_bitmap) ;
            ASSERT (!A_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_08n  : GB_cov[3038]++ ;   // C(I,J)<M> += A, no S
// covered (3038): 59885
            // Method 08s is used instead of 08n if M or A are bitmap.
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            ASSERT (!M_is_bitmap) ;
            ASSERT (!A_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_26  : GB_cov[3039]++ ;    // C(:,j) = A, append column, no S
// covered (3039): 15
            // Method 26, C is hypersparse, A is sparse: no bitmap method used
            ASSERT (!C_is_bitmap) ;
            ASSERT (!A_is_bitmap) ;
            break ;

        // case GB_SUBASSIGN_METHOD_BITMAP:
        default :;
            subassign_method = GB_SUBASSIGN_METHOD_BITMAP ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (subassign_method) ;
}

