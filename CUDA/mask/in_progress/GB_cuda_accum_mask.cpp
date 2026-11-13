//------------------------------------------------------------------------------
// GB_cuda_accum_mask: C<M>=accum(C,T) or C<M>=T
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

    // If the mask M is present (valued or structural), the following table
    // can be computed.  Let S be a matrix initially empty.  The matrix T
    // has been computed by the caller, and may already have exploited the
    // mask, so any computation of T-M or T∩M can be skipped in that case.
    // In the tables below, C and Z always have the same type.

    // The table uses 4 different basic operations:

        // (1) The complemented mask:
        // S=R-M means S<!M>=R for a valued or structural mask M, where
        // any entry R(i,j) with M(i,j) "false" is typecasted into S(i,j).
        // In this case, "false" means M(i,j) is not present in a
        // structural mask.  For a valued matrix, it means M(i,j) is either
        // not present, or present but its value is false.


static GrB_Info GB_cuda_R_mask_M  // S=R-M or S=R∩M where M is a mask matrix
(
    GrB_Matrix *Shandle,    // output; NULL handle on input
    // FIXME: add stype, for typecasting?
    GrB_Matrix R,
    GrB_Matrix M,
    const bool Mask_comp,   // if true: S=R-M, else: S=R∩M
    const bool Mask_struct,
    const int device,
    cudaStream_t stream,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Matrix S = NULL ;
    (*Shandle) = NULL ;

    //--------------------------------------------------------------------------
    // get the iso value of R
    //--------------------------------------------------------------------------

    GB_void rscalar [GB_VLA(rsize)] ;
    if (R->iso)
    {
        memcpy (rscalar, R->x, rsize) ;
    }

    //--------------------------------------------------------------------------
    // define the header of S
    //--------------------------------------------------------------------------

    uint64_t rnz ;
    if (Mask_comp)
    {
        // upper bound on nvals (S) for S=R-M
        rnz = GB_nnz_held (R) ;
    }
    else
    {
        // upper bound on nvals (S) for S=R∩M
        rnz = std::min (GB_nnz_held (R), GB_nnz_held (M)) ;
    }

    bool Sp_is_32, Sj_is_32, Si_is_32 ;
    GB_determine_pji_is_32 (&Sp_is_32, &Sj_is_32, &Si_is_32,
        GxB_HYPERSPARSE, rnz, R->vlen, R->vdim, Werk) ;

    // initialize the header of S, with no phybix content,
    // and initialize the type and dimension of S.
    GB_OK (GB_new (&S, // hyper, new header
        R->type, R->vlen, R->vdim, GB_ph_null, R->is_csc,
        GxB_HYPERSPARSE, GB_Global_hyper_switch_get ( ), 0,
        Sp_is_32, Sj_is_32, Si_is_32, data_arena, data_arena)) ;

    //--------------------------------------------------------------------------
    // S=R-M on CUDA, in the JIT
    //--------------------------------------------------------------------------

    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (rnz, GB_CUDA_ADD_CHUNKSIZE0) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    GBURBLE ("(cuda R-M, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    // S<#M> = R
    struct GB_BinaryOp_opaque op_header ;
    GrB_BinaryOp first = GB_binop_first (R->type, &op_header) ;
    ASSERT_BINARYOP_OK (first, "first_atype for S<*M>=R", GB0) ;
    GB_OK (GB_cuda_mask_assign_jit (S, R, M, Mask_struct, Mask_comp,
        first, device, stream, gridsz)) ;

    if (R->iso)
    {
        // assign the iso-value of S, if R is iso
        memcpy (S->x, R->x, R->size) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*Shandle) = S ;
}


        // (2) The non-complemented mask:
        // S=R∩M means S<M>=R for a valued or structural mask M,
        // any entry R(i,j) with M(i,j) "true" is typecasted into S(i,j).
        // In this case, "true" means M(i,j) is present in a structural
        // mask.  For a valued matrix, it means M(i,j) is present with a
        // value of true.

        // (3) Applying a binary operator in an eWiseAdd sense:
        // S=A+B computes A+B where "+" is the accum operator, and then
        // typecasts its results to the requested type of S, computing
        // S = (stype) (A+B).

        // (4) A set union, like eWiseAdd, but where the intersection is
        // known to be empty:
        // S=A∪B computes the union of A and B, without the need for an
        // operator since A and B are always disjoint.

    //-----------------------------------------------------------------
    // M is present:
    //-----------------------------------------------------------------
    // Repl   Accum  Compl   C<M>+=T formula
    // ----   -----  -----   ---------------
    // TRUE   TRUE   TRUE    C_out = (C+T) - M          or (C-M)+(T-M)
    // TRUE   TRUE   FALSE   C_out = (C ∩ M) + (T ∩ M)
    // TRUE   FALSE  TRUE    C_out = (ctype) (T - M)            C_in ignored
    // TRUE   FALSE  FALSE   C_out = (ctype) (T ∩ M)            C_in ignored
    // FALSE  TRUE   TRUE    C_out = C + (T - M)
    // FALSE  TRUE   FALSE   C_out = C + (T ∩ M)
    // FALSE  FALSE  TRUE    C_out = (C ∩ M) ∪ (T - M)
    // FALSE  FALSE  FALSE   C_out = (C - M) ∪ (T ∩ M)
    //-----------------------------------------------------------------

    // If there is no mask matrix M, then it is implicitly a dense matrix
    // with all entries present and true.  The structural/valued option for
    // the mask has no effect on this implicit mask matrix.  This implicit
    // mask M may be complemented, in which case no work is needed (see the
    // GB_RETURN_IF_QUICK_MASK macro) and this method is not called.  Thus,
    // the complemented empty mask case is not needed here.  Otherwise, if
    // the mask is not complemented, the replace option has no effect, so
    // the following table can ignore the C_replace option as well.

    //-----------------------------------------------------------------
    // M is not present:
    //-----------------------------------------------------------------
    // Repl   Accum  Compl   C<no mask M>+=T formula
    // ----   -----  -----   ---------------
    // ignore TRUE   FALSE   C_out = C + T
    // ignore FALSE  FALSE   C_out = T                  C_in ignored
    //-----------------------------------------------------------------

GrB_Info GB_cuda_accum_mask     // C<M> = accum (C,T) using CUDA
(
    GrB_Matrix C,               // input/output matrix for results
    const GrB_Matrix M,         // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,   // optional accum for Z=accum(C,results)
    GrB_Matrix *Thandle,        // results of computation, freed when done
    const bool C_replace,       // if true, clear C first
    const bool Mask_comp,       // if true, complement the mask
    const bool Mask_struct,     // if true, use the only structure of M
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C_out = NULL ;
    GrB_Matrix T = *Thandle ;

    int data_arena = C->data_arena ;
    int device = data_arena - GxB_NARENAS ;
    uint64_t mem = GB_mem (data_arena, 0) ;

    //--------------------------------------------------------------------------
    // finish any pending work on C, M, or T
    //--------------------------------------------------------------------------

    // if C_in_ignored is true, the input value of C is ignored
    bool C_in_ignored = (accum == NULL) && (M == NULL || C_replace) ;

    if (C_in_ignore)
    {
        GB_OK (GB_clear (C, Werk)) ;
    }
    else
    {
        GB_MATRIX_WAIT (C) ;
    }
    GB_MATRIX_WAIT (M) ;
    GB_MATRIX_WAIT (T) ;

    //--------------------------------------------------------------------------
    // ensure all matrices are sparse/hypersparse
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (C) || GB_IS_FULL (C)
        || (M != NULL && (GB_IS_BITMAP (M) || GB_IS_FULL (M)))
        || GB_IS_BITMAP (T) || GB_IS_FULL (T))
    {
        // FIXME: the CUDA kernels do not yet handle the case when C, M, and/or
        // T are bitmap/full.  Need to add these cases here.
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // handle this case on CUDA
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;
    GB_OK (GB_wait_arenas (C)) ;
    GB_OK (GB_wait_arenas (M)) ;
    GB_OK (GB_wait_arenas (T)) ;

    if (M != NULL)
    {

        //----------------------------------------------------------------------
        // mask is present: C<M>=accum(C,T) or C<M>=T
        //----------------------------------------------------------------------

        // S2 = (T - M) or (T ∩ M), or S2 = T if already done
        if (mask_applied)
        {
            // T already constructed using the mask
            // FIXME: but S2 might need to be typecasted below
            S2 = T ; (*Thandle) = NULL ;
        }
        else
        {
            // S2<*M> = T, computed as S2 = (T - M) for a complemented mask,
            // or (T ∩ M) for a non-complemented mask.
            // FIXME: need typecasting for C_out = T-M case below; others
            // must create S2 as T->type
            GB_OK (GB_cuda_R_mask_M (&S2, T, M, Mask_comp, Mask_struct,
                device, stream, Werk)) ;
        }

        if (C_replace)
        {
            if (accum != NULL)
            {
                //--------------------------------------------------------------
                // Repl   Accum  Compl   C<M>+=T formula
                // TRUE   TRUE   TRUE    C_out = (C+T) - M, or (C-M)+(T-M)
                // TRUE   TRUE   FALSE   C_out = (C ∩ M) + (T ∩ M)
                //--------------------------------------------------------------

                // S1 = (C - M) or (C ∩ M)
                GB_OK (GB_cuda_R_mask_M (&S1, C, M, Mask_comp, Mask_struct,
                    device, stream, Werk)) ;
                // C_out = (ctype) (S1 + S2)
                GB_OK (GB_cuda_A_plus_B (&C_out, C->type, S1, S2, accum,
                    device, stream, Werk)) ;

            }
            else
            {

                //--------------------------------------------------------------
                // Repl   Accum  Compl   C<M>+=T formula
                // TRUE   FALSE  TRUE    C_out = (T - M), C_in ignored
                // TRUE   FALSE  FALSE   C_out = (T ∩ M), C_in ignored
                //--------------------------------------------------------------

                // C_out = T-M or (T ∩ M), or C_out = T if already done
                // FIXME: need S2 in ctype here
                C_out = S2 ; (*Thandle) = NULL :
            }

        }
        else
        {

            if (accum != NULL)
            {

                //--------------------------------------------------------------
                // Repl   Accum  Compl   C<M>+=T formula
                // FALSE  TRUE   TRUE    C_out = C + (T - M)
                // FALSE  TRUE   FALSE   C_out = C + (T ∩ M)
                //--------------------------------------------------------------

                // C_out = (ctype) (C + S2)
                GB_OK (GB_cuda_A_plus_B (&C_out, C->type, C, S2, accum,
                    device, stream, Werk)) ;

            }
            else
            {

                //--------------------------------------------------------------
                // Repl   Accum  Compl   C<M>+=T formula
                // FALSE  FALSE  TRUE    C_out = (C ∩ M) ∪ (T - M)
                // FALSE  FALSE  FALSE   C_out = (C - M) ∪ (T ∩ M)
                //--------------------------------------------------------------

                // S1 = (C ∩ M) or (C - M)
                GB_OK (GB_cuda_R_mask_M (&S1, C, M, !Mask_comp, Mask_struct,
                    device, stream, Werk)) ;
                // C_out = (ctype) (S1 + S2), no operator; if op is NULL,
                // the method should create the FIRST_ctype operator as a
                // placeholder, which isn't actually used.
                GB_OK (GB_cuda_A_plus_B (&C_out, C->type, S1, S2, NULL,
                    device, stream, Werk)) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // no mask present: C=accum(C,T) or C=T
        //----------------------------------------------------------------------

        ASSERT (!Mask_comp) ;

        if (accum != NULL)
        {
            //------------------------------------------------------------------
            // Repl   Accum  Compl   C<M>+=T formula
            // ignore TRUE   FALSE   C_out = C + T
            //------------------------------------------------------------------

            // C_out = (ctype) (C + T)
            GB_OK (GB_cuda_A_plus_B (&C_out, C->type, C, T, accum,
                device, stream, Werk)) ;

        }
        else
        {
            //------------------------------------------------------------------
            // Repl   Accum  Compl   C<M>+=T formula
            // ignore FALSE  FALSE   C_out = T, C_in ignored
            //------------------------------------------------------------------

            // FIXME: use GB_transplant here, need CUDA memcpy and cast_matrix

        }
    }

}
