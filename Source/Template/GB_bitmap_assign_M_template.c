//------------------------------------------------------------------------------
// GB_bitmap_assign_M_template: traverse over M for bitmap assignment into C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This template traverses over all the entries of the mask matrix M, and
// operates on C(i,j) if the mask M(i,j) == 1, via the GB_MASK_WORK macro,
// where C(i,j) is at Cx [pC] and Cb [pC].  M is hypersparse or sparse.

// GB_ek_slice has alreadly sliced M for parallel work.  The tasks are held in
// pstart_Mslice, kfirst_Mslice, klast_Mslice, mtasks, and the work is done
// by mthreads threads.

// The work done by this kernel is independent of Mask_comp; both M and !M
// do the same work by scattering their entries into the C bitmap.

ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;

switch (assign_kind)
{

    //--------------------------------------------------------------------------
    // row assignment: C<M'>(iC,:), M is a column vector
    //--------------------------------------------------------------------------

    case GB_ROW_ASSIGN :
    {
        #include "GB_bitmap_assign_M_row_template.c"
    }
    break ;

    //--------------------------------------------------------------------------
    // column assignment: C<M>(:,jC), M is a column vector
    //--------------------------------------------------------------------------

    case GB_COL_ASSIGN :
    {
        #include "GB_bitmap_assign_M_col_template.c"
    }
    break ;

    //--------------------------------------------------------------------------
    // GrB_assign: C<M>(I,J), M is the same size as C
    //--------------------------------------------------------------------------

    case GB_ASSIGN :
    {
        #include "GB_bitmap_assign_M_all_template.c"
    }
    break ;

    //--------------------------------------------------------------------------
    // GxB_subassign: C(I,J)<M>, M is the same size as C(I,J)
    //--------------------------------------------------------------------------

    case GB_SUBASSIGN :
    {
        #include "GB_bitmap_assign_M_sub_template.c"
    }
    break ;

    default: ;
}

#undef GB_MASK_WORK

