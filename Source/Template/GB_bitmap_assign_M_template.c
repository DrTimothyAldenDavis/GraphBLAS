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
    case GB_ROW_ASSIGN :
        // row assignment: C<M>(iC,J), where M is a row vector
        #include "GB_bitmap_assign_M_row_template.c"
        break ;
    case GB_COL_ASSIGN :
        // column assignment: C<M>(I,jC), where M is a column vector
        #include "GB_bitmap_assign_M_col_template.c"
        break ;
    case GB_ASSIGN :
        // GrB_assign: C<M>(I,J), where M is the same size as C
        #include "GB_bitmap_assign_M_all_template.c"
        break ;
    #ifndef GB_NO_SUBASSIGN_CASE
    case GB_SUBASSIGN :
        // GxB_subassign: C(I,J)<M>, where M is the same size as C(I,J) and A
        #include "GB_bitmap_assign_M_sub_template.c"
        break ;
    #endif
    default: ;
}

