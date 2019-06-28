//------------------------------------------------------------------------------
// GB_assign.h: definitions for GB_assign and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

void GB_assign_zombie1
(
    GrB_Matrix C,
    const int64_t j,
    GB_Context Context
) ;

void GB_assign_zombie2
(
    GrB_Matrix C,
    const int64_t i,
    GB_Context Context
) ;

void GB_assign_zombie3
(
    GrB_Matrix Z,
    const GrB_Matrix M,
    const bool Mask_comp,
    const int64_t j,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    GB_Context Context
) ;

void GB_assign_zombie4
(
    GrB_Matrix Z,
    const GrB_Matrix M,
    const bool Mask_comp,
    const int64_t i,
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Context Context
) ;

void GB_assign_zombie5
(
    GrB_Matrix Z,
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Context Context
) ;

