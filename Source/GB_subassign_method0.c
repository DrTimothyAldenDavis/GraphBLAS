//------------------------------------------------------------------------------
// GB_subassign_method0: C(I,J) = 0 ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method0
(
    GrB_Matrix C,
    // input:
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix S,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_S ;

    //--------------------------------------------------------------------------
    // Method 0: C(I,J) = 0 ; using S
    //--------------------------------------------------------------------------

    GBI_for_each_vector (S)
    {
        GBI_for_each_entry (jnew, pS, pS_end)
        {
            // S (inew,jnew) is a pointer back into C (I(inew), J(jnew))
            GB_C_S_LOOKUP ;
            if (!is_zombie)
            { 
                // ----[C - 0] replace
                // action: ( delete ): becomes a zombie
                GB_DELETE ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

