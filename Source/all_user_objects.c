//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Source/all_user_objects.c
//------------------------------------------------------------------------------

// This file is constructed automatically by cmake and m4 when GraphBLAS is
// compiled, from the Config/user_def*.m4 and *.m4 files in User/.  Do not edit
// this file directly.  It contains references to internally-defined functions
// and objects inside GraphBLAS, which are not user-callable.

#include "GB.h"

//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def1.m4: define user-defined objects
//------------------------------------------------------------------------------


















//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_terminal.m4: example user built-in objects
//------------------------------------------------------------------------------

// user-defined Boolean semiring.  This is just for testing.  The semiring
// is identical to GxB_LOR_LAND_BOOL, and the monoid is identical to
// GxB_LOR_BOOL_MONOID.  The only difference is that these objects are
// user-defined.

#ifdef GxB_USER_INCLUDE

    #define MY_BOOL

#endif

// The LOR monoid, with identity = false and terminal = true

    #define GB_DEF_My_LOR_add GB_DEF_GrB_LOR_function
    #define GB_DEF_My_LOR_zsize sizeof (GB_DEF_GrB_LOR_ztype)
    #define GB_DEF_My_LOR_is_user_terminal
    GB_DEF_GrB_LOR_ztype GB_DEF_My_LOR_identity = false ;
    GB_DEF_GrB_LOR_ztype GB_DEF_My_LOR_user_terminal = true ;
    struct GB_Monoid_opaque GB_opaque_My_LOR =
    {
        GB_MAGIC,                   // object is defined
        & GB_opaque_GrB_LOR,             // binary operator
        & GB_DEF_My_LOR_identity,       // identity value
        GB_DEF_My_LOR_zsize,            // identity and terminal size
        GB_USER_COMPILED,           // user-defined at compile-time
        & GB_DEF_My_LOR_user_terminal   // terminal value
    } ;
    GrB_Monoid My_LOR = & GB_opaque_My_LOR ;

// The LOR_LAND semiring
 
    #define GB_AgusB    GB_AxB_user_gus_My_LOR_LAND
    #define GB_AdotB    GB_AxB_user_dot_My_LOR_LAND
    #define GB_AheapB   GB_AxB_user_heap_My_LOR_LAND
    #define GB_identity    GB_DEF_My_LOR_identity
    #define GB_ADD(z,y)    GB_DEF_My_LOR_add (&(z), &(z), &(y))
    #if defined ( GB_DEF_My_LOR_is_user_terminal )
        #define GB_terminal if (memcmp (&cij, &GB_DEF_My_LOR_user_terminal, GB_DEF_My_LOR_zsize) == 0) break ;
    #elif defined ( GB_DEF_My_LOR_terminal )
        #define GB_terminal if (cij == GB_DEF_My_LOR_terminal) break ;
    #else
        #define GB_terminal ;
    #endif
    #define GB_MULT(z,x,y) GB_DEF_GrB_LAND_function (&(z), &(x), &(y))
    #define GB_ztype       GB_DEF_GrB_LAND_ztype
    #define GB_xtype       GB_DEF_GrB_LAND_xtype
    #define GB_ytype       GB_DEF_GrB_LAND_ytype
    #define GB_handle_flipxy 1
    #undef GBCOMPACT
    #include "GB_AxB.c"
    #undef GB_identity
    #undef GB_terminal
    #undef GB_ADD
    #undef GB_xtype
    #undef GB_ytype
    #undef GB_ztype
    #undef GB_MULT
    #undef GB_AgusB
    #undef GB_AdotB
    #undef GB_AheapB
    struct GB_Semiring_opaque GB_opaque_My_LOR_LAND =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_My_LOR,     // add monoid
        & GB_opaque_GrB_LAND,     // multiply operator
        GB_USER_COMPILED    // user-defined at compile-time
    } ;
    GrB_Semiring My_LOR_LAND = & GB_opaque_My_LOR_LAND ;


//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def2.m4: code to call user semirings
//------------------------------------------------------------------------------

GrB_Info GB_AxB_user
(
    const GrB_Desc_Value GB_AxB_method,
    const GrB_Semiring GB_s,

    GrB_Matrix *GB_Chandle,
    const GrB_Matrix GB_M,
    const GrB_Matrix GB_A,
    const GrB_Matrix GB_B,
    bool GB_flipxy,

    // for dot method only:
    const bool GB_mask_comp,

    // for heap method only:
    int64_t *restrict GB_List,
    GB_pointer_pair *restrict GB_pA_pair,
    GB_Element *restrict GB_Heap,
    const int64_t GB_bjnz_max,

    // for Gustavson method only:
    GB_Sauna GB_C_Sauna
)
{
    GrB_Info GB_info = GrB_SUCCESS ;
    if (0)
    {
        ;
    }
    else if (GB_s == My_LOR_LAND)
    {
        if (GB_AxB_method == GxB_AxB_GUSTAVSON)
        { 
            GB_info = GB_AxB_user_gus_My_LOR_LAND
                (*GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy, GB_C_Sauna) ;
        }
        else if (GB_AxB_method == GxB_AxB_DOT)
        { 
            GB_info = GB_AxB_user_dot_My_LOR_LAND
                (GB_Chandle, GB_M, GB_mask_comp, GB_A, GB_B, GB_flipxy) ;
        }
        else // (GB_AxB_method == GxB_AxB_HEAP)
        { 
            GB_info = GB_AxB_user_heap_My_LOR_LAND
                (GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy,
                GB_List, GB_pA_pair, GB_Heap, GB_bjnz_max) ;
        }
    } 
    return (GB_info) ;
}

