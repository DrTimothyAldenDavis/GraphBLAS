//------------------------------------------------------------------------------
// GB_AxB_saxpy3_sparsity: determine the sparsity structure for C<M or !M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A*B, C<M>=A*B, or
// C<!M>=A*B, based on the sparsity structures of C (on input), M, A, and B,
// and whether or not M is complemented.  Whether or not the computation can be
// done in-place depends on the the C_replace option, the accum operator, and
// the semiring.

#include "GB_AxB_saxpy3.h"

int GB_AxB_saxpy3_sparsity          // return the sparsity structure for C
(
    // output:
    bool *in_place,                 // if true, compute C in-place
    // input:
    const GrB_Matrix C_in,          // input matrix C
    const bool C_replace,           // C_replace option
    const GrB_BinaryOp accum,       // optional accum operator
    const GrB_Semiring semiring,    // semiring
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input A matrix
    const GrB_Matrix B              // input B matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    int C_sparsity ;
    int C_in_sparsity = GB_sparsity (C_in) ;

    double m = (double) A->vlen ;
    double k = (double) A->vdim ;
    double n = (double) B->vdim ;
    bool C_is_large = (m*n) > 4 * (m*k + k*n) ;

    int M_sparsity = (M == NULL) ? 0 : GB_sparsity (M) ;
    int B_sparsity = GB_sparsity (B) ;
    int A_sparsity = GB_sparsity (A) ;
    bool M_is_hyper  = (M_sparsity == GxB_HYPERSPARSE) ;
    bool M_is_sparse = (M_sparsity == GxB_SPARSE) ;

    // If no mask is present, C_replace can be ignored and is effectively
    // false.
    bool C_replace_effective = (M == NULL) ? false : C_replace ;

    (*in_place) = (C_in_sparsity == GxB_BITMAP || C_in_sparsity == GxB_FULL)
        && !C_replace_effective && (accum != NULL && accum == semiring->add)
        && !GB_aliased (C, M) && !GB_aliased (C, A) && !GB_aliased (C, B) ;

    if (*in_place)
    {

        //-----------------------------------------------------
        // C_in            +=               A     *     B
        //-----------------------------------------------------

        // bitmap                           any         any
        // full                             any         any

        //-----------------------------------------------------
        // C               <M>+=            A     *     B
        // C               <!M>+=           A     *     B
        //-----------------------------------------------------

        // bitmap           any             any         any  
        // full -> bitmap   any             any         any  

        // C_in can be modified in-place if it is bitmap or full, the C_replace
        // option is not in effect (either false, or true but is effectively
        // false because M is not present), the accum operator is present and
        // matches the semiring monoid, and C is not aliased with M, A, or B.

        // If C_in is modified in-place: If C_in is full and a mask is present,
        // C_in must be converted to bitmap first, however, and then it can be
        // modified in-place.

        // BFS: will not come here since it uses no accum operator, it uses
        // C_replace == true (with a mask), and aliases C with B.

        C_sparsity = C_in_sparsity ;
        if (M != NULL)
        {
            // if C is full, it is converted to bitmap if M is present
            C_sparsity = GxB_BITMAP ;
        }

    }
    else if (M != NULL && !Mask_comp && (M_is_hyper || M_is_sparse))
    {

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            hyper           any         any  
        // hyper            sparse          any         hyper
        // sparse           sparse          any         sparse/bitmap/full

        // The non-empty columns of C are a subset of the non-empty columns of
        // B, so in general, if B is hypersparse, so is C.

        // When M is hypersparse or sparse, and not complemented, C has the
        // same format as M, except when B is hypersparse, in which case C is
        // also hypersparse.

        // BFS: will not come here since M is bitmap/full

        if (B_sparsity == GxB_HYPERSPARSE)
        {
            C_sparsity = GxB_HYPERSPARSE ;
        }
        else
        {
            C_sparsity = M_sparsity ;
        }

    }
    else
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // hyper            .               hyper       hyper
        // hyper            .               sparse      hyper
        // hyper/bitmap     .               bitmap      hyper
        // hyper/bitmap     .               full        hyper

        // sparse           .               hyper       sparse
        // sparse           .               sparse      sparse
        // sparse/bitmap    .               bitmap      sparse
        // sparse/bitmap    .               full        sparse

        // sparse/bitmap    .               hyper       bitmap
        // sparse/bitmap    .               sparse      bitmap
        // bitmap           .               bitmap      bitmap
        // bitmap           .               full        bitmap

        // sparse/bitmap    .               hyper       full 
        // sparse/bitmap    .               sparse      full
        // bitmap           .               bitmap      full
        // full             .               full        full

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper/bitmap     any             bitmap      hyper
        // hyper/bitmap     any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    bitmap/full     hyper       full    (*)
        // sparse/bitmap    bitmap/full     sparse      full    (*)
        // bitmap           bitmap/full     bitmap      full    (*)
        // bitmap           bitmap/full     full        full    (*)

        // (*): if M hyper/sparse, then C is hyper/sparse; see above

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper/bitmap     any             bitmap      hyper
        // hyper/bitmap     any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    any             hyper       full 
        // sparse/bitmap    any             sparse      full
        // bitmap           any             bitmap      full
        // bitmap           any             full        full

        // If M is complemented, or not complemented and bitmap/full, then C
        // has the same sparsity as listed above, except when A and B are both
        // full.

        // For the cases where C is labelled as hyper/bitmap or sparse/bitmap:
        // Let A by m-by-k, let B by k-by-n, then C is m-by-n.  If m*n is much
        // larger than (m*k + k*n), then always construct C as sparse/hyper,
        // not bitmap.

        // BFS: C<!M>=A*B, no accum, M is complemented, not structural,
        // full (or bitmap in the future).  So C is not computed in-place.
        // Also, C is aliased with B.

        switch (B_sparsity)
        {
            case GxB_HYPERSPARSE :
            case GxB_SPARSE :
                switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE :
                    case GxB_SPARSE :
                        C_sparsity = B_sparsity ;
                        break ;
                    case GxB_BITMAP :
                    case GxB_FULL :
                        C_sparsity = C_is_large ? B_sparsity : GxB_BITMAP ;
                        break ;
                    default: ;
                }
                break ;
            case GxB_BITMAP :
                // BFS comes here: B is bitmap
            case GxB_FULL : switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE :
                    case GxB_SPARSE :
                        // BFS comes here: A is sparse, and C is not huge
                        C_sparsity = C_is_large ? GxB_SPARSE : GxB_BITMAP ;
                        break ;
                    case GxB_BITMAP :
                        C_sparsity = GxB_BITMAP ;
                        break ;
                    case GxB_FULL :
                        C_sparsity = (M == NULL) ? GxB_FULL : GxB_BITMAP ;
                        break ;
                    default: ;
                }
                break ;
            default: ;
        }
    }

    return (C_sparsity) ;
}

