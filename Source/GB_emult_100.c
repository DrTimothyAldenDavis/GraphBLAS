//------------------------------------------------------------------------------
// GB_emult_100: C<M>= A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M>= A.*B, M sparse/hyper, A and B bitmap/full.  C has the same sparsity
// structure as M, and its pattern is a subset of M.

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          bitmap  (method: 100)
            //      sparse  sparse      bitmap          full    (method: 100)
            //      sparse  sparse      full            bitmap  (method: 100)

// TODO: this function can also do eWiseAdd, just as easily.
// Just change the "&&" to "||" in the GB_emult_100_template. 
// It can also handle the case when both A and B are full.

#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_unused.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"
#endif

#define GB_FREE_WORK                                                    \
{                                                                       \
    GB_FREE (Wfirst) ;                                                  \
    GB_FREE (Wlast) ;                                                   \
    GB_FREE (Cp_kfirst) ;                                               \
    GB_ek_slice_free (&pstart_Mslice, &kfirst_Mslice, &klast_Mslice) ;  \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORK ;              \
    GB_Matrix_free (&C) ;       \
}

GrB_Info GB_emult_100       // C<M>=A.*B, M sparse/hyper, A and B bitmap/full
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // sparse/hyper, not NULL
    const bool Mask_struct, // if true, use the only structure of M
    const GrB_Matrix A,     // input A matrix (bitmap/full)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    GrB_Matrix C = NULL ;
    (*Chandle) = NULL ;

    ASSERT_MATRIX_OK (M, "M for emult_100", GB0) ;
    ASSERT_MATRIX_OK (A, "A for emult_100", GB0) ;
    ASSERT_MATRIX_OK (B, "B for emult_100", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for emult_100", GB0) ;

    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A) || GB_as_if_full (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B) || GB_as_if_full (B)) ;

    int C_sparsity = GB_sparsity (M) ;

    GBURBLE ("emult_100:(%s<%s>=%s.*%s) ",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

//  printf ("emult_sbb:(%s<%s>=%s.*%s)\n",
//      GB_sparsity_char (C_sparsity),
//      GB_sparsity_char_matrix (M),
//      GB_sparsity_char_matrix (A),
//      GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Wfirst = NULL ;
    int64_t *GB_RESTRICT Wlast = NULL ;
    int64_t *GB_RESTRICT Cp_kfirst = NULL ;
    int64_t *GB_RESTRICT pstart_Mslice = NULL ;
    int64_t *GB_RESTRICT kfirst_Mslice = NULL ;
    int64_t *GB_RESTRICT klast_Mslice  = NULL ;

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Mp = M->p ;
    const int64_t *GB_RESTRICT Mh = M->h ;
    const int64_t *GB_RESTRICT Mi = M->i ;
    const GB_void *GB_RESTRICT Mx = (Mask_struct) ? NULL : M->x ;
    const int64_t vlen = M->vlen ;
    const int64_t vdim = M->vdim ;
    const int64_t nvec = M->nvec ;
    const int64_t mnz = GB_NNZ (M) ;
    const size_t  msize = M->type->size ;

    const int8_t *GB_RESTRICT Ab = A->b ;
    const int8_t *GB_RESTRICT Bb = B->b ;

    //--------------------------------------------------------------------------
    // allocate C->p and C->h
    //--------------------------------------------------------------------------

    GB_OK (GB_new (&C,      // sparse or hyper (same as M), new header
        ctype, vlen, vdim, GB_Ap_calloc, C_is_csc,
        C_sparsity, M->hyper_switch, nvec, Context)) ;
    int64_t *GB_RESTRICT Cp = C->p ;

    //--------------------------------------------------------------------------
    // slice the mask matrix M
    //--------------------------------------------------------------------------

    int M_nthreads, M_ntasks ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    GB_SLICE_MATRIX (M, 8) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // TODO: use one malloc
    Wfirst = GB_MALLOC (M_ntasks, int64_t) ;
    Wlast  = GB_MALLOC (M_ntasks, int64_t) ;
    Cp_kfirst = GB_MALLOC (M_ntasks, int64_t) ;
    if (Wfirst == NULL || Wlast  == NULL || Cp_kfirst == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count entries in C
    //--------------------------------------------------------------------------

    // This phase is very similar to GB_select_phase1 (GB_ENTRY_SELECTOR).

    int tid ;
    #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < M_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Mslice [tid] ;
        int64_t klast  = klast_Mslice  [tid] ;
        Wfirst [tid] = 0 ;
        Wlast  [tid] = 0 ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // count the entries in C(:,j)
            int64_t j = GBH (Mh, k) ;
            int64_t pstart = j * vlen ;     // start of A(:,j) and B(:,j)
            int64_t pM, pM_end ;
            GB_get_pA (&pM, &pM_end, tid, k,
                kfirst, klast, pstart_Mslice, Mp, vlen) ;
            int64_t cjnz = 0 ;
            for ( ; pM < pM_end ; pM++)
            { 
                bool mij = GB_mcast (Mx, pM, msize) ;
                if (mij)
                {
                    int64_t i = Mi [pM] ;
                    cjnz +=
                        (GBB (Ab, pstart + i)
                        &&      // TODO: for GB_add, use || instead
                        GBB (Bb, pstart + i)) ;
                }
            }
            if (k == kfirst)
            { 
                Wfirst [tid] = cjnz ;
            }
            else if (k == klast)
            { 
                Wlast [tid] = cjnz ;
            }
            else
            { 
                Cp [k] = cjnz ; 
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize Cp, cumulative sum of Cp and compute Cp_kfirst
    //--------------------------------------------------------------------------

    GB_ek_slice_merge1 (Cp, Wfirst, Wlast, kfirst_Mslice, klast_Mslice,
        M_ntasks) ;
    GB_ek_slice_merge2 (&(C->nvec_nonempty), Cp_kfirst, Cp, nvec,
        Wfirst, Wlast, kfirst_Mslice, klast_Mslice, M_ntasks, M_nthreads) ;

    //--------------------------------------------------------------------------
    // allocate C->i and C->x
    //--------------------------------------------------------------------------

    int64_t cnz = Cp [nvec] ;
    GB_OK (GB_bix_alloc (C, cnz, false, false, true, true, Context)) ;

    //--------------------------------------------------------------------------
    // copy pattern into C
    //--------------------------------------------------------------------------

    // TODO: could make these components of C shallow instead

    if (GB_IS_HYPERSPARSE (M))
    {
        // copy M->h into C->h
        GB_memcpy (C->h, Mh, nvec * sizeof (int64_t), M_nthreads) ;
    }

    C->nvec = nvec ;
    C->jumbled = M->jumbled ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_first  = (opcode == GB_FIRST_opcode) ;
    bool op_is_second = (opcode == GB_SECOND_opcode) ;
    bool op_is_pair   = (opcode == GB_PAIR_opcode) ;
    GB_Type_code ccode = ctype->code ;

    //--------------------------------------------------------------------------
    // check if the values of A and/or B are ignored
    //--------------------------------------------------------------------------

    // With C = ewisemult (A,B), only the intersection of A and B is used.
    // If op is SECOND or PAIR, the values of A are never accessed.
    // If op is FIRST  or PAIR, the values of B are never accessed.
    // If op is PAIR, the values of A and B are never accessed.
    // Contrast with ewiseadd.

    // A is passed as x, and B as y, in z = op(x,y)
    bool A_is_pattern = op_is_second || op_is_pair || op_is_positional ;
    bool B_is_pattern = op_is_first  || op_is_pair || op_is_positional ;

    //--------------------------------------------------------------------------
    // using a built-in binary operator (except for positional operators)
    //--------------------------------------------------------------------------

    bool done = false ;

    #ifndef GBCOMPACT

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_AemultB_100(mult,xname) GB_AemultB_100_ ## mult ## xname

        #define GB_BINOP_WORKER(mult,xname)                                 \
        {                                                                   \
            info = GB_AemultB_100(mult,xname) (C, M, Mask_struct, A, B,     \
                pstart_Mslice, kfirst_Mslice, klast_Mslice, Cp_kfirst,      \
                M_ntasks, M_nthreads) ;                                     \
            done = (info != GrB_NO_VALUE) ;                                 \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Type_code xcode, ycode, zcode ;
        if (!op_is_positional &&
            GB_binop_builtin (A->type, A_is_pattern, B->type, B_is_pattern,
            op, false, &opcode, &xcode, &ycode, &zcode) && ccode == zcode)
        { 
            #include "GB_binop_factory.c"
        }

    #endif

    //--------------------------------------------------------------------------
    // generic worker
    //--------------------------------------------------------------------------

    if (!done)
    { 
        // TODO: make this a function
        // see GB_emult_01, GB_emult_phase2, and even GB_add_phase2

        GB_BURBLE_MATRIX (C, "(generic emult_100: %s) ", op->name) ;

        GxB_binary_function fmult ;
        size_t csize, asize, bsize, xsize, ysize, zsize ;
        GB_cast_function cast_A_to_X, cast_B_to_Y, cast_Z_to_C ;

        fmult = op->function ;      // NULL if op is positional
        csize = ctype->size ;
        asize = B->type->size ;
        bsize = A->type->size ;

        GrB_Type xtype = op->xtype ;
        GrB_Type ytype = op->ytype ;

        if (A_is_pattern)
        { 
            // the op does not depend on the value of A(i,j)
            xsize = 1 ;
            cast_A_to_X = NULL ;
        }
        else
        { 
            xsize = xtype->size ;
            cast_A_to_X = GB_cast_factory (xtype->code, A->type->code) ;
        }

        if (B_is_pattern)
        { 
            // the op does not depend on the value of B(i,j)
            ysize = 1 ;
            cast_B_to_Y = NULL ;
        }
        else
        { 
            ysize = ytype->size ;
            cast_B_to_Y = GB_cast_factory (ytype->code, B->type->code) ;
        }

        zsize = op->ztype->size ;
        cast_Z_to_C = GB_cast_factory (ccode, op->ztype->code) ;

        // aij = (xtype) A(i,j), located in Ax [pA]
        #define GB_GETA(aij,Ax,pA)                                          \
            GB_void aij [GB_VLA(xsize)] ;                                   \
            if (cast_A_to_X != NULL)                                        \
            {                                                               \
                cast_A_to_X (aij, Ax +((pA)*asize), asize) ;                \
            }

        // bij = (ytype) B(i,j), located in Bx [pB]
        #define GB_GETB(bij,Bx,pB)                                          \
            GB_void bij [GB_VLA(ysize)] ;                                   \
            if (cast_B_to_Y != NULL)                                        \
            {                                                               \
                cast_B_to_Y (bij, Bx +((pB)*bsize), bsize) ;                \
            }

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        if (op_is_positional)
        { 

            //------------------------------------------------------------------
            // C(i,j) = positional_op (aij, bij)
            //------------------------------------------------------------------

            int64_t offset = GB_positional_offset (opcode) ;

            if (op->ztype == GrB_INT64)
            {
                switch (opcode)
                {
                    case GB_FIRSTI_opcode    : // z = first_i(A(i,j),y) == i
                    case GB_FIRSTI1_opcode   : // z = first_i1(A(i,j),y) == i+1
                    case GB_SECONDI_opcode   : // z = second_i(x,A(i,j)) == i
                    case GB_SECONDI1_opcode  : // z = second_i1(x,A(i,j)) == i+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)   \
                            int64_t z = i + offset ;            \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_100_template.c"
                        break ;
                    case GB_FIRSTJ_opcode    : // z = first_j(A(i,j),y) == j
                    case GB_FIRSTJ1_opcode   : // z = first_j1(A(i,j),y) == j+1
                    case GB_SECONDJ_opcode   : // z = second_j(x,A(i,j)) == j
                    case GB_SECONDJ1_opcode  : // z = second_j1(x,A(i,j)) == j+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)   \
                            int64_t z = j + offset ;            \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_100_template.c"
                        break ;
                    default: ;
                }
            }
            else
            {
                switch (opcode)
                {
                    case GB_FIRSTI_opcode    : // z = first_i(A(i,j),y) == i
                    case GB_FIRSTI1_opcode   : // z = first_i1(A(i,j),y) == i+1
                    case GB_SECONDI_opcode   : // z = second_i(x,A(i,j)) == i
                    case GB_SECONDI1_opcode  : // z = second_i1(x,A(i,j)) == i+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)       \
                            int32_t z = (int32_t) (i + offset) ;    \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_100_template.c"
                        break ;
                    case GB_FIRSTJ_opcode    : // z = first_j(A(i,j),y) == j
                    case GB_FIRSTJ1_opcode   : // z = first_j1(A(i,j),y) == j+1
                    case GB_SECONDJ_opcode   : // z = second_j(x,A(i,j)) == j
                    case GB_SECONDJ1_opcode  : // z = second_j1(x,A(i,j)) == j+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)       \
                            int32_t z = (int32_t) (j + offset) ;    \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_100_template.c"
                        break ;
                    default: ;
                }
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // standard binary operator
            //------------------------------------------------------------------

            // C(i,j) = (ctype) (A(i,j) + B(i,j))
            #undef  GB_BINOP
            #define GB_BINOP(cij, aij, bij, i, j)   \
                GB_void z [GB_VLA(zsize)] ;         \
                fmult (z, aij, bij) ;               \
                cast_Z_to_C (cij, z, csize) ;
            #include "GB_emult_100_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    // TODO: allow C->h to be shallow; if modified, make a copy
    GB_OK (GB_hypermatrix_prune (C, Context)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    ASSERT_MATRIX_OK (C, "C output for emult_100", GB0) ;
    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

