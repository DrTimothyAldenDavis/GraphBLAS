//------------------------------------------------------------------------------
// GB_emult_01: C = A.*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A.*B where A is sparse/hyper and B is bitmap/full constructs C with
// the same sparsity structure as A.

#include "GB_emult.h"
#include "GB_binop.h"

#define GB_FREE_WORK                                                    \
{                                                                       \
    GB_FREE (Wfirst) ;                                                  \
    GB_FREE (Wlast) ;                                                   \
    GB_FREE (Cp_kfirst) ;                                               \
    GB_ek_slice_free (&pstart_Aslice, &kfirst_Aslice, &klast_Aslice) ;  \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORK ;              \
    GB_Matrix_free (&C) ;       \
}

GrB_Info GB_emult_01        // C=A.*B when A is sparse/hyper, B bitmap/full
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix A,     // input A matrix (sparse/hyper)
    const GrB_Matrix B,     // input B matrix (bitmap/full)
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    bool flipxy,            // if true use fmult(y,x) else fmult(x,y)
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

    ASSERT (GB_IS_SPARSE (A) || GB_IS_BITMAP (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Wfirst = NULL ;
    int64_t *GB_RESTRICT Wlast = NULL ;
    int64_t *GB_RESTRICT Cp_kfirst = NULL ;
    int64_t *GB_RESTRICT pstart_Aslice = NULL ;
    int64_t *GB_RESTRICT kfirst_Aslice = NULL ;
    int64_t *GB_RESTRICT klast_Aslice  = NULL ;

    //--------------------------------------------------------------------------
    // get A and B
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    int64_t vlen = A->vlen ;
    int64_t vdim = A->vdim ;
    int64_t nvec = A->nvec ;

    const int8_t *GB_RESTRICT Bb = B->b ;
    bool B_is_bitmap = GB_IS_BITMAP (B) ;

    //--------------------------------------------------------------------------
    // allocate C->p and C->h
    //--------------------------------------------------------------------------

    GB_OK (GB_new (&C,      // sparse or hyper (same as A), new header
        ctype, vlen, vdim, GB_Ap_malloc, C_is_csc,
        GB_sparsity (A), A->hyper_switch, A->plen, Context)) ;
    int64_t *GB_RESTRICT Cp = C->p ;

    //--------------------------------------------------------------------------
    // slice the input matrix A
    //--------------------------------------------------------------------------

    int A_nthreads, A_ntasks ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    GB_SLICE_MATRIX (A, 8) ;

    //--------------------------------------------------------------------------
    // phase1: count entries in C
    //--------------------------------------------------------------------------

    C->nvec_nonempty = A->nvec_nonempty ;

    if (B_is_bitmap)
    {

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        // TODO: use one malloc
        Wfirst = GB_CALLOC (A_ntasks, int64_t) ;
        Wlast  = GB_CALLOC (A_ntasks, int64_t) ;
        Cp_kfirst = GB_CALLOC (A_ntasks, int64_t) ;
        if (Wfirst == NULL || Wlast  == NULL || Cp_kfirst == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // count entries in C
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < A_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Aslice [tid] ;
            int64_t klast  = klast_Aslice  [tid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // count the entries in C(:,j)
                int64_t j = GBH (Ah, k) ;
                int64_t pB_start = j * vlen ;
                int64_t pA, pA_end ;
                GB_get_pA (&pA, &pA_end, tid, k,
                    kfirst, klast, pstart_Aslice, Ap, vlen) ;
                int64_t cjnz = 0 ;
                for ( ; pA < pA_end ; pA++)
                { 
                    cjnz += Bb [pB_start + Ai [pA]] ;
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

        // finalize Cp
        GB_ek_slice_merge1 (Cp, Ap, vlen,
            Wfirst, Wlast, pstart_Aslice, kfirst_Aslice, klast_Aslice,
            A_ntasks) ;

        // cumulative sum of Cp and compute Cp_kfirst
        GB_ek_slice_merge2 (&(C->nvec_nonempty), Cp_kfirst, Cp, nvec,
            Wfirst, Wlast, kfirst_Aslice, klast_Aslice,
            A_ntasks, A_nthreads) ;
    }

    //--------------------------------------------------------------------------
    // allocate C->i and C->x
    //--------------------------------------------------------------------------

    int64_t cnz = (B_is_bitmap) ? Cp [nvec] : GB_NNZ (A) ;
    GB_OK (GB_bix_alloc (C, cnz, false, false, true, true, Context)) ;

    //--------------------------------------------------------------------------
    // copy pattern into C
    //--------------------------------------------------------------------------

    // TODO: could make these components of C shallow instead

    if (GB_IS_HYPERSPARSE (A))
    {
        GB_memcpy (C->h, Ah, nvec * sizeof (int64_t), A_nthreads) ;
    }

    if (!B_is_bitmap)
    {
        GB_memcpy (Cp, Ap, (nvec+1) * sizeof (int64_t), A_nthreads) ;
        GB_memcpy (C->i, Ai, cnz * sizeof (int64_t), A_nthreads) ;
    }

    C->jumbled = A->jumbled ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get the opcode
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;

    if (flipxy)
    {
        bool handled ;
        opcode = GB_binop_flip (opcode, &handled) ;
        if (handled) flipxy = false ;
    }

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

        #define GB_AemultB_01(mult,xname) GB_AemultB_01_ ## mult ## xname

        #define GB_BINOP_WORKER(mult,xname)                                 \
        {                                                                   \
            info = GB_AemultB_01(mult,xname) (C, A, B, flipxy,              \
                pstart_Aslice, kfirst_Aslice, klast_Aslice, Cp_kfirst,      \
                A_ntasks, A_nthreads) ;                                     \
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
        GB_BURBLE_MATRIX (C, "(generic emult_01: %s) ", op->name) ;

        GxB_binary_function fmult ;
        size_t csize, asize, bsize, xsize, ysize, zsize ;
        GB_cast_function cast_A_to_X, cast_B_to_Y, cast_Z_to_C ;

        // C = A .* B with optional typecasting
        fmult = op->function ;      // NULL if op is positional
        csize = ctype->size ;
        asize = A->type->size ;
        bsize = B->type->size ;

        if (A_is_pattern)
        { 
            // the op does not depend on the value of A(i,j)
            xsize = 1 ;
            cast_A_to_X = NULL ;
        }
        else
        { 
            xsize = op->xtype->size ;
            cast_A_to_X = GB_cast_factory (op->xtype->code, A->type->code) ;
        }

        if (B_is_pattern)
        { 
            // the op does not depend on the value of B(i,j)
            ysize = 1 ;
            cast_B_to_Y = NULL ;
        }
        else
        { 
            ysize = op->ytype->size ;
            cast_B_to_Y = GB_cast_factory (op->ytype->code, B->type->code) ;
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

        #define GB_FLIPPED 0

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
                        #include "GB_emult_01_template.c"
                        break ;
                    case GB_FIRSTJ_opcode    : // z = first_j(A(i,j),y) == j
                    case GB_FIRSTJ1_opcode   : // z = first_j1(A(i,j),y) == j+1
                    case GB_SECONDJ_opcode   : // z = second_j(x,A(i,j)) == j
                    case GB_SECONDJ1_opcode  : // z = second_j1(x,A(i,j)) == j+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)   \
                            int64_t z = j + offset ;            \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_01_template.c"
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
                        #include "GB_emult_01_template.c"
                        break ;
                    case GB_FIRSTJ_opcode    : // z = first_j(A(i,j),y) == j
                    case GB_FIRSTJ1_opcode   : // z = first_j1(A(i,j),y) == j+1
                    case GB_SECONDJ_opcode   : // z = second_j(x,A(i,j)) == j
                    case GB_SECONDJ1_opcode  : // z = second_j1(x,A(i,j)) == j+1
                        #undef  GB_BINOP
                        #define GB_BINOP(cij, aij, bij, i, j)       \
                            int32_t z = (int32_t) (j + offset) ;    \
                            cast_Z_to_C (cij, &z, csize) ;
                        #include "GB_emult_01_template.c"
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
                ASSERT (op != NULL) ;               \
                GB_void z [GB_VLA(zsize)] ;         \
                if (flipxy)                         \
                {                                   \
                    fmult (z, bij, aij) ;           \
                }                                   \
                else                                \
                {                                   \
                    fmult (z, aij, bij) ;           \
                }                                   \
                cast_Z_to_C (cij, z, csize) ;
            #include "GB_emult_01_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // construct the final C->h
    //--------------------------------------------------------------------------

    if (C->h != NULL)
    { 
        // TODO: allow C->h to be shallow; if modified, make a copy
        GB_OK (GB_hypermatrix_prune (C, Context)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    ASSERT_MATRIX_OK (C, "C output for emult_01", GB0) ;
    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

