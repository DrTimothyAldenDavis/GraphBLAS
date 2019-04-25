//------------------------------------------------------------------------------
// GB_AxB_dot2: compute C<M> = A'*B in parallel, in place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_dot2 is very similar to GB_AxB_dot, except that it does the
// computation in two phases.  The first phase counts the number of entries in
// each column of C.  The second phase can then construct the result C in
// place, and thus this method can be done in parallel for the single matrix
// computation C=A'*B.  GB_AxB_dot2 operates in parallel on the slices of A,
// whereas GB_AxB_dot can only work on a single matrix (or a single slice).

// Any variant of the mask is handled: C=A'*B, C<M>=A'*B, and C<!M>=A'*B.

// Parallel: done

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

#define GB_FREE_ALL                                                     \
{                                                                       \
    for (int taskid = 0 ; taskid < naslice ; taskid++)                  \
    {                                                                   \
        GB_FREE_MEMORY (C_counts [taskid], cnvec, sizeof (int64_t)) ;   \
    }                                                                   \
}

GrB_Info GB_AxB_dot2                // C = A'*B using dot product method
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix M,             // mask matrix for C<M>=A'*B or C<!M>=A'*B
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix *Aslice,       // input matrices (already sliced)
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, mask was applied
    int nthreads,
    int naslice,
    int nbslice,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = Aslice [0] ;     // just for type and dimensions
    ASSERT (Chandle != NULL) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for dot A'*B", GB0)) ;
    ASSERT_OK (GB_check (A, "A for dot A'*B", GB0)) ;
    for (int taskid = 0 ; taskid < naslice ; taskid++)
    {
        ASSERT_OK (GB_check (Aslice [taskid], "A slice for dot2 A'*B", GB0)) ;
        ASSERT (!GB_PENDING (Aslice [taskid])) ;
        ASSERT (!GB_ZOMBIES (Aslice [taskid])) ;
        ASSERT ((Aslice [taskid])->vlen == B->vlen) ;
        ASSERT (A->vlen == (Aslice [taskid])->vlen) ;
        ASSERT (A->vdim == (Aslice [taskid])->vdim) ;
        ASSERT (A->type == (Aslice [taskid])->type) ;
    }
    ASSERT_OK (GB_check (B, "B for dot A'*B", GB0)) ;
    ASSERT (!GB_PENDING (M)) ; ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT_OK (GB_check (semiring, "semiring for numeric A'*B", GB0)) ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (mask_applied != NULL) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;

    bool op_is_first  = mult->opcode == GB_FIRST_opcode ;
    bool op_is_second = mult->opcode == GB_SECOND_opcode ;
    bool A_is_pattern = false ;
    bool B_is_pattern = false ;

    if (flipxy)
    { 
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  ;
        B_is_pattern = op_is_second ;
        if (!A_is_pattern) ASSERT (GB_Type_compatible (A->type, mult->ytype)) ;
        if (!B_is_pattern) ASSERT (GB_Type_compatible (B->type, mult->xtype)) ;
    }
    else
    { 
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second ;
        B_is_pattern = op_is_first  ;
        if (!A_is_pattern) ASSERT (GB_Type_compatible (A->type, mult->xtype)) ;
        if (!B_is_pattern) ASSERT (GB_Type_compatible (B->type, mult->ytype)) ;
    }

    (*Chandle) = NULL ;

    // the dot method handles any mask, complemented or not complemented

    //--------------------------------------------------------------------------
    // compute # of entries in each vector of C
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    int64_t cvlen = A->vdim ;
    int64_t cvdim = B->vdim ;

    if (B->nvec_nonempty < 0)
    { 
        B->nvec_nonempty = GB_nvec_nonempty (B, NULL) ;
    }

    int64_t *C_counts [naslice] ;
    double t = omp_get_wtime ( ) ;
    GrB_Info task_info [naslice] ;

//  #pragma omp parallel for num_threads(nthreads) schedule(static,1) \
//      reduction(&&:ok) 

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp single
        {
            for (int taskid = 0 ; taskid < naslice ; taskid++)
            {
                #pragma omp task
                {

                    if ((Aslice [taskid])->nvec_nonempty < 0)
                    { 
                        (Aslice [taskid])->nvec_nonempty =
                            GB_nvec_nonempty (Aslice [taskid], NULL) ;
                    }
                    // count # of entries in each vector of C, for the slice
                    task_info [taskid] = GB_AxB_dot2_phase1
                        (&(C_counts [taskid]), M, Mask_comp, Aslice [taskid],
                        B, nthreads, naslice, nbslice) ;
                }
            }
        }
    }

GB_HERE ;

    // collect all thread-specific info
    bool ok = true ;
    for (int taskid = 0 ; taskid < naslice ; taskid++)
    {
        ok = ok && (task_info [taskid] == GrB_SUCCESS) ;
    }

    int64_t cnvec = B->nvec ;

    GB_NEW (Chandle, ctype, cvlen, cvdim, GB_Ap_malloc, true,
        GB_SAME_HYPER_AS (B->is_hyper), B->hyper_ratio, cnvec, Context) ;
    if (!ok || info != GrB_SUCCESS)
    {
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    GrB_Matrix C = (*Chandle) ;
    int64_t *restrict Cp = C->p ;

    // cumulative sum of counts in each column
    // TODO skip if naslice == 1
    #pragma omp parallel for num_threads(nthreads)
    for (int64_t k = 0 ; k < cnvec ; k++)
    {
        int64_t s = 0 ;
        // #pragma omp simd reduction(+:s)
        for (int taskid = 0 ; taskid < naslice ; taskid++)
        {
            int64_t *C_count = C_counts [taskid] ;
            int64_t c = C_count [k] ;
            // printf ("taskid %d k "GBd" c "GBd"\n", taskid, k, c) ;
            C_count [k] = s ;
            s += c ;
        }
        Cp [k] = s ;
    }
    Cp [cnvec] = 0 ;
    C->nvec = cnvec ;

    // Cp = cumulative sum of Cp
    GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nthreads) ;
    int64_t cnz = Cp [cnvec] ;

// for (int64_t k = 0 ; k <= cnvec ; k++) printf ("Cp ["GBd"] = "GBd"\n", k, Cp [k]) ;
// printf ("C->nvec is "GBd"\n", C->nvec) ;

    // C->h = B->h
    if (B->is_hyper)
    {
        GB_memcpy (C->h, B->h, cnvec * sizeof (int64_t), nthreads) ;
    }

    // free C_count for the first thread; it is no longer needed
    GB_FREE_MEMORY (C_counts [0], cnvec, sizeof (int64_t)) ;
    C->magic = GB_MAGIC ;

    t = omp_get_wtime ( ) - t ;
    printf ("dot2 phase1: %g\n", t) ;
    t = omp_get_wtime ( )  ;

    //--------------------------------------------------------------------------
    // allocate C->x and C->i
    //--------------------------------------------------------------------------

    info = GB_ix_alloc (C, cnz, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // C = A'*B, computing each entry with a dot product, via builtin semiring
    //--------------------------------------------------------------------------

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp single
        {
            for (int taskid = 0 ; taskid < naslice ; taskid++)
            {
                #pragma omp task
                {

// printf ("\n====================== taskid %d of %d\n", taskid, naslice) ;

    int64_t *restrict C_count_start =
        (taskid == 0) ?          NULL : C_counts [taskid] ;
    int64_t *restrict C_count_end   =
        (taskid == naslice-1) ? NULL : C_counts [taskid+1] ;

    GrB_Matrix A = Aslice [taskid] ;

    // TODO call this a function GB_AxB_dot2_phase2

// GxB_print (Aslice [taskid], 3) ;

// if (C_count_start != NULL)
//    for (int64_t k = 0 ; k < cnvec ; k++) printf ("C_count_start ["GBd"] = "GBd"\n",
//        k, C_count_start [k]) ;

// if (C_count_end != NULL)
//    for (int64_t k = 0 ; k < cnvec ; k++) printf ("C_count_end   ["GBd"] = "GBd"\n",
//        k, C_count_end   [k]) ;

    bool done = false ;

#ifndef GBCOMPACT

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_Adot2B(add,mult,xyname) GB_Adot2B_ ## add ## mult ## xyname

    #define GB_AxB_WORKER(add,mult,xyname)                              \
    {                                                                   \
        info = GB_Adot2B (add,mult,xyname) (Chandle, M, Mask_comp,      \
            A, A_is_pattern, B, B_is_pattern,                           \
            C_count_start, C_count_end, nthreads, naslice, nbslice) ;   \
        done = true ;                                                   \
    }                                                                   \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    GB_Opcode mult_opcode, add_opcode ;
    GB_Type_code xycode, zcode ;

    if (GB_AxB_semiring_builtin (A, A_is_pattern, B, B_is_pattern, semiring,
        flipxy, &mult_opcode, &add_opcode, &xycode, &zcode))
    { 
        #include "GB_AxB_factory.c"
    }

#endif

    //--------------------------------------------------------------------------
    // user semirings created at compile time
    //--------------------------------------------------------------------------

    if (semiring->object_kind == GB_USER_COMPILED)
    {

        // determine the required type of A and B for the user semiring
        GrB_Type atype_required, btype_required ;

        if (flipxy)
        { 
            // A is passed as y, and B as x, in z = mult(x,y)
            atype_required = mult->ytype ;
            btype_required = mult->xtype ;
        }
        else
        { 
            // A is passed as x, and B as y, in z = mult(x,y)
            atype_required = mult->xtype ;
            btype_required = mult->ytype ;
        }

        if (A->type == atype_required && B->type == btype_required)
        {
            info = GB_AxB_user (GxB_AxB_DOT2, semiring, Chandle, M, A, B,
                flipxy, Mask_comp, NULL, NULL, NULL, 0, NULL,
                C_count_start, C_count_end) ;
            done = true ;
        }
    }

    //--------------------------------------------------------------------------
    // C = A'*B, computing each entry with a dot product, with typecasting
    //--------------------------------------------------------------------------

    if (!done)
    {

        //----------------------------------------------------------------------
        // get operators, functions, workspace, contents of A, B, C, and M
        //----------------------------------------------------------------------

        GxB_binary_function fmult = mult->function ;
        GxB_binary_function fadd  = add->op->function ;

        size_t csize = C->type->size ;
        size_t asize = A_is_pattern ? 0 : A->type->size ;
        size_t bsize = B_is_pattern ? 0 : B->type->size ;

        size_t xsize = mult->xtype->size ;
        size_t ysize = mult->ytype->size ;

        // scalar workspace: because of typecasting, the x/y types need not
        // be the same as the size of the A and B types.
        // flipxy false: aki = (xtype) A(k,i) and bkj = (ytype) B(k,j)
        // flipxy true:  aki = (ytype) A(k,i) and bkj = (xtype) B(k,j)
        size_t aki_size = flipxy ? ysize : xsize ;
        size_t bkj_size = flipxy ? xsize : ysize ;

        GB_void *restrict identity = add->identity ;
        GB_void *restrict terminal = add->terminal ;

        GB_cast_function cast_A, cast_B ;
        if (flipxy)
        { 
            // A is typecasted to y, and B is typecasted to x
            cast_A = A_is_pattern ? NULL : 
                     GB_cast_factory (mult->ytype->code, A->type->code) ;
            cast_B = B_is_pattern ? NULL : 
                     GB_cast_factory (mult->xtype->code, B->type->code) ;
        }
        else
        { 
            // A is typecasted to x, and B is typecasted to y
            cast_A = A_is_pattern ? NULL :
                     GB_cast_factory (mult->xtype->code, A->type->code) ;
            cast_B = B_is_pattern ? NULL :
                     GB_cast_factory (mult->ytype->code, B->type->code) ;
        }

        //----------------------------------------------------------------------
        // C = A'*B via dot products, function pointers, and typecasting
        //----------------------------------------------------------------------

        // aki = A(k,i), located in Ax [pA]
        #define GB_GETA(aki,Ax,pA)                                          \
            GB_void aki [aki_size] ;                                        \
            if (!A_is_pattern) cast_A (aki, Ax +((pA)*asize), asize) ;

        // bkj = B(k,j), located in Bx [pB]
        #define GB_GETB(bkj,Bx,pB)                                          \
            GB_void bkj [bkj_size] ;                                        \
            if (!B_is_pattern) cast_B (bkj, Bx +((pB)*bsize), bsize) ;

        // break if cij reaches the terminal value
        #define GB_DOT_TERMINAL(cij)                                        \
            if (terminal != NULL && memcmp (cij, terminal, csize) == 0)     \
            {                                                               \
                break ;                                                     \
            }

        // C(i,j) = A(i,k) * B(k,j)
        #define GB_MULT(cij, aki, bkj)                                      \
            GB_MULTIPLY (cij, aki, bkj) ;                                   \

        // C(i,j) += A(i,k) * B(k,j)
        #define GB_MULTADD(cij, aki, bkj)                                   \
            GB_void zwork [csize] ;                                         \
            GB_MULTIPLY (zwork, aki, bkj) ;                                 \
            fadd (cij, cij, zwork) ;

        // define cij for each task
        #define GB_CIJ_DECLARE(cij)                                         \
            GB_void cij [csize] ;

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        // save the value of C(i,j)
        #define GB_CIJ_SAVE(cij,p)                                          \
            memcpy (GB_CX (p), cij, csize) ;

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        #define GB_PHASE_2_OF_2

        if (flipxy)
        { 
            #define GB_MULTIPLY(z,x,y) fmult (z,y,x)
            #include "GB_AxB_dot_meta.c"
            #undef GB_MULTIPLY
        }
        else
        { 
            #define GB_MULTIPLY(z,x,y) fmult (z,x,y)
            #include "GB_AxB_dot_meta.c"
            #undef GB_MULTIPLY
        }
    }

    } } } }

    t = omp_get_wtime ( ) - t ;
    printf ("dot2 phase2: %g\n", t) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // TODO: if C->nvec_nonempty is small, then check if C should be converted
    // to hypersparse form.  If already hypersparse, prune C->h.

    GB_FREE_ALL ;
    ASSERT_OK (GB_check (C, "dot: C = A'*B output", GB0)) ;
    ASSERT (*Chandle == C) ;
    (*mask_applied) = (M != NULL) ;
    return (GrB_SUCCESS) ;
}

