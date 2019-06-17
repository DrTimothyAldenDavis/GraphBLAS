//------------------------------------------------------------------------------
// GB_subassign.h: helper macros for GB_subassigner and GB_subassign_method*
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#define GB_DEBUG
#include "GB.h"

//------------------------------------------------------------------------------
// get the C matrix
//------------------------------------------------------------------------------

#define GB_GET_C                                                            \
    GrB_Info info ;                                                         \
    ASSERT_OK (GB_check (C, "C for subassign kernel", GB0)) ;               \
    const int64_t *Ch = C->h ;                                              \
    const int64_t *Cp = C->p ;                                              \
    int64_t *Ci = C->i ;                                                    \
    GB_void *Cx = C->x ;                                                    \
    const size_t csize = C->type->size ;                                    \
    const GB_Type_code ccode = C->type->code ;                              \
    int64_t cnvec = C->nvec ;                                               \
    int64_t cvlen = C->vlen ;                                               \
    int64_t cvdim = C->vdim ;                                               \
    bool C_is_hyper = C->is_hyper && (cnvec < cvdim) ;

//------------------------------------------------------------------------------
// get the content of the mask matrix M
//------------------------------------------------------------------------------

#define GB_GET_MASK                                                           \
    ASSERT_OK (GB_check (M, "M for assign", GB0)) ;                           \
    const int64_t *Mi = M->i ;                                                \
    const GB_void *Mx = M->x ;                                                \
    size_t msize = M->type->size ;                                            \
    GB_cast_function cast_M = GB_cast_factory (GB_BOOL_code, M->type->code) ; \

//------------------------------------------------------------------------------
// get the accumulator operator and its related typecasting functions
//------------------------------------------------------------------------------

#define GB_GET_ACCUM                                                           \
    ASSERT_OK (GB_check (accum, "accum for assign", GB0)) ;                    \
    GxB_binary_function faccum = accum->function ;                             \
    GB_cast_function cast_A_to_Y = GB_cast_factory (accum->ytype->code, acode);\
    GB_cast_function cast_C_to_X = GB_cast_factory (accum->xtype->code, ccode);\
    GB_cast_function cast_Z_to_C = GB_cast_factory (ccode, accum->ztype->code);\
    size_t xsize = accum->xtype->size ;                                        \
    size_t ysize = accum->ytype->size ;                                        \
    size_t zsize = accum->ztype->size ;

//------------------------------------------------------------------------------
// get the A matrix
//------------------------------------------------------------------------------

#define GB_GET_A                                                            \
    ASSERT_OK (GB_check (A, "A for assign", GB0)) ;                         \
    GrB_Type atype = A->type ;                                              \
    size_t asize = atype->size ;                                            \
    GB_Type_code acode = atype->code ;                                      \
    const int64_t *restrict Ai = A->i ;                                     \
    const GB_void *restrict Ax = A->x ;                                     \
    GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;

//------------------------------------------------------------------------------
// get the scalar
//------------------------------------------------------------------------------

#define GB_GET_SCALAR                                                       \
    ASSERT_OK (GB_check (atype, "atype for assign", GB0)) ;                 \
    size_t asize = atype->size ;                                            \
    GB_Type_code acode = atype->code ;                                      \
    GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;         \
    GB_void cwork [csize] ;                                                 \
    cast_A_to_C (cwork, scalar, asize) ;                                    \

//------------------------------------------------------------------------------
// get the scalar and the accumulator
//------------------------------------------------------------------------------

#define GB_GET_ACCUM_SCALAR                                                 \
    GB_GET_SCALAR ;                                                         \
    GB_GET_ACCUM ;                                                          \
    GB_void ywork [ysize] ;                                                 \
    cast_A_to_Y (ywork, scalar, asize) ;

//------------------------------------------------------------------------------
// get the S matrix
//------------------------------------------------------------------------------

#define GB_GET_S                                                            \
    ASSERT_OK (GB_check (S, "S extraction", GB0)) ;                         \
    const int64_t *restrict Si = S->i ;                                     \
    const int64_t *restrict Sx = S->x ;

//------------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // S_Extraction: finding C(iC,jC) via lookup through S=C(I,J)
    //--------------------------------------------------------------------------

    // S is the symbolic pattern of the submatrix S = C(I,J).  The "numerical"
    // value (held in S->x) of an entry S(i,j) is not a value, but a pointer
    // back into C where the corresponding entry C(iC,jC) can be found, where
    // iC = I [i] and jC = J [j].

    // The following macro performs the lookup.  Given a pointer pS into a
    // column S(:,j), it finds the entry C(iC,jC), and also determines if the
    // C(iC,jC) entry is a zombie.  The column indices j and jC are implicit.

    // This is used for Methods 7 to 14, all of which use S.

    #define GB_C_S_LOOKUP                                                   \
        int64_t pC = Sx [pS] ;                                              \
        int64_t iC = Ci [pC] ;                                              \
        bool is_zombie = GB_IS_ZOMBIE (iC) ;                                \
        if (is_zombie) iC = GB_FLIP (iC) ;

    //--------------------------------------------------------------------------
    // C(:,jC) is dense: iC = I [iA], and then look up C(iC,jC)
    //--------------------------------------------------------------------------

    // C(:,jC) is dense, and thus can be accessed with a constant-time lookup
    // with the index iC, where the index iC comes from I [iA] or via a
    // colon notation for I.

    // This used for Methods 1 to 6, which do not use S.

    #define GB_CDENSE_I_LOOKUP                                              \
        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;                     \
        int64_t pC = pC_start + iC ;                                        \
        bool is_zombie = GB_IS_ZOMBIE (Ci [pC]) ;                           \
        ASSERT (GB_UNFLIP (Ci [pC]) == iC) ;

    //--------------------------------------------------------------------------
    // get the C(:,jC) vector where jC = J [j]
    //--------------------------------------------------------------------------

    // C may be standard sparse, or hypersparse
    // time: O(1) if standard, O(log(cnvec)) if hyper

    // This used for Methods 1 to 6, which do not use S.

    #define GB_jC_LOOKUP                                                    \
        /* lookup jC in C */                                                \
        /* jC = J [j] ; or J is ":" or jbegin:jend or jbegin:jinc:jend */   \
        jC = GB_ijlist (J, j, Jkind, Jcolon) ;                              \
        int64_t pC_start, pC_end, pleft = 0, pright = cnvec-1 ;             \
        GB_lookup (C_is_hyper, Ch, Cp, &pleft, pright, jC, &pC_start, &pC_end) ;

    //--------------------------------------------------------------------------
    // get C(iC,jC) via binary search of C(:,jC)
    //--------------------------------------------------------------------------

    // PARALLEL: pC_start and pC_end for fine tasks need to be for the slice of
    // C(:,jC), not all of C(:,jC).  The pattern of C is not changing, except
    // that zombies can be introduced.  If one threads tries to do a
    // binary_zombie search and reads an entry outside its slice, another
    // thread might be changing that index to/from a zombie.  Either is OK for
    // the binary search to work, but a race condition may arise if the read
    // and write are not atomic and the reader thread gets a partial result in
    // the middle of the writer's update.  Atomics are slow, so to avoid the
    // simultaneous read/write, each fine task limits its binary search to its
    // slice of C(:,jC).

    // This used for Methods 1 to 6, which do not use S.

    #define GB_iC_BINARY_SEARCH                                                \
        int64_t pC = pC_start ;                                                \
        int64_t pright = pC_end - 1 ;                                          \
        bool found, is_zombie ;                                                \
        /* TODO parallel check for zombies */                                  \
        GB_BINARY_ZOMBIE (iC, Ci, pC, pright, found, C->nzombies, is_zombie) ;

    //--------------------------------------------------------------------------
    // for a 2-way or 3-way merge
    //--------------------------------------------------------------------------

    // An entry S(i,j), A(i,j), or M(i,j) has been processed;
    // move to the next one.
    #define GB_NEXT(X) (p ## X)++ ;

    //--------------------------------------------------------------------------
    // basic operations
    //--------------------------------------------------------------------------

    #define GB_COPY_scalar_to_C                                         \
    {                                                                   \
        /* C(iC,jC) = scalar, already typecasted into cwork      */     \
        memcpy (Cx +(pC*csize), cwork, csize) ;                         \
    }

    #define GB_COPY_aij_to_C                                            \
    {                                                                   \
        /* C(iC,jC) = A(i,j), with typecasting                   */     \
        cast_A_to_C (Cx +(pC*csize), Ax +(pA*asize), csize) ;           \
    }

    #define GB_COPY_aij_to_ywork                                        \
    {                                                                   \
        /* ywork = A(i,j), with typecasting                      */     \
        cast_A_to_Y (ywork, Ax +(pA*asize), asize) ;                    \
    }

    #define GB_ACCUMULATE                                               \
    {                                                                   \
        /* C(iC,jC) = accum (C(iC,jC), ywork)                    */     \
        GB_void xwork [xsize] ;                                         \
        cast_C_to_X (xwork, Cx +(pC*csize), csize) ;                    \
        GB_void zwork [zsize] ;                                         \
        faccum (zwork, xwork, ywork) ;                                  \
        cast_Z_to_C (Cx +(pC*csize), zwork, csize) ;                    \
    }                                                                   \

    #define GB_DELETE                                                   \
    {                                                                   \
        /* turn C(iC,jC) into a zombie */                               \
        C->nzombies++ ;  /* TODO: thread private; reduce when done*/\
        Ci [pC] = GB_FLIP (iC) ;                                        \
    }

    #define GB_UNDELETE                                                 \
    {                                                                   \
        /* bring a zombie C(iC,jC) back to life;                 */     \
        /* the value of C(iC,jC) must also be assigned.          */     \
        Ci [pC] = iC ;                                                  \
        C->nzombies-- ;  /* TODO: thread private; reduce when done*/\
    }

    #define GB_INSERT(aij)                                              \
    {                                                                   \
        /* C(iC,jC) = aij, inserting a pending tuple.  aij is */        \
        /* either A(i,j) or the scalar for scalar expansion */          \
        info = GB_pending_add (C, aij, atype, accum, iC, jC, Context) ; \
        if (info != GrB_SUCCESS)                                        \
        {                                                               \
            /* failed to add pending tuple */                           \
            return (info) ;                                             \
        }                                                               \
    }

    //--------------------------------------------------------------------------
    // C(I,J)<M> = accum (C(I,J),A): consider all cases
    //--------------------------------------------------------------------------

        // The matrix C may have pending tuples and zombies:

        // (1) pending tuples:  this is a list of pending updates held as a set
        // of (i,j,x) tuples.  They had been added to the list via a prior
        // GrB_setElement or GxB_subassign.  No operator needs to be applied to
        // them; the implied operator is SECOND, for both GrB_setElement and
        // GxB_subassign, regardless of whether or not an accum operator is
        // present.  Pending tuples are inserted if and only if the
        // corresponding entry C(i,j) does not exist, and in that case no accum
        // operator is applied.

        //      The GrB_setElement method (C(i,j) = x) is same as GxB_subassign
        //      with: accum is SECOND, C not replaced, no mask M, mask not
        //      complemented.  If GrB_setElement needs to insert its update as
        //      a pending tuple, then it will always be compatible with all
        //      pending tuples inserted here, by GxB_subassign.

        // (2) zombie entries.  These are entries that are still present in the
        // pattern but marked for deletion (via GB_FLIP(i) for the row index).

        // For the current GxB_subassign, there are 16 cases to handle,
        // all combinations of the following options:

        //      accum is NULL, accum is not NULL
        //      C is not replaced, C is replaced
        //      no mask, mask is present
        //      mask is not complemented, mask is complemented

        // Complementing an empty mask:  This does not require the matrix A
        // at all so it is handled as a special case.  It corresponds to
        // the GB_RETURN_IF_QUICK_MASK option in other GraphBLAS operations.
        // Thus only 12 cases are considered in the tables below:

        //      These 4 cases are listed in Four Tables below:
        //      2 cases: accum is NULL, accum is not NULL
        //      2 cases: C is not replaced, C is replaced

        //      3 cases: no mask, M is present and not complemented,
        //               and M is present and complemented.  If there is no
        //               mask, then M(i,j)=1 for all (i,j).  These 3 cases
        //               are the columns of each of the Four Tables.

        // Each of these 12 cases can encounter up to 12 combinations of
        // entries in C, A, and M (6 if no mask M is present).  The left
        // column of the Four Tables below consider all 12 combinations for all
        // (i,j) in the cross product IxJ:

        //      C(I(i),J(j)) present, zombie, or not there: C, X, or '.'
        //      A(i,j) present or not, labeled 'A' or '.' below
        //      M(i,j) = 1 or 0 (but only if M is present)

        //      These 12 cases become the left columns as listed below.
        //      The zombie cases are handled a sub-case for "C present:
        //      regular entry or zombie".  The acronyms below use "D" for
        //      "dot", meaning the entry (C or A) is not present.

        //      [ C A 1 ]   C_A_1: both C and A present, M=1
        //      [ X A 1 ]   C_A_1: both C and A present, M=1, C is a zombie
        //      [ . A 1 ]   D_A_1: C not present, A present, M=1

        //      [ C . 1 ]   C_D_1: C present, A not present, M=1
        //      [ X . 1 ]   C_D_1: C present, A not present, M=1, C a zombie
        //      [ . . 1 ]          only M=1 present, but nothing to do

        //      [ C A 0 ]   C_A_0: both C and A present, M=0
        //      [ X A 0 ]   C_A_0: both C and A present, M=0, C is a zombie
        //      [ . A 0 ]          C not present, A present, M=0,
        //                              nothing to do

        //      [ C . 0 ]   C_D_0: C present, A not present, M=1
        //      [ X . 0 ]   C_D_0: C present, A not present, M=1, C a zombie
        //      [ . . 0 ]          only M=0 present, but nothing to do

        // Legend for action taken in the right half of the table:

        //      delete   live entry C(I(i),J(j)) marked for deletion (zombie)
        //      =A       live entry C(I(i),J(j)) is overwritten with new value
        //      =C+A     live entry C(I(i),J(j)) is modified with accum(c,a)
        //      C        live entry C(I(i),J(j)) is unchanged

        //      undelete entry C(I(i),J(j)) a zombie, bring back with A(i,j)
        //      X        entry C(I(i),J(j)) a zombie, no change, still zombie

        //      insert   entry C(I(i),J(j)) not present, add pending tuple
        //      .        entry C(I(i),J(j)) not present, no change

        //      blank    the table is left blank where the the event cannot
        //               occur:  GxB_subassign with no M cannot have
        //               M(i,j)=0, and GrB_setElement does not have the M
        //               column

        //----------------------------------------------------------------------
        // GrB_setElement and the Four Tables for GxB_subassign:
        //----------------------------------------------------------------------

            //------------------------------------------------------------
            // GrB_setElement:  no mask
            //------------------------------------------------------------

            // C A 1        =A                               |
            // X A 1        undelete                         |
            // . A 1        insert                           |

            //          GrB_setElement acts exactly like GxB_subassign with the
            //          implicit GrB_SECOND_Ctype operator, I=i, J=j, and a
            //          1-by-1 matrix A containing a single entry (not an
            //          implicit entry; there is no "." for A).  That is,
            //          nnz(A)==1.  No mask, and the descriptor is the default;
            //          C_replace effectively false, mask not complemented, A
            //          not transposed.  As a result, GrB_setElement can be
            //          freely mixed with calls to GxB_subassign with C_replace
            //          effectively false and with the identical
            //          GrB_SECOND_Ctype operator.  These calls to
            //          GxB_subassign can use the mask, either complemented or
            //          not, and they can transpose A if desired, and there is
            //          no restriction on I and J.  The matrix A can be any
            //          type and the type of A can change from call to call.

            //------------------------------------------------------------
            // NO accum  |  no mask     mask        mask
            // NO repl   |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =A          =A          C        |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        delete      delete      C        |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    C           =A       |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    C           delete   |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well: first extract pattern
            //          of S=C(I,J). Then examine all of A, M, S, and update
            //          C(I,J).  The method needs to examine all entries in
            //          in C(I,J) to delete them if A is not present, so
            //          S=C(I,J) is not costly.

            //------------------------------------------------------------
            // NO accum  |  no mask     mask        mask
            // WITH repl |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =A          =A          delete   |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        delete      delete      delete   |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    delete      =A       |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    delete      delete   |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well, since all of C(I,J)
            //          needs to be traversed, S=C(I,J) is reasonable to
            //          compute.

            //          With no accum: If there is no M and M is not
            //          complemented, then C_replace is irrelevant,  Whether
            //          true or false, the results in the two tables
            //          above are the same.

            //------------------------------------------------------------
            // ACCUM     |  no mask     mask        mask
            // NO repl   |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =C+A        =C+A        C        |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        C           C           C        |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    C           =C+A     |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    C           C        |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          With ACCUM but NO C_replace: This method only needs to
            //          examine entries in A.  It does not need to examine all
            //          entries in C(I,J), nor all entries in M.  Entries in
            //          C but in not A remain unchanged.  This is like an
            //          extended GrB_setElement.  No entries in C can be
            //          deleted.  All other methods must examine all of C(I,J).

            //          Without S_Extraction: C(:,J) or M have many entries
            //          compared with A, do not extract S=C(I,J); use
            //          binary search instead.  Otherwise, use the same
            //          S_Extraction method as the other 3 cases.

            //          S_Extraction method: if nnz(C(:,j)) + nnz(M) is
            //          similar to nnz(A) then the S_Extraction method would
            //          work well.

            //------------------------------------------------------------
            // ACCUM     |  no mask     mask        mask
            // WITH repl |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =C+A        =C+A        delete   |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        C           C           delete   |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    delete      =C+A     |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    delete      C        |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well since all entries
            //          in C(I,J) must be examined.

            //          With accum: If there is no M and M is not
            //          complemented, then C_replace is irrelavant,  Whether
            //          true or false, the results in the two tables
            //          above are the same.

            //          This condition on C_replace holds with our without
            //          accum.  Thus, if there is no M, and M is
            //          not complemented, the C_replace can be set to false.

            //------------------------------------------------------------

            // ^^^^^ legend for left columns above:
            // C        prior entry C(I(i),J(j)) exists
            // X        prior entry C(I(i),J(j)) exists but is a zombie
            // .        no prior entry C(I(i),J(j))
            //   A      A(i,j) exists
            //   .      A(i,j) does not exist
            //     1    M(i,j)=1, assuming M exists (or if implicit)
            //     0    M(i,j)=0, only if M exists

        //----------------------------------------------------------------------
        // Actions in the Four Tables above
        //----------------------------------------------------------------------

            // Each entry in the Four Tables above are now explained in more
            // detail, describing what must be done in each case.  Zombies and
            // pending tuples are disjoint; they do not mix.  Zombies are IN
            // the pattern but pending tuples are updates that are NOT in the
            // pattern.  That is why a separate list of pending tuples must be
            // kept; there is no place for them in the pattern.  Zombies, on
            // the other hand, are entries IN the pattern that have been
            // marked for deletion.

            //--------------------------------
            // For entries NOT in the pattern:
            //--------------------------------

            // They can have pending tuples, and can acquire more.  No zombies.

            //      ( insert ):

            //          An entry C(I(i),J(j)) is NOT in the pattern, but its
            //          value must be modified.  This is an insertion, like
            //          GrB_setElement, and the insertion is added as a pending
            //          tuple for C(I(i),J(j)).  There can be many insertions
            //          to the same element, each in the list of pending
            //          tuples, in order of their insertion.  Eventually these
            //          pending tuples must be assembled into C(I(i),J(j)) in
            //          the right order using the implied SECOND operator.

            //      ( . ):

            //          no change.  C(I(i),J(j)) not in the pattern, and not
            //          modified.  This C(I(i),J(j)) position could have
            //          pending tuples, in the list of pending tuples, but none
            //          of them are changed.  If C_replace is true then those
            //          pending tuples would have to be discarded, but that
            //          condition will not occur because C_replace=true forces
            //          all prior tuples to the matrix to be assembled.

            //--------------------------------
            // For entries IN the pattern:
            //--------------------------------

            // They have no pending tuples, and acquire none.  It can be
            // zombie, can become a zombie, or a zombie can come back to life.

            //      ( delete ):

            //          C(I(i),J(j)) becomes a zombie, by flipping its row
            //          index via the GB_FLIP function.

            //      ( undelete ):

            //          C(I(i),J(j)) = A(i,j) was a zombie and is no longer a
            //          zombie.  Its row index is restored with GB_FLIP.

            //      ( X ):

            //          C(I(i),J(j)) was a zombie, and still is a zombie.
            //          row index is < 0, and actual index is GB_FLIP(I(i))

            //      ( C ):

            //          no change; C(I(i),J(j)) already an entry, and its value
            //          doesn't change.

            //      ( =A ):

            //          C(I(i),J(j)) = A(i,j), value gets overwritten.

            //      ( =C+A ):

            //          C(I(i),J(j)) = accum (C(I(i),J(j)), A(i,j))
            //          The existing balue is modified via the accumulator.


    //--------------------------------------------------------------------------
    // handling each action
    //--------------------------------------------------------------------------

        // Each of the 12 cases are handled by the following actions,
        // implemented as macros.  The Four Tables are re-sorted below,
        // and folded together according to their left column.

        // Once the M(i,j) entry is extracted, the codes below explicitly
        // complement the scalar value if Mask_complement is true, before using
        // these action functions.  For the [no mask] case, M(i,j)=1.  Thus,
        // only the middle column needs to be considered by each action; the
        // action will handle all three columns at the same time.  All three
        // columns remain in the re-sorted tables below for reference.

        //----------------------------------------------------------------------
        // ----[C A 1] or [X A 1]: C and A present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C A 1        =A          =A          C        | no accum,no Crepl
            // C A 1        =A          =A          delete   | no accum,Crepl
            // C A 1        =C+A        =C+A        C        | accum, no Crepl
            // C A 1        =C+A        =C+A        delete   | accum, Crepl

            // X A 1        undelete    undelete    X        | no accum,no Crepl
            // X A 1        undelete    undelete    X        | no accum,Crepl
            // X A 1        undelete    undelete    X        | accum, no Crepl
            // X A 1        undelete    undelete    X        | accum, Crepl

            // Both C(I(i),J(j)) == S(i,j) and A(i,j) are present, and mij = 1.
            // C(I(i),J(i)) is updated with the entry A(i,j).
            // C_replace has no impact on this action.

            // [X A 1] matrix case
            #define GB_X_A_1_matrix                                         \
            {                                                               \
                /* ----[X A 1]                                           */ \
                /* action: ( undelete ): bring a zombie back to life     */ \
                GB_UNDELETE ;                                               \
                GB_COPY_aij_to_C ;                                          \
            }

            // [X A 1] scalar case
            #define GB_X_A_1_scalar                                         \
            {                                                               \
                /* ----[X A 1]                                           */ \
                /* action: ( undelete ): bring a zombie back to life     */ \
                GB_UNDELETE ;                                               \
                GB_COPY_scalar_to_C ;                                       \
            }

            // [C A 1] scalar case, with accum
            #define GB_C_A_1_accum_matrix                                   \
            {                                                               \
                /* ----[C A 1] with accum, scalar expansion              */ \
                /* action: ( =C+A ): apply the accumulator               */ \
                GB_void ywork [ysize] ;                                     \
                GB_COPY_aij_to_ywork ;                                      \
                GB_ACCUMULATE ;                                             \
            }                                                               \

            // [C A 1] scalar case, with accum
            #define GB_C_A_1_accum_scalar                                   \
            {                                                               \
                /* ----[C A 1] with accum, scalar expansion              */ \
                /* action: ( =C+A ): apply the accumulator               */ \
                GB_ACCUMULATE ;                                             \
            }

            // [C A 1] matrix case when accum is present
            #define GB_withaccum_C_A_1_matrix                               \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_matrix ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] with accum, scalar expansion          */ \
                    /* action: ( =C+A ): apply the accumulator           */ \
                    GB_C_A_1_accum_matrix ;                                 \
                }                                                           \
            }

            // [C A 1] matrix case when no accum is present
            #define GB_noaccum_C_A_1_matrix                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_matrix ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] no accum, scalar expansion            */ \
                    /* action: ( =A ): copy A into C                     */ \
                    GB_COPY_aij_to_C ;                                      \
                }                                                           \
            }

            // [C A 1] scalar case when accum is present
            #define GB_withaccum_C_A_1_scalar                               \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_scalar ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] with accum, scalar expansion          */ \
                    /* action: ( =C+A ): apply the accumulator           */ \
                    GB_C_A_1_accum_scalar ;                                 \
                }                                                           \
            }

            // [C A 1] scalar case when no accum is present
            #define GB_noaccum_C_A_1_scalar                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_scalar ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] no accum, scalar expansion            */ \
                    /* action: ( =A ): copy A into C                     */ \
                    GB_COPY_scalar_to_C ;                                   \
                }                                                           \
            }

        //----------------------------------------------------------------------
        // ----[. A 1]: C not present, A present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // . A 1        insert      insert      .        | no accum,no Crepl
            // . A 1        insert      insert      .        | no accum,Crepl
            // . A 1        insert      insert      .        | accum, no Crepl
            // . A 1        insert      insert      .        | accum, Crepl

            // C(I(i),J(j)) == S (i,j) is not present, A (i,j) is present, and
            // mij = 1. The mask M allows C to be written, but no entry present
            // in C (neither a live entry nor a zombie).  This entry must be
            // added to C but it doesn't fit in the pattern.  It is added as a
            // pending tuple.  Zombies and pending tuples do not intersect.

            // If adding the pending tuple fails, C is cleared entirely.
            // Otherwise the matrix C would be left in an incoherent partial
            // state of computation.  It's cleaner to just free it all.

            #define GB_D_A_1_scalar                                         \
            {                                                               \
                /* ----[. A 1]                                           */ \
                /* action: ( insert )                                    */ \
                GB_INSERT (scalar) ;                                        \
            }

            #define GB_D_A_1_matrix                                         \
            {                                                               \
                /* ----[. A 1]                                           */ \
                /* action: ( insert )                                    */ \
                GB_INSERT (Ax +(pA*asize)) ;                                \
            }

        //----------------------------------------------------------------------
        // ----[C . 1] or [X . 1]: C present, A not present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C . 1        delete      delete      C        | no accum,no Crepl
            // C . 1        delete      delete      delete   | no accum,Crepl
            // C . 1        C           C           C        | accum, no Crepl
            // C . 1        C           C           delete   | accum, Crepl

            // X . 1        X           X           X        | no accum,no Crepl
            // X . 1        X           X           X        | no accum,Crepl
            // X . 1        X           X           X        | accum, no Crepl
            // X . 1        X           X           X        | accum, Crepl

            // C(I(i),J(j)) == S (i,j) is present, A (i,j) not is present, and
            // mij = 1. The mask M allows C to be written, but no entry present
            // in A.  If no accum operator is present, C becomes a zombie.

            // This condition cannot occur if A is a dense matrix,
            // nor for scalar expansion

            // [C . 1] matrix case when no accum is present
            #define GB_noaccum_C_D_1_matrix                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X . 1]                                       */ \
                    /* action: ( X ): still a zombie                     */ \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C . 1] no accum                              */ \
                    /* action: ( delete ): becomes a zombie              */ \
                    GB_DELETE ;                                             \
                }                                                           \
            }

        //----------------------------------------------------------------------
        // ----[C A 0] or [X A 0]: both C and A present but M=0
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C A 0                    C           =A       | no accum,no Crepl
            // C A 0                    delete      =A       | no accum,Crepl
            // C A 0                    C           =C+A     | accum, no Crepl
            // C A 0                    delete      =C+A     | accum, Crepl

            // X A 0                    X           undelete | no accum,no Crepl
            // X A 0                    X           undelete | no accum,Crepl
            // X A 0                    X           undelete | accum, no Crepl
            // X A 0                    X           undelete | accum, Crepl

            // Both C(I(i),J(j)) == S(i,j) and A(i,j) are present, and mij = 0.
            // The mask prevents A being written to C, so A has no effect on
            // the result.  If C_replace is true, however, the entry is
            // deleted, becoming a zombie.  This case does not occur if
            // the mask M is not present.  This action also handles the
            // [C . 0] and [X . 0] cases; see the next section below.

            // This condition can still occur if A is dense, so if a mask M is
            // present, entries can still be deleted from C.  As a result, the
            // fact that A is dense cannot be exploited when the mask M is
            // present.

            #define GB_C_A_0                                                \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 0]                                       */ \
                    /* ----[X . 0]                                       */ \
                    /* action: ( X ): still a zombie                     */ \
                }                                                           \
                else if (C_replace)                                         \
                {                                                           \
                    /* ----[C A 0] replace                               */ \
                    /* ----[C . 0] replace                               */ \
                    /* action: ( delete ): becomes a zombie              */ \
                    GB_DELETE ;                                             \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 0] no replace                            */ \
                    /* ----[C . 0] no replace                            */ \
                    /* action: ( C ): no change                          */ \
                }                                                           \
            }

            // The above action is very similar to C_D_1.  The only difference
            // is how the entry C becomes a zombie.  With C_D_1, there is no
            // entry in A, so C becomes a zombie if no accum function is used
            // because the implicit value A(i,j) gets copied into C, causing it
            // to become an implicit value also (deleting the entry in C).
            // With C_A_0, the entry C is protected from any modification from
            // A (regardless of accum or not).  However, if C_replace is true,
            // the entry is cleared.  The mask M does not protect C from the
            // C_replace action.

        //----------------------------------------------------------------------
        // ----[C . 0] or [X . 0]: C present, A not present, M=0
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------

            // C . 0                    C           delete   | no accum,no Crepl
            // C . 0                    delete      delete   | no accum,Crepl
            // C . 0                    C           C        | accum, no Crepl
            // C . 0                    delete      C        | accum, Crepl

            // X . 0                    X           X        | no accum,no Crepl
            // X . 0                    X           X        | no accum,Crepl
            // X . 0                    X           X        | accum, no Crepl
            // X . 0                    X           X        | accum, Crepl

            // C(I(i),J(j)) == S(i,j) is present, but A(i,j) is not present,
            // and mij = 0.  Since A(i,j) has no effect on the result,
            // this is the same as the C_A_0 action above.

            // This condition cannot occur if A is a dense matrix, nor for
            // scalar expansion, but the existance of the entry A is not
            // relevant.

            #define GB_C_D_0 GB_C_A_0

        //----------------------------------------------------------------------
        // ----[. A 0]: C not present, A present, M=0
        //----------------------------------------------------------------------

            // . A 0                    .           insert   | no accum,no Crepl
            // . A 0                    .           insert   | no accum,no Crepl
            // . A 0                    .           insert   | accum, no Crepl
            // . A 0                    .           insert   | accum, Crepl

            // C(I(i),J(j)) == S(i,j) is not present, A(i,j) is present,
            // but mij = 0.  The mask M prevents A from modifying C, so the
            // A(i,j) entry is ignored.  C_replace has no effect since the
            // entry is already cleared.  There is nothing to do.

        //----------------------------------------------------------------------
        // ----[. . 1] and [. . 0]: no entries in C and A, M = 0 or 1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------

            // . . 1        .           .           .        | no accum,no Crepl
            // . . 1        .           .           .        | no accum,Crepl
            // . . 1        .           .           .        | accum, no Crepl
            // . . 1        .           .           .        | accum, Crepl

            // . . 0        .           .           .        | no accum,no Crepl
            // . . 0        .           .           .        | no accum,Crepl
            // . . 0        .           .           .        | accum, no Crepl
            // . . 0        .           .           .        | accum, Crepl

            // Neither C(I(i),J(j)) == S(i,j) nor A(i,j) are not present,
            // Nothing happens.  The M(i,j) entry is present, otherwise
            // this (i,j) position would not be considered at all.
            // The M(i,j) entry has no effect.  There is nothing to do.

//------------------------------------------------------------------------------
// GB_subassign_method0: C(I,J) = 0 ; using S
//------------------------------------------------------------------------------

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
) ;

//------------------------------------------------------------------------------
// GB_subassign_method1: C(I,J)<M> = scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method1
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
    const GrB_Matrix M,
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method2: C(I,J)<M> += scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method2
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
    const GrB_Matrix M,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method3: C(I,J) += scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method3
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
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method4: C(I,J)<!M> += scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method4
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
    const GrB_Matrix M,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method5: C(I,J) += A ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method5
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
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method6: C(I,J)<#M> += A ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method6
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
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method7: C(I,J) = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method7
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method8: C(I,J) += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method8
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method9: C(I,J) = A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method9
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method10: C(I,J) += A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method10
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method11: C(I,J)<#M> = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method11
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method12: C(I,J)<#M> += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method12
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method13: C(I,J)<#M> = A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method13
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
) ;

//------------------------------------------------------------------------------
// GB_subassign_method14: C(I,J)<#M> += A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_method14
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
) ;

