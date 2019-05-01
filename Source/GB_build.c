//------------------------------------------------------------------------------
// GB_build: build a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// CALLED BY: GB_user_build and GB_reduce_to_column
// CALLS:     GB_builder

// GB_user_build constructs a GrB_Matrix or GrB_Vector from the tuples provided
// by the user.  In that case, the tuples must be checked for duplicates.  They
// might be sorted on input, so this condition is checked and exploited if
// found.  GB_reduce_to_column constructs a GrB_Vector froma GrB_Matrix, by
// discarding the column index.  As a result, duplicates are likely to appear,
// and the input is likely to be unsorted.  The sorted condition is checked
// here.  The duplicates are found in GB_builder.

// Construct a matrix C from a list of indices and values.  Any duplicate
// entries with identical indices are assembled using the binary dup operator
// provided on input.  All three types (x,y,z for z=dup(x,y)) must be
// identical.  The types of dup, S, and C must all be compatible.

// Duplicates are assembled using T(i,j) = dup (T (i,j), S (k)) into a
// temporary matrix T that has the same type as the dup operator.  The
// GraphBLAS spec requires dup to be associative so that entries can be
// assembled in any order.  There is no way to check this condition if dup is a
// user-defined operator.  It could be checked for built-in operators, but the
// GraphBLAS spec does not require this condition to cause an error so that is
// not done here.  If dup is not associative, the GraphBLAS spec states that
// the results are not defined.

// SuiteSparse:GraphBLAS provides a well-defined order of assembly, however.
// Entries in [I,J,S] are first sorted in increasing order of row and column
// index via a stable sort, with ties broken by the position of the tuple in
// the [I,J,S] list.  If duplicates appear, they are assembled in the order
// they appear in the [I,J,S] input.  That is, if the same indices i and j
// appear in positions k1, k2, k3, and k4 in [I,J,S], where k1 < k2 < k3 < k4,
// then the following operations will occur in order:

//      T (i,j) = S (k1) ;

//      T (i,j) = dup (T (i,j), S (k2)) ;

//      T (i,j) = dup (T (i,j), S (k3)) ;

//      T (i,j) = dup (T (i,j), S (k4)) ;

// This is a well-defined order but the user should not depend upon it since
// the GraphBLAS spec does not require this ordering.  Results may differ in
// different implementations of GraphBLAS.

// However, with this well-defined order, the SECOND operator will result in
// the last tuple overwriting the earlier ones.  This is relied upon internally
// by GB_wait.

// After the matrix T is assembled, it is typecasted into the type of C, the
// final output matrix.  No typecasting is done during assembly of duplicates,
// since mixing the two can break associativity and lead to unpredictable
// results.  Note that this is not the case for GB_wait, which must typecast
// each tuple into its output matrix in the same order they are seen in
// the [I,J,S] pending tuples.

// On input, C must not be NULL.  C->type, C->vlen, C->vdim and C->is_csc must
// be valid on input and are unchanged on output.  C must not have any existing
// entries on input (GrB_*_nvals (C) must return zero, per the specification).
// However, all existing content in C is freed.

// The list of numerical values is given by the void * S array and a type code,
// scode.  The latter is defined by the actual C type of the S parameter in
// the user-callable functions.  However, for user-defined types, there is no
// way of knowing that the S array has the same type as dup or C, since in that
// case S is just a void * pointer.  Behavior is undefined if the user breaks
// this condition.

// C is returned as hypersparse or non-hypersparse, depending on the number of
// non-empty vectors of C.  If C has very few non-empty vectors, then it is
// returned as hypersparse.  Only if the number of non-empty vectors is
// Omega(n) is C returned as non-hypersparse, which implies nvals is Omega(n),
// where n = # of columns of C if CSC, or # of rows if CSR.  As a result, the
// time taken by this function is just O(nvals*log(nvals)), regardless of what
// format C is returned in.

// If nvals == 0, I_in, J_in, and S may be NULL.

// PARALLEL: done.
// checks I and J fully in parallel.  Remaining work is done in GB_builder.

#include "GB.h"

GrB_Info GB_build               // build matrix
(
    GrB_Matrix C,               // matrix to build
    const GrB_Index *I_in,      // row indices of tuples
    const GrB_Index *J_in,      // col indices of tuples
    const void *S,              // array of values of tuples
    const GrB_Index nvals,      // number of tuples
    const GrB_BinaryOp dup,     // binary function to assemble duplicates
    const GB_Type_code scode,   // GB_Type_code of S array
    const bool is_matrix,       // true if C is a matrix, false if GrB_Vector
    const bool ijcheck,         // true if I and J are to be checked
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    GrB_Type ctype = C->type ;
    int64_t vlen = C->vlen ;
    int64_t vdim = C->vdim ;
    bool C_is_csc = C->is_csc ;
    int64_t nrows = GB_NROWS (C) ;
    int64_t ncols = GB_NCOLS (C) ;

    //--------------------------------------------------------------------------
    // free all content of C
    //--------------------------------------------------------------------------

    // the type, dimensions, and hyper ratio are still preserved in C.
    GB_PHIX_FREE (C) ;
    ASSERT (GB_EMPTY (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (C->magic == GB_MAGIC2) ;

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format
    //--------------------------------------------------------------------------

    int64_t *I, *J ;
    if (C_is_csc)
    { 
        // C can be a CSC GrB_Matrix, or a GrB_Vector.
        // If C is a typecasted GrB_Vector, then J_in and J must both be NULL.
        I = (int64_t *) I_in ;  // indices in the range 0 to vlen-1
        J = (int64_t *) J_in ;  // indices in the range 0 to vdim-1
    }
    else
    { 
        // C can only be a CSR GrB_Matrix
        I = (int64_t *) J_in ;  // indices in the range 0 to vlen-1
        J = (int64_t *) I_in ;  // indices in the range 0 to vdim-1
    }

    // J contains vector names and I contains indices into those vectors.
    // The rest of this function is agnostic to the CSR/CSC format.

    //--------------------------------------------------------------------------
    // create T
    //--------------------------------------------------------------------------

    GrB_Matrix T = NULL ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // allocate workspace to load and sort the index tuples:

    // vdim <= 1: iwork and kwork for (i,k) tuples, where i = I(k)
    // vdim > 1: also jwork for (j,i,k) tuples where i = I(k) and j = J (k).

    // The k value in the tuple gives the position in the original set of
    // tuples: I[k] and S[k] when vdim <= 1, and also J[k] for matrices with
    // vdim > 1.

    // The workspace iwork and jwork are allocated here but freed (or
    // transplanted) inside GB_builder.  kwork is allocated, used, and freed
    // in GB_builder.

    GB_MALLOC_MEMORY (int64_t *iwork, nvals, sizeof (int64_t)) ;
    bool ok = (iwork != NULL) ;
    int64_t *jwork = NULL ;
    if (vdim > 1)
    { 
        GB_MALLOC_MEMORY (jwork, nvals, sizeof (int64_t)) ;
        ok = ok && (jwork != NULL) ;
    }

    if (!ok)
    { 
        // out of memory
        GB_FREE_MEMORY (iwork, nvals, sizeof (int64_t)) ;
        GB_FREE_MEMORY (jwork, nvals, sizeof (int64_t)) ;
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // create the tuples to sort, and check for any invalid indices
    //--------------------------------------------------------------------------

    bool known_sorted = true ;
    bool no_duplicates_found = true ;

    if (nvals == 0)
    { 

        //----------------------------------------------------------------------
        // nothing to do
        //----------------------------------------------------------------------

        ;

    }
    else if (is_matrix)
    {

        //----------------------------------------------------------------------
        // C is a matrix; check both I and J
        //----------------------------------------------------------------------

        // but if vdim <= 1, do not create jwork
        ASSERT (J != NULL) ;
        ASSERT (iwork != NULL) ;
        ASSERT ((vdim > 1) == (jwork != NULL)) ;
        ASSERT (I != NULL) ;
        int64_t kbad [nthreads] ;

        #pragma omp parallel for num_threads(nthreads) schedule(static) \
            reduction(&&:known_sorted)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            // each thread checks its own part
            int64_t kstart, kend ;
            GB_PARTITION (kstart, kend, nvals, tid, nthreads) ;
            kbad [tid] = -1 ;
            int64_t ilast = (kstart == 0) ? -1 : I [kstart-1] ;
            int64_t jlast = (kstart == 0) ? -1 : J [kstart-1] ;
            for (int64_t k = kstart ; k < kend ; k++)
            {
                // get kth index from user input: (i,j)
                int64_t i = I [k] ;
                int64_t j = J [k] ;
                if (i < 0 || i >= vlen || j < 0 || j >= vdim)
                { 
                    // halt if out of bounds
                    kbad [tid] = k ;
                    break ;
                }

                // check if the tuples are already sorted
                known_sorted = known_sorted &&
                    ((jlast < j) || (jlast == j && ilast <= i)) ;

                // check if this entry is a duplicate of the one just before it
                if (jlast == j && ilast == i) no_duplicates_found = false ;

                // copy the tuple into the work arrays to be sorted
                iwork [k] = i ;
                if (jwork != NULL) jwork [k] = j ;
                // log the last index seen
                ilast = i ; jlast = j ;
            }
        }

        // collect the report from each thread
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            if (kbad [tid] >= 0)
            { 
                // invalid index
                GB_FREE_MEMORY (iwork, nvals, sizeof (int64_t)) ;
                GB_FREE_MEMORY (jwork, nvals, sizeof (int64_t)) ;
                int64_t i = I [kbad [tid]] ;
                int64_t j = J [kbad [tid]] ;
                int64_t row = C_is_csc ? i : j ;
                int64_t col = C_is_csc ? j : i ;
                return (GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS, (GB_LOG,
                    "index ("GBd","GBd") out of bounds,"
                    " must be < ("GBd", "GBd")", row, col, nrows, ncols))) ;
            }
        }

    }
    else if (ijcheck)
    {

        //----------------------------------------------------------------------
        // C is a typecasted GrB_Vector; check only I
        //----------------------------------------------------------------------

        ASSERT (I != NULL) ;
        int64_t kbad [nthreads] ;

        #pragma omp parallel for num_threads(nthreads) schedule(static) \
            reduction(&&:known_sorted)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            // each thread checks its own part
            int64_t kstart, kend ;
            GB_PARTITION (kstart, kend, nvals, tid, nthreads) ;
            kbad [tid] = -1 ;
            int64_t ilast = (kstart == 0) ? -1 : I [kstart-1] ;

            for (int64_t k = kstart ; k < kend ; k++)
            {
                // get kth index from user input: (i)
                int64_t i = I [k] ;

                if (i < 0 || i >= vlen)
                { 
                    // halt if out of bounds
                    kbad [tid] = k ;
                    break ;
                }

                // check if the tuples are already sorted
                known_sorted = known_sorted && (ilast <= i) ;

                // check if this entry is a duplicate of the one just before it
                if (ilast == i) no_duplicates_found = false ;

                // copy the tuple into the work arrays to be sorted
                iwork [k] = i ;

                // log the last index seen
                ilast = i ;
            }
        }

        // collect the report from each thread
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            if (kbad [tid] >= 0)
            { 
                // invalid index
                GB_FREE_MEMORY (iwork, nvals, sizeof (int64_t)) ;
                GB_FREE_MEMORY (jwork, nvals, sizeof (int64_t)) ;
                int64_t i = I [kbad [tid]] ;
                return (GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS, (GB_LOG,
                    "index ("GBd") out of bounds, must be < ("GBd")",
                    i, vlen))) ;
            }
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // GB_reduce_to_column: do not check I, assume not sorted
        //----------------------------------------------------------------------

        // Many duplicates are possible, since the tuples are being used to
        // construct a single vector.  For a CSC format, each entry A(i,j)
        // becomes an (i,aij) tuple, with the column index j discarded.  All
        // entries in a single row i are reduced to a single entry in the
        // vector.  The input is unlikely to be sorted, so don't bother to
        // check.

        GB_memcpy (iwork, I, nvals * sizeof (int64_t), nthreads) ;
        known_sorted = false ;
    }

    //--------------------------------------------------------------------------
    // determine if duplicates are possible
    //--------------------------------------------------------------------------

    // The input is now known to be sorted, or not.  If it is sorted, and if no
    // duplicates were found, then it is known to have no duplicates.
    // Otherwise, duplicates might appear, but a sort is required first to
    // check for duplicates.

    bool known_no_duplicates = known_sorted && no_duplicates_found ;

    //--------------------------------------------------------------------------
    // build the matrix T and transplant it into C
    //--------------------------------------------------------------------------

    // If successful, GB_builder will transplant iwork into its output matrix T
    // as the row indices T->i and set iwork to NULL, or if it fails it has
    // freed iwork.  In either case iwork is NULL when GB_builder returns.  It
    // always frees jwork and sets it to NULL.  T can be non-hypersparse or
    // hypersparse, as determined by GB_builder; it will typically be
    // hypersparse.  Its type is the same as the z output of the z=dup(x,y)
    // operator.

    GrB_Info info = GB_builder (&T,     // create T
        dup->ztype,     // T has the type determined by the dup operator
        vlen,           // T->vlen = C->vlen
        vdim,           // T->vdim = C->vdim
        C_is_csc,       // T has the same CSR/CSC format as C
        &iwork,         // iwork_handle, becomes T->i on output
        &jwork,         // jwork_handle, freed on output
        known_sorted,        // tuples may or may not be sorted, as found above
        known_no_duplicates, // tuples might have duplicates: need to check
        S,              // original array of values, of type scode, size nvals
        nvals,          // number of tuples
        nvals,          // size of iwork, jwork, and S
        dup,            // operator to assemble duplicates
        scode,          // type of the S array
        Context) ;

    ASSERT (iwork == NULL) ;
    ASSERT (jwork == NULL) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // transplant and typecast T into C, conform C, and free T
    //--------------------------------------------------------------------------

    return (GB_transplant_conform (C, ctype, &T, Context)) ;
}

