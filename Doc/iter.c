//------------------------------------------------------------------------------
// PROPOSED DRAFT GxB_Iterator.  Tim Davis, Dec 29, 2021.
//------------------------------------------------------------------------------

// The purpose of the GxB_Iterator is to provide an object and a set of fast
// methods that support the following user pseudo-code, while at the same time
// preserving the opacity of the underlying GrB_Matrix or GrB_Vector.

/*
    e is a tuple (i,j,aij)

    for all e in the matrix A:
        ...

    for all e in A(i1:i2,:)
        ...

    for all e in A(:,j1:j2)
        ...

    for all e in the vector v
        e is the tuple (i,0,vi)
        ...

    The tuples e would be considered in any order; no order is guaranteed.
    The pseudo code:

        for all e in the matrix A:
            ...

    would act like a lazy extractTuples but be much faster and require no space
    for the arrays I, J, and X:

        extractTuples (I, J, X, A, nvals)
        for p = 0 to nvals-1
            e is the tuple (I [p], J [p], X [p])
            ...

*/

// The matrix A (or a vector v) must not be modified if any iterator is in use
// on the matrix or vector.  Parallel user algorithms are supported by
// providing a well defined, user-visible range to the iterator: p = 0:pmax,
// where the user application can query the pmax value of an interator, and
// the current state p of an iterator.

// p = 0 denotes the first entry e in A, A(i1:i2,:), A(:,j1:j2) or v.
// Advancing to the next entry e increments p, and p=pmax denotes an exhausted
// iterator, where retrieving the next entry e returns GrB_NO_VALUE.  pmax will
// often be nvals of A, A(i1:i2,:), A(:,j1:j2), or v, but it might differ
// depending upon the opaque format of A or v.  For example, in
// SuiteSparse:GraphBLAS, a matrix can be sparse, hypersparse, bitmap or full.
// For three of these formats (sparse/hyper/full), pmax is equal to nvals (A),
// nvals (A(i1:i2,:)), etc, and advancing to the next entry is always just p++.
// For the bitmap format, pmax is not nvals(A), but # of values present if the
// matrix was instead full.  In this case, when advancing to the next entry, p
// can advance by more just p++.  This decision is up to the underlying
// library.

// The iterator position p is user-visible, but does not place any constraints
// on the underlying opaque GrB_Matrix or GrB_Vector.  The GxB_Iterator_*
// methods translate the user-visible position p into whatever internal format
// it is using for the matrix.  The only accomodation is that p might advance
// by more than one for a single entry, and pmax >= nvals (submatrix being
// iterated over).

//------------------------------------------------------------------------------
// GxB_Iterator_new
//------------------------------------------------------------------------------

// Create a new iterator, not attached to any matrix or vector.  The iterator
// is allocated but cannot be used until it is attached to a matrix or vector.

// The advantage of separating the "new" and "attach" method (see below) is
// speed.  An iterator can be rapidly reused, even on the same matrix.  If the
// range of the iterator needs to change, just reattach it to the same matrix
// with the updated range.

GrB_Info GxB_Iterator_new
(
    GxB_Iterator *iterator
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_free
//------------------------------------------------------------------------------

// Free an iterator, detaching it if attached to a matrix or vector.

GrB_Info GxB_Iterator_free
(
    GxB_Iterator *iterator
) ;

//------------------------------------------------------------------------------
// GxB_Matrix_Iterator_attach and GxB_Vector_Iterator_attach
//------------------------------------------------------------------------------

// Attach an iterator to a matrix or vector, detaching it from a prior matrix
// if already attached.  The iterator can change kind.

/* kinds of iterators

    entire matrix or vector:  "for e in A do ..."  e is (i,j,aij)
        any sparsity (sparse/hyper/bitmap/full), any format (row/col)
        iso/non-iso

        Any library should be able to do this with any matrix/vector format.

    range of rows (matrix only):  "for e in A (i1:i2,:) do ..."
        any sparsity (sparse/hyper/bitmap/full);
        bitmap/full: any format (row/col)
        sparse/hyper: only by-row supported
        iso/non-iso

        A library is free to return GrB_NOT_IMPLEMENTED if it does not support
        iteration of A(i1:i2,:).  In particular, SuiteSparse:GraphBLAS will
        return GrB_NOT_IMPLEMENTED if is sparse or hypersparse held by column,
        since it is too hard to iterate over the rows of a CSC matrix.

    range of cols (matrix only):  "for e in A (:,j1:j2) do... "
        transpose of the above

        A library is free to return GrB_NOT_IMPLEMENTED if it does not support
        iteration of A(:,j1:j2).  In particular, SuiteSparse:GraphBLAS will
        return GrB_NOT_IMPLEMENTED if is sparse or hypersparse held by row,
        since it is too hard to iterate over the columns of a CSR matrix.

    e is (i,j,aij) or (i,0,vi) for a vector

    The iterator position starts at p = 0, which may refer to a position
    inside a matrix, for the row/col iterator.

    The matrix A must not be modified in any way if any iterator is attached to
    it.  Results are undefined if an iterator is used that has been attached to
    a matrix that then subsequently modified.  This rule could perhaps be
    relaxed in the future if the values of A are modified but not its
    structure, but this would require atomics if multiple threads did this at
    the same time.
*/

GrB_Info GxB_Matrix_Iterator_attach
(
    // input/output
    GxB_Iterator iterator      // must already be allocated on input
    // input
    GrB_Matrix A,               // never modified by any iterator method
    GxB_Format_Value format,    // GxB_BY_ROW, GxB_BY_COL, GxB_BY_ENTRY (-1)
    GrB_Index k1,               // rows k1:k2, cols k1:k2, or not used if entry
    GrB_Index k2,
    GrB_Descriptor desc         // only needed for # of threads for GrB_wait (A)
) ;

GrB_Info GxB_Vector_Iterator_attach
(
    // input/output
    GxB_Iterator iterator      // must already be allocated on input
    // input
    GrB_Vector v,               // only by-entry iterator supported;
                                // never modified by any iterator method.
    GrB_Descriptor desc         // only needed for # of threads for GrB_wait (v)
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_detach
//------------------------------------------------------------------------------

// Detach an iterator from its current matrix or vector, does nothing if the
// iterator is not attached to any matrix/vector.  After all iterators are
// detached from a matrix A or vector v, it is then safe to modify A or v.
//
// A detached iterator is not freed.  It acts just like a new iterator from
// GxB_Iterator_new.
//
// The matrix A or vector v is not modified at all by any iterator method.
// It is up to the user application to ensure all iterators are detached before
// modifying the matrix or vector.

GrB_Info GxB_Iterator_detach
(
    // input/output
    GxB_Iterator iterator
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_get
//------------------------------------------------------------------------------

// Get current entry e from a matrix/vector iterator.  The iterator is not
// advanced. If (I,J,X) were the tuples from extractTuples, this would be
// analogus to:
//
//      e = { I [p], J [p], X [p] } ;
//
// Except that p = 0:pmax-1 may have gaps if the matrix is in bitmap format,
// and p = 0 need not refer to the first tuple in the entire matrix A.  p=0
// denotes the first tuple in the whole matrix A (if by entry), or in the
// submatrix A(i1:i2,:) if by row, or A(:,j1:j2) if by column.
//
// Returns GrB_NO_VALUE if p is at pmax; this is not an error condition.
// This denotes an exhausted iterator.
//
// This might best be written as a static inline function in GraphBLAS.h,
// or even a macro to allow for fast typecasting (instead of calling a typecast
// function).
//
// Should typecasting be allowed?  This slows down the iterator, unless done as
// a macro.

GrB_Info GxB_Iterator_get_TYPE      // TYPE is BOOL, INT8, etc, ..., UDT
(
    // output
    GrB_Index *i,       // row index of current entry e
    GrB_Index *j,       // column index of current entry e, or zero if vector
    TYPE *x,            // value of current entry (bool, int8_t, ...)
    // input
    GxB_Iterator iterator   // not modified
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_next
//------------------------------------------------------------------------------

// Advance the iterator to the next entry.  This is analogous to p++, and will
// be just that in many cases.  The bitmap case requires p to advance farther.
//
// Returns GrB_NO_VALUE if the updated p has reached pmax; this is not an error
// condition.  This denotes an exhausted iterator.
//
// This might best be written as a static inline function in GraphBLAS.h.
// Most of the time it would be just "p++", except for the bitmap case, and
// and check if the row/column has advanced.  The special cases (like bitmap)
// could call a non-static-inline function, since it might be a bit
// complicated.

GrB_Info GxB_Iterator_next
(
    // input/output
    GxB_Iterator iterator
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_peek
//------------------------------------------------------------------------------

// Return the current position of the iterator.
//
// This could be written as a static inline function in GraphBLAS.h, but it
// would not be a commonly-used method so a standard function might be fine.

GrB_Info GxB_Iterator_peek
(
    // output
    GrB_Index *p,       // current position of the iterator, in range 0 to pmax
    // input
    GxB_Iterator iterator   // not modified
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_seek
//------------------------------------------------------------------------------

// Change iterator to position p, which must be in range 0 to pmax.
// p = 0 is the initial state of an iterator when it is first attached.
// p = pmax denotes an exhausted iterator.
//
// This could be written as a static inline function in GraphBLAS.h, but it
// would not be a commonly-used method so a standard function might be fine.
//
// The range p = 0:pmax is provided to the user application with a well-defined
// meaning, so that multiple user threads can cooperate when accessing the same
// matrix or vector.  An entire matrix (or submatrix A(i1:i2,:) or A(:,j1:j2))
// can be accessed in parallel with a set of iterators, one per user thread, if
// each thread seeks to its own assigned subrange of 0:pmax.  User threads can
// also read the same set of entries, as they like.

GrB_Info GxB_Iterator_seek
(
    // input
    GrB_Index p,        // new position for the iterator
    // input/output
    GxB_Iterator iterator
) ;

//------------------------------------------------------------------------------
// GxB_Iterator_query
//------------------------------------------------------------------------------

// Query information about an iterator.  This information is not modified by
// the next/seek methods, but is defined when GxB_Iterator_attach is performed.
// This information is destroyed by GxB_Iterator_detach or GxB_Iterator_free.
// Otherwise, it never changes.

GrB_Info GxB_Iterator_query
(
    // output: lots of stuff, break this into separate methods?
    GrB_Index *pmax,        // iterator range; p is in range 0:pmax.
    GrB_Index *nvals,       // # of entries in iterator range; equal to pmax
                            // for sparse/hyper/full, pmax >= nvals for bitmap.
                            // This is costly to compute if for a bitmap matrix
                            // if the format is not 'by entry', but it would
                            // be computed just once, on the first query.
    GxB_Format_Value *format,   // by row, by col, or by entry
    GrB_Index *k1,          // row/column range of the iterator
    GrB_Index *k2,
    char *type_name,        // name of the type of the matrix (char array of
                            // size at least GxB_MAX_NAME_LEN, owned by the
                            // user application); see also GxB_Matrix_type_name.
    size_t *type_size,      // size of the matrix type
    GrB_Index *nrows,       // matrix dimension or vector length
    GrB_Index *ncols,       // matrix dimension; 1 if vector
    // input:
    GxB_Iterator iterator   // not modified
) ;

//------------------------------------------------------------------------------
// Example:
//------------------------------------------------------------------------------

// for all e in A do ... would be as follows, assuming A is GrB_FP64:

/*
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    GxB_Matrix_Iterator_attach (iterator, A, GxB_BY_ENTRY, 0, 0, NULL) ;
    while (true)
    {
        GrB_Index i, j ;
        double aij ;
        info result = GxB_Iterator_get_FP64 (&i, &j, &aij, iterator) ;
        if (result == GrB_NO_VALUE) break ;
        ... do something with the tuple (i,j,aij) ...
        GxB_Iterator_next (iterator) ;
    }
    GxB_Matrix_Iterator_free (&iterator) ;
*/

// Note that GxB_Iterator_get_TYPE and GxB_Iterator_Next are in the tight
// inner loop, and so they must be very fast (GrB_Matrix_extractElement is
// slow since it cannot assume anything about the sequence of access; it often
// must use a binary search to get a single entry).  Thus, it might be best if
// at least these 2 methods can be implemented as static inline methods or
// even macros.

// for all e in A(i1:i2,:) would be identical, except this line:
//
//      GxB_Matrix_Iterator_attach (iterator, A, GxB_BY_ROW, i1, i2, NULL) ;
//
// but a check would be required, in case the library does not support iterator
// access of the matrix by-row.  This would work in SS:GrB if the matrix is in
// its default format (GxB_BY_ROW), or if A were bitmap/full (those sparsity
// data structures can be accessed easily by both row and column).



//------------------------------------------------------------------------------
// ... TLDR;  ...  the GB_Iterator_opaque struct
//------------------------------------------------------------------------------

// The gory details: this is SS:GrB specific.  Stop reading here if your
// eyes are already glazed over :-).

// The contents of this struct are opaque, but this struct may need to appear
// in GraphBLAS.h itself, so that the iterator functions can be written as fast
// static inline methods.  The contents of the GB_Iterator_opaque struct must
// not be accessed by the user application, even if this content appears in
// GraphBLAS.h.  The advantage is performance; the downside is that
// SuiteSpars:GraphBLAS would have to bump its major version number if this
// struct were to change.

// I will first write this without placing this struct in GraphBLAS.h, and by
// not using any static inline methods or macros.  I'll then experiment with
// the iterator methods to see if there is a performance improvement to
// defining some of the iterator methods as static inline.

struct GB_Iterator_opaque
{
    int64_t magic ;         // to detect invalid objects
    size_t header_size ;    // # of bytes in the struct

    // iterator range: 
    int64_t k1, k2, pmax ;
    GxB_Format_Value format ;
    int64_t pstart ;    // position in the matrix/vector where p = 0

    // current position of the iterator, updated by seek/next:
    int64_t p ;         // current position, starts at p = 0, ends at p = pmax
    int64_t k ;         // current vector (row or column)

    // rest of the information is copied from the attached matrix/vector:  this
    // is copied here to speed up access to the matrix/vector contents.
    int sparsity ;      // GxB_SPARSE, GxB_HYPERSPARSE, GxB_BITMAP, or GxB_FULL

    GrB_Type type ;
    size_t type_size ;

    int64_t plen ;      // size of A->p and A->h
    int64_t vlen ;      // length of each sparse vector
    int64_t vdim ;      // number of vectors in the matrix
    int64_t nvec ;      // number of non-empty vectors for hypersparse form,
                        // or same as vdim otherwise.  nvec <= plen.
                        // some of these vectors in Ah may actually be empty.

    int64_t nvals ;     // nvals if A is bitmap, or -1 if not yet known,
                        // for all of A if format is GxB_BY_ENTRY,
                        // A(k1:k2,:) if by row, A(:,k1:k2) if by col.
                        // only needed for bitmap format.

    int64_t *Ah ;       // list of non-empty vectors
    int64_t *Ap ;       // pointers
    int64_t *Ai ;       // indices
    void *Ax ;          // values
    int8_t *Ab ;        // bitmap

    bool is_csc ;       // true if stored by column, false if by row
    bool iso ;          // true if all entries have the same value
}

typedef struct GB_Iterator_opaque *GxB_Iterator ;

