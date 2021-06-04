
GrB_Matrix_tryExtractElement (GrB_Scalar s, GrB_Index i, GrB_Index j,
    GrB_Matrix A)

    if A is jumbled
        no call to tryExtractElement can succeed
        return GxB_ALL_FAIL

    look for A(i,j) in the CSC/CSR pattern (binary search in A(i,:) if CSR)

    if A(i,j) is present:
        set the scalar
        time is O(nnz(A(i,:))) if A is CSR
        return GrB_SUCCESS

    if A(i,j) is a zombie:
        it cannot be in any pending tuple
        time is O(nnz(A(i,:))) if A is CSR
        return GrB_NO_VALUE (but also need to return an indication
            that trySetElement would succeed for this entry)

    if A(i,j) is not present in the CSC/CSR pattern:
        if no pending tuples return GrB_NO_VALUE
        if pending tuples exist
            the entry to remove might be a pending tuple; the call to
            GrB_Matrix_extractElement would assemble the pending tuples
            but this is costly so just fail here
            return GxB_FAIL


GrB_Matrix_trySetElement (GrB_Matrix A, GrB_Scalar s, GrB_Index i, GrB_Index j)

    if A is jumbled
        no call to trySetElement can succeed
        return GxB_ALL_FAIL

    look for A(i,j) in the CSC/CSR pattern (binary search in A(i,:) if CSR)

    if A(i,j) is present or a zombie
        set the element; it cannot affected by any pending tuple
        time is O(nnz(A(i,:))) if A is CSR
        return GrB_SUCCESS

    if A(i,j) is not present in the CSC/CSR pattern:
        the entry could be added as a pending tuple, but that should be
        done by GrB_Matrix_setElement instead
        return GxB_FAIL


GrB_Matrix_tryRemoveElement (GrB_Matrix A, GrB_Index i, GrB_Index j)

    if A is jumbled
        no call to tryRemoveElement can succeed
        return GxB_ALL_FAIL

    look for A(i,j) in the CSC/CSR pattern (binary search in A(i,:) if CSR)

    if A(i,j) is found or is found to be a zombie
        make it a zombie (unless it already is a zombie)
        return GrB_SUCCESS

    if no pending tuples
        return GrB_NO_VALUE

    if any pending tuple
        the entry to remove might be a pending tuple; the call to
        GrB_Matrix_removeElement would assemble the pending tuples but this is
        costly so just fail here
        return GxB_FAIL


If these functions return FAIL then the corresponding GrB_ set/remove/extract
Element function could be called instead to do the work.  Some of those would
do a GrB_wait on the matrix.

if these functions return ALL_FAIL, then no call to any GxB_try*Element method
can succeed.  Use GrB_wait first, or use the corresponding GrB_
set/remove/extract Element method instead.


