//------------------------------------------------------------------------------
// GB_subref_method: select a method for C(:,k) = A(I,j), for one vector of C
//------------------------------------------------------------------------------

static inline int GB_subref_method  // return the method to use (1 to 12)
(
    // output
    int64_t *p_work,                // work required
    bool *p_this_needs_I_inverse,   // true if I needs to be inverted
    // input:
    const int64_t ajnz,             // nnz (A (:,j))
    const int64_t avlen,            // A->vlen
    const int Ikind,                // GB_ALL, GB_RANGE, GB_STRIDE, or GB_LIST
    const int64_t nI,               // length of I
    const bool I_inverse_ok,        // true if I is invertable 
    const bool need_qsort,          // true if C(:,k) requires sorting
    const int64_t iinc,             // increment for GB_STRIDE
    const int64_t nduplicates       // # of duplicates in I (zero if not known) 
)
{

    //--------------------------------------------------------------------------
    // initialize return values
    //--------------------------------------------------------------------------

    int method ;            // determined below
    bool this_needs_I_inverse = false ; // most methods do not need I inverse
    int64_t work ;          // most methods require O(nnz(A(:,j))) work

    //--------------------------------------------------------------------------
    // determine the method to use for C(:,j) = A (I,j)
    //--------------------------------------------------------------------------

    if (ajnz == avlen)
    {
        // A(:,j) is dense
        if (Ikind == GB_ALL)
        {
            // Case 1: C(:,k) = A(:,j) are both dense
            method = 1 ;
            work = nI ;   // ajnz == avlen == nI
        }
        else
        {
            // Case 2: C(:,k) = A(I,j), where A(:,j) is dense,
            // for Ikind == GB_RANGE, GB_STRIDE, or GB_LIST
            method = 2 ;
            work = nI ;
        }
    }
    else if (nI == 1)
    {
        // Case 3: one index
        method = 3 ;
        work = 1 ;
    }
    else if (Ikind == GB_ALL)
    {
        // Case 4: I is ":"
        method = 4 ;
        work = ajnz ;
    }
    else if (Ikind == GB_RANGE)
    {
        // Case 5: C (:,k) = A (ibegin:iend,j)
        method = 5 ;
        work = ajnz ;
    }
    else if (
        (Ikind == GB_LIST && !I_inverse_ok) ||  // must do Case 6
        (64 * nI < ajnz))    // Case 6 faster
    {
        // Case 6: nI not large; binary search of A(:,j) for each i in I
        method = 6 ;
        work = nI * 64 ;
    }
    else if (Ikind == GB_STRIDE)
    {
        if (iinc >= 0)
        {
            // Case 7: I = ibegin:iinc:iend with iinc >= 0
            method = 7 ;
            work = ajnz ;
        }
        else if (iinc < -1)
        {
            // Case 8: I = ibegin:iinc:iend with iinc < =1
            method = 8 ;
            work = ajnz ;
        }
        else // iinc == -1
        {
            // Case 9: I = ibegin:(-1):iend
            method = 9 ;
            work = ajnz ;
        }
    }
    else // Ikind == GB_LIST, and I inverse buckets will be used
    {
        // construct the I inverse buckets
        this_needs_I_inverse = true ;
        if (need_qsort)
        {
            // Case 10: nI large, need qsort
            // duplicates are possible so cjnz > ajnz can hold.  If fine tasks
            // use this method, a post sort is needed when all tasks are done.
            method = 10 ;
            work = ajnz * 32 ;
        }
        else if (nduplicates > 0)
        {
            // Case 11: nI large, no qsort, with duplicates
            // duplicates are possible so cjnz > ajnz can hold.  Note that the
            // # of duplicates is only known after I is inverted, which might
            // not yet be done.  In that case, nuplicates is assumed to be
            // zero, and Case 11 is assumed to be used instead.  This is
            // revised after I is inverted.
            method = 11 ;
            work = ajnz * 2 ;
        }
        else
        {
            // Case 12: nI large, no qsort, no dupl
            method = 12 ;
            work = ajnz ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (p_work != NULL)
    {
        (*p_work) = work ;
    }
    if (p_this_needs_I_inverse != NULL)
    {
        (*p_this_needs_I_inverse) = this_needs_I_inverse ;
    }
    return (method) ;
}

