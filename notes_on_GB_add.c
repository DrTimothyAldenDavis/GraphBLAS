

C = A+B
C<M> = A+B
C<!M> = A+B


phase 0:

    time take is at most O(n), if the matrices are not hypersparse.
    ideal is O (A->nvec + B->nvec + M->nvec)

    determine the non-empty columns of C, and their positions in M, A, B

    compute the following for k = 0 ... cnvec-1:
        C_to_A [k] = ka if kth column of C comes from ka-th col of A
        C_to_B [k] = kb if kth column of C comes from ka-th col of B

    if (M == NULL or M is present but complemented)
    {
        // GB_set_union2 (C_list, NULL, A, B)
        columns of C to compute = set union of non empty colums of A and B
    }
    else if M is not complemented
    {
        // GB_set_union2 (C_list, M, A, B)
        columns of C to compute = (set union of non empty colums of A, B)
            intersected with non-empty columns of M
    }

    cnvec = # columns of C to compute
    now C_to_A [0...cnvec-1]
        C_to_B [0...cnvec-1]
    gives the connection of columns of C to A and B

    also decide if C should be hypersparse

    in O(1) time: if all matrices are not hypersparse, and if
        they have many non-empty columns (>nvec_nonempty),
        then let cnvec=n, and C_to_* is NULL; implicit.

phase 1:

    for k = 0 to cnvec-1 in parallel
        count the # of entries in each column of C, doing all 3 cases:
            C = A+B, C<M> = A+B, C<!M> = A+B

    Cp = cumsum (C_count)

phase 2:

    do the numerical work (all the work above is symbolic)

    for k = 0 to cnvec-1 in parallel
        compute entries in each column of C, doing:
            C = A+B, C<M> = A+B, C<!M> = A+B

if one thread: phase 1 and 2 can be combined

//----------------------------------------------------------------------------
// GB_emult
//----------------------------------------------------------------------------

C = A.*B
C<M> = A.*B
C<!M> = A.*B

phase 0:

    determine the non-empty columns of C, and their positions in M, A, B

    if (M == NULL or M is present but complemented)
    {
        columns of C to compute = set intersection of non empty cols of A and B
    }
    else if M is not complemented
    {
        columns of C to compute = 
            intersection of nonempty cols of (A, B, M)
    }

phase 1 and 2: similar to the above; inner part differs

//----------------------------------------------------------------------------

