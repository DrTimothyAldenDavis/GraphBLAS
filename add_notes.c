
GB_add:

use a finer slice.  see GB_ek_slice in GB_selector and GB_reduce_each_vector.

C = A+B.  // what about the mask ?

step1:  compute upper bound of C(:,k) in each vector k.
    cnz (k) = anz (k) + bnz (k)

step2:  for each task:
    find the boundaries of each task, where vectors are not (yet) split.
    use GB_pslice to do this.

step3:  if any task has < 0.5 or more than 1.5 the work it needs, then
    use a finer slice.  Otherwise, discard the GB_pslice results and
    use the slice from step2.

    finer method:

    ilast = 0
    klast = 0

    for task t = 0 to # of tasks

        goal is to find indices i and k so that
        the computation of C(ilast,klast) to C(i,k) has a balanced amount
        of work.

                klast
                    2   4
                . x x - -
                . x x - -
            2   . x x - -
                . x x - 
      ilast 4   . x - -
                . x - -

       assume pA_last and pB_last have been found in A(:,klast) and B(:,klast)

hmmmm ... too gnarly

instead, consider 2 kinds of tasks:

    (1) a task that does one or more whole vectors of C
    (2) a task that does a partial vector of C

build a list of both tasks and do them all in parallel.

step 0: build a set of tasks for list (1), with GB_pslice on upper-bound
    counts for nnz (C(:,k)).  Do this as phase 0.5 (after phase0 and
    before phase1)

    Then if any of them are too big,
    slice them and create subtasks for list (2).

    ideal task size:  Let ntasks_total = 32 * nthreads (say).
    Then each task should do about ideal = nnz(C) / ntasks_total work.

    Let the work in the tasks in list 1 by nwork (0... ntasks1).
    If nwork(t) > 2 * ideal, split it into ceil (nwork (t) / ideal)
    subtasks.

    To slice a single C(:,k):  start with the pattern of A(:,k) and B(:,k).
    Decide how many tasks to create for C(:,k).
    Let cknz = upper bound, or nnz (A (:,k) + B (:,k))

    to split C(:,k) into ntasks:

        for task t = 1 to ntasks-1

            binary search for index i so that nnz (A(0:i-1,k) + B(0,i-1,k))
            is about (cknz/t).  return the pointers into the kth vector of
            both A and B, and the index i.

            let n = vector length of C, A, and B
            ileft = 0
            iright = n-1

            while (   )

                imiddle = (ileft + iright) / 2
                binary search for imiddle in A(:,k) and B(:,k)
                work_left = nnz (A (0:imiddle,k)) + nnz (B(0:imiddle,k))
                if (work_left is within range of cknz/t, plus or minus %)
                    halt binary search
                else if work_left is too low: ileft = imiddle + 1
                else, work_left is too high:  iright = imiddle









