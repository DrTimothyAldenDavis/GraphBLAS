
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

step 1: build a set of tasks for list (1), with GB_pslice on upper-bound
    counts for nnz (C(:,k))

    Then if any of them are too big,
    slice them and create tasks for list (2).







