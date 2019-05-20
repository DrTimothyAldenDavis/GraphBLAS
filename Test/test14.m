function test14
%TEST14 test GrB_reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('\ntest14: reduce to column and scalar\n') ;

[~, ~, ~, classes, ~, ~] = GB_spec_opsall ;

rng ('default') ;

m = 8 ;
n = 4 ;
dt = struct ('inp0', 'tran') ;

for k1 = 1:length(classes)
    aclass = classes {k1} ;
    fprintf ('.') ;
    A = GB_spec_random (m, n, 0.3, 100, aclass) ;
    B = GB_spec_random (n, m, 0.3, 100, aclass) ;
    w = GB_spec_random (m, 1, 0.3, 100, aclass) ;
    cin = cast (0, aclass) ;
    mask = GB_random_mask (m, 1, 0.5, true, false) ;

    if (isequal (aclass, 'logical'))
        ops = {'or', 'and', 'xor', 'eq'} ;
    else
        ops = {'min', 'max', 'plus', 'times'} ;
    end

    if (isequal (aclass, 'double'))
        hrange = [0 1] ;
        crange = [0 1] ;
    else
        hrange = 0 ;
        crange = 1 ;
    end

    is_float = isequal (aclass, 'single') || isequal (aclass, 'double') ;
    if (is_float)
        tol = 64 * eps (aclass) ;
    else
        tol = 0 ;
    end

    for A_is_hyper = 0:1
    for A_is_csc   = 0:1

    A.is_csc    = A_is_csc ; A.is_hyper    = A_is_hyper ;
    B.is_csc    = A_is_csc ; B.is_hyper    = A_is_hyper ;

    for k2 = 1:length(ops)
        op = ops {k2} ;

here = 1 ; save gunk w op A tol here
        % no mask
        w1 = GB_spec_reduce_to_vector (w, [], [], op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, [], [], op, A, []) ;
        GB_spec_compare (w1, w2, tol) ;

here = 2 ; save gunk w op A tol here
        % no mask, with accum
        w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, A, []) ;
        GB_spec_compare (w1, w2, tol) ;

here = 3 ; save gunk w op A tol mask here
        % with mask
        w1 = GB_spec_reduce_to_vector (w, mask, [], op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, [], op, A, []) ;
        GB_spec_compare (w1, w2, tol) ;

here = 4 ; save gunk w op A tol mask here
        % with mask and accum
        w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, A, []) ;
        GB_spec_compare (w1, w2, tol) ;

here = 5 ; save gunk w op B dt tol mask here
        % no mask, transpose
        w1 = GB_spec_reduce_to_vector (w, [], [], op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, [], [], op, B, dt) ;
        GB_spec_compare (w1, w2, tol) ;

here = 6 ; save gunk w op B dt tol here
        % no mask, with accum, transpose
        w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, B, dt) ;
        GB_spec_compare (w1, w2, tol) ;

here = 7 ; save gunk w op B dt tol mask here
        % with mask, transpose
        w1 = GB_spec_reduce_to_vector (w, mask, [], op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, [], op, B, dt) ;
        GB_spec_compare (w1, w2, tol) ;

here = 8 ; save gunk w op B dt tol mask here
        % with mask and accum, transpose
        w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, B, dt) ;
        GB_spec_compare (w1, w2, tol) ;

        % GB_spec_reduce_to_scalar always operates column-wise, but GrB_reduce
        % operates in whatever order it is given: by column if CSC or by row if
        % CSR.  The result can vary slightly because of different round off
        % errors.  A_hack causes GB_spec_reduce_to_scalar to operate in the
        % same order as GrB_reduce.

        A_hack = A ;
        if (~A.is_csc && is_float)
            A_hack.matrix = A_hack.matrix' ;
            A_hack.pattern = A_hack.pattern' ;
            A_hack.is_csc = true ;
        end

        % Parallel reduction leads to different roundoff.  So even with A_hack,
        % c1 and c2 can only be compared to within round-off error.

        % to scalar
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_hack) ;
        c2 = GB_mex_reduce_to_scalar  (cin, [ ], op, A) ;
        if (is_float)
            assert (abs (c1-c2) < 4 * eps (A.class) *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end

        % to scalar, with accum
        c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, A_hack) ;
        c2 = GB_mex_reduce_to_scalar  (cin, 'plus', op, A) ;
        if (is_float)
            assert (abs (c1-c2) < 4 * eps (A.class) *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end

    end
    end
    end
end

fprintf ('\ntest14: all tests passed\n') ;

