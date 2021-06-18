% function test209
%TEST209 test iso build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[~, ~, ~, types ~, ~,] = GB_spec_opsall ;
types = types.all ;
op.opname = 'second' ;

% GrB.burble (1) ;

for prob = 1:2

    if (prob == 1)
        n = 100 ;
        m = 200 ;
        nnz = 1000 ;
    else
        n = 100 ;
        m = 200 ;
        nnz = 65000 ;
    end
    fprintf ('\nm: %d n: %d nnz: %d\n', m, n, nnz) ;

    I = irand (0, m-1, nnz, 1) ;
    J = irand (0, n-1, nnz, 1) ;
    Y = 10 * rand (nnz, 1) ;

    for k = 1:length(types)
        type = types {k} ;
        fprintf ('%s ', type) ;
        X = GB_mex_cast (Y, type) ;
        Z = X (1) ;
        X (:) = Z ;
        op.optype = type ;

        % non-iso matrix build
        C1 = GB_mex_Matrix_build (I, J, X, m, n, op, type) ;
        C2 = GB_spec_build (I, J, X, m, n, op) ;
        GB_spec_compare (C1, C2) ;

        % iso matrix build
        C1 = GB_mex_Matrix_build (I, J, Z, m, n, op, type) ;
        C2 = GB_spec_build (I, J, X, m, n, op) ;
        GB_spec_compare (C1, C2) ;

        % non-iso vector build
        C1 = GB_mex_Vector_build (I, X, m, op, type) ;
        C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
        GB_spec_compare (C1, C2) ;

        % iso vector build
        C1 = GB_mex_Vector_build (I, Z, m, op, type) ;
        C2 = GB_spec_build (I, [ ], X, m, 1, op) ;
        GB_spec_compare (C1, C2) ;

    end
end

GrB.burble (0) ;
fprintf ('\ntest209: all tests passed\n') ;

