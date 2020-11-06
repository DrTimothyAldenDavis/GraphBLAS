function test173
%TEST173 test GrB_assign C<A>=A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('test173 ------------ GrB_assign C<A>=A\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

m = 10 ;
n = 14 ;

rng ('default') ;

desc.mask = 'structural' ;

for k = 1:length (types)

    ctype = types {k} ;
    fprintf ('%s, ', ctype) ;
    C = GB_spec_random (m, n, 0.5, 100, ctype) ;
    C = GB_spec_matrix (C) ;

    A = GB_spec_random (m, n, 0.5, 100, ctype) ;
    A = GB_spec_matrix (A) ;
    A_nonzero = full (A.matrix ~= 0) ;

    for C_sparsity = 1:15
        C.sparsity = C_sparsity ;

        for A_sparsity = 1:15
            A.sparsity = A_sparsity ;

            % C<A> = A
            C1 = GB_mex_assign_alias_mask (C, A, [ ]) ;
            C2 = full (C.matrix) ;
            C2 (A_nonzero) = full (A.matrix (A_nonzero)) ;
            err = norm (double (C2) - double (C1.matrix), 1) ;
            assert (err == 0) ;

            if (isequal (ctype, 'double'))

                % C<A,struct> = A
                B = A ;
                B.matrix = sparse (B.matrix) ;
                C3 = GB_mex_assign_alias_mask (C, B, desc) ;
                err = norm (double (C2) - double (C3.matrix), 1) ;
                assert (err == 0) ;

            end

        end
    end
end

fprintf ('\ntest173: all tests passed\n') ;

