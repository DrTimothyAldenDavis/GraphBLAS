function test284
%TEST282 test GrB_mxm using the (MIN,SECONDI1) semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest284: GrB_mxm with (min,secondi1)\n') ;

rng ('default') ;

n = 5 ;
A = GB_spec_random (n, n, 0.3, 100, 'double') ;
B = GB_spec_random (n, n, 0.3, 100, 'double') ;
D = speye (n) ;

for A_is_csc = 0:1
    A.is_csc = A_is_csc ;

    for B_is_csc = 0:1
        B.is_csc = A_is_csc ;

        for C_is_csc = 0:1

            for A_sparsity = [1 2 4]
                if (A_sparsity == 0)
                    A.is_hyper = 0 ;
                    A.is_bitmap = 0 ;
                    A.sparsity = 2 ;    % sparse
                elseif (A_sparsity == 1)
                    A.is_hyper = 1 ;
                    A.is_bitmap = 0 ;
                    A.sparsity = 1 ;    % hypersparse
                else
                    A.is_hyper = 0 ;
                    A.is_bitmap = 1 ;
                    A.sparsity = 4 ;    % bitmap
                end

                for B_sparsity = [1 2 4]
                    if (B_sparsity == 0)
                        B.is_hyper = 0 ;
                        B.is_bitmap = 0 ;
                        B.sparsity = 2 ;    % sparse
                    elseif (B_sparsity == 1)
                        B.is_hyper = 1 ;
                        B.is_bitmap = 0 ;
                        B.sparsity = 1 ;    % hypersparse
                    else
                        B.is_hyper = 0 ;
                        B.is_bitmap = 1 ;
                        B.sparsity = 4 ;    % bitmap
                    end

                    for at = 0:1
                        for bt = 0:1
                            for method = [0 7081 7083 7084 7085]
                                % C = A*B, A'*B, A*B', or A'*B'
                                C1 = GB_mex_AxB_idx (A, B, at, bt, ...
                                    method, C_is_csc, 1) ;
                                C2 = GB_mex_AxB_idx (A, B, at, bt, ...
                                    method, C_is_csc, 0) ;
                                GB_spec_compare (C1, C2) ;
                            end
                        end
                    end

                    for at = 0:1
                        % C = A*D, A'*D
                        C1 = GB_mex_AxB_idx (A, D, at, 0, 0, C_is_csc, 1) ;
                        C2 = GB_mex_AxB_idx (A, D, at, 0, 0, C_is_csc, 0) ;
                        GB_spec_compare (C1, C2) ;
                    end

                    for bt = 0:1
                        % C = D*B, D*B'
                        C1 = GB_mex_AxB_idx (D, B, 0, bt, 0, C_is_csc, 1) ;
                        C2 = GB_mex_AxB_idx (D, B, 0, bt, 0, C_is_csc, 0) ;
                        GB_spec_compare (C1, C2) ;
                    end

                    fprintf ('.') ;
                end
            end
        end
    end
end

fprintf ('\ntest284: all tests passed\n') ;

