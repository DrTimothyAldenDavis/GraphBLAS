function test104
%TEST104 export/import

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default')
fprintf ('\ntest104: export/import tests\n') ;

for m = [0 1 5 100]
    for n = [0 1 5 100]
        if (n == 1)
            fmts = [-8:-1 1:8] ;
        else
            fmts = [1:8] ;
        end
        for d = [0 0.1 0.5 inf]
            A = GB_spec_random (m, n, d) ;
            nz = nnz (A.pattern) ;
            is_sparse = (nz < m*n) ;
            for fmt_matrix = fmts
                fprintf ('.') ;
                for fmt_export = 0:11
                    try
                        C = GB_mex_export_import (A, fmt_matrix, fmt_export) ;
                        GB_spec_compare (C, A) ;
                    catch me
                        % should fail if A is sparse and it is attempted to
                        % be exported as full
                        ok = is_sparse && ...
                            (fmt_export == 6 || fmt_export == 7 || ...
                             fmt_export == -6 || fmt_export == -7) ;
                        assert (ok) ;
                    end
                end
            end
        end
    end
end

fprintf ('\ntest104: all tests passed\n') ;
