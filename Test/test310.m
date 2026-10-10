function test310
%TEST310 test GxB_Matrix_reshape

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_test52 ;

%{
rng ('default') ;
n1 = 2^28 ;
nz = 1000 ;
d = nz / (n1*n1) ;
A = sprand (n1, n1, d) ;

m2 = 2^24 ;
n2 = 2^32 ;
GB_mex_burble (0) ;

for inplace = 0:1
    for bycol = 0:1
        C = GB_mex_reshape (A, m2, n2, bycol, inplace) ;
        fprintf ('bycol: %d, inplace: %d\n', bycol, inplace) ;
        C2 = GB_mex_reshape (C, n1, n1, bycol, inplace) ;
        fprintf ('bycol: %d, inplace: %d\n', bycol, inplace) ;
        assert (isequal (A, C2.matrix)) ;
        clear C C2
    end
end
%}

fprintf ('test310: reshape tests passed\n') ;

