function gbtest19
%GBTEST19 test mpower

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
A = rand (4) ;
G = gb (A) ;

for k = 0:10
    C1 = A^k ;
    C2 = G^k ;
    err = C1 - C2 ;
    assert (logical (norm (err,1) < 1e-12))
end

fprintf ('gbtest19: all tests passed\n') ;

