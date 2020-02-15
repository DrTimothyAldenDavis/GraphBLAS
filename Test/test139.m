function test139
%TEST139 merge sort, special cases

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('test139 --------------- merge sort, special cases\n') ;
rng ('default') ;

n = 1e6 ;
I = 42 * ones (n,1) ;
J = (1:n)' ;

I0 = int64 (I) ;
J0 = int64 (I) ;

IJ1 = sortrows ([I0 J0]) ;

[a b] = GB_mex_msort_2 (I0, J0, 2) ;
assert (isequal (IJ1, [a b])) ;

IJ1 = sortrows ([J0 I0]) ;

[a b] = GB_mex_msort_2 (J0, I0, 2) ;
assert (isequal (IJ1, [a b])) ;

fprintf ('test139 --------------- all tests passed\n') ;
