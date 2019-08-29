function C = not (A)
%~ logical negation of a GraphBLAS matrix
% C = ~A computes the logical negation of a GraphBLAS matrix A.  The result
% C is dense, and the computation takes O(m*n) time and space, so sparsity
% is not exploited.  To negate just the entries in the pattern of A, use
% C = gb.apply ('~.logical', A), which has the same pattern as A.
%
% See also gb.apply.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.select ('nonzero', gb.apply ('~.logical', full (A))) ;

