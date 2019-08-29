function C = gt (A, B)
%A > B
% Element-by-element comparison of A and B.  One or both may be scalars.
% Otherwise, A and B must have the same size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = lt (B, A) ;

