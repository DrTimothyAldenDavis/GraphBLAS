function C = minus (A, B)
%MINUS sparse matrix subtraction, C = A-B.
% A and B can be GraphBLAS matrices or MATLAB sparse or full matrices, in
% any combination.  If A and B are matrices, the pattern of C is the set
% union of A and B.  If one of A or B is a scalar, the scalar is expanded
% into a dense matrix the size of the other matrix, and the result is a
% dense matrix.  If the type of A and B differ, the type of A is used, as:
% C = A + gb (B, gb.type (A)).
%
% See also gb.eadd, plus, uminus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = A + (-B) ;

