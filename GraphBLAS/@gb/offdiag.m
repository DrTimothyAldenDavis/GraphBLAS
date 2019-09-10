function C = offdiag (A)
%GB.OFFDIAG removes diaogonal entries from the matrix A
% C = gb.offdiag (A) removes diagonal entries from A.
%
% See also gb/diag, gb.prune, gb.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.select ('offdiag', A) ;

