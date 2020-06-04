function C = gb_expand (scalar, S, type)
%GB_EXPAND expand a scalar into a GraphBLAS matrix.
% Implements C = GrB.expand (scalar, S).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin < 3)
    type = gbtype (scalar) ;
end

[m, n] = gbsize (S) ;
desc.mask = 'structure' ;
C = gbassign (gbnew (m, n, type), S, scalar, desc) ;

