function C = gb_expand (scalar, S, type)
%GB_EXPAND expand a scalar into a GraphBLAS matrix.
% Implements C = GrB.expand (scalar, S, type).  This function assumes the
% first input is a scalar; the caller has checked this already.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin < 3)
    % the type defaults to the type of the scalar, not S.
    type = gbtype (scalar) ;
else
    % typecast the scalar to the desired type
    scalar = gbnew (scalar, type) ;
end

C = gbapply2 (['1st.' type], scalar, S) ;

