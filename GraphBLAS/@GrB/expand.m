function C = expand (scalar, S)
%GRB.EXPAND expand a scalar into a GraphBLAS matrix.
% C = GrB.expand (scalar, S) expands the scalar into a matrix with the
% same size and pattern as S, as C = scalar*spones(S).  C has the same
% type as the scalar.  The numerical values of S are ignored; only the
% pattern of S is used.  The inputs may be either GraphBLAS and/or
% MATLAB matrices/scalars, in any combination.  C is returned as a
% GraphBLAS matrix.
%
% See also GrB.assign.

% TODO allow for a type to be passed in

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% FUTURE: as much as possible, replace scalar expansion with binary
% operators used in a unary apply, when they are added to
% SuiteSparse:GraphBLAS from the v1.3 C API.

if (isobject (scalar))
    % do not use gb_get_scalar, to keep it sparse
    scalar = scalar.opaque ;
    if (~gb_isscalar (scalar))
        gb_error ('input parameter (%s) must be a scalar', inputname (1)) ;
    end
end

if (isobject (S))
    S = S.opaque ;
end

C = GrB (gb_expand (scalar, S)) ;

