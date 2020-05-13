function C = bitor (A, B, assumedtype)
%BITOR bitwise OR.
% C = bitor (A,B) computes the bitwise OR of A and B.  If A and B are
% matrices, the pattern of C is the set union of A and B.  If one of A or
% B is a nonzero scalar, the scalar is expanded into a dense matrix the
% size of the other matrix, and the result is a dense matrix.
%
% With a third parameter, C = bitor (A,B,assumedtype) provides a data
% type to convert A and B to if they are floating-point types.  If A or B
% already have integer types, then they are not modified.  Otherwise, A or
% B are converted to assumedtype, which can be 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32' or 'uint64'.  The default is
% 'uint64'.
%
% The input matrices must be real, and may be either GraphBLAS and/or
% MATLAB matrices, in any combination.  C is returned as a GraphBLAS
% matrix.  The type of C is given by GrB.optype (A,B), after any
% conversion to assumedtype, if needed.
%
% Example:
%
%   A = GrB (magic (4), 'uint8')
%   B = GrB (13 * eye (4), 'uint8') ;
%   B (3,4) = 107
%   C = bitor (A, B)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nB: ') ; fprintf ('%3x ', B) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   % in MATLAB:
%   C2 = bitor (uint8 (A), uint8 (B))
%   assert (all (C2 == C, 'all'))
%
% See also GrB/bitand, GrB/bitxor, GrB/bitcmp, GrB/bitshift, GrB/bitget,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isreal (A) || ~isreal (B))
    error ('inputs must be real') ;
end

if (nargin < 3)
    assumedtype = 'uint64' ;
end

if (~contains (assumedtype, 'int'))
    error ('assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = GrB.type (A) ;

if (isfloat (A))
    A = GrB (A, assumedtype) ;
end

if (isfloat (B))
    B = GrB (B, assumedtype) ;
end

if (~isequal (GrB.type (A), GrB.type (B)))
    error ('integer inputs must have the same type') ;
end

C = gb_eadd (A, 'bitor', B) ;

% recast C back to the original type of A
if (~isequal (ctype, GrB.type (C)))
    C = GrB (C, ctype) ;
end
