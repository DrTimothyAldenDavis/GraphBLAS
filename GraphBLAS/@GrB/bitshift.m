function C = bitshift (A, B, assumedtype)
%BITSHIFT bitwise left and right shift.
% C = bitshift (A,B) computes the bitwise shift of A; if B > 0 then A is
% shifted left by B bits, and if B < 0 then A is shifted right by -B bits.
% If either A or B are scalars, they are expanded to the pattern of the
% other matrix.  C has the pattern of A (after expansion, if needed).
%
% With a third parameter, C = bitshift (A,B,assumedtype) provides a data
% type to convert A to if it is a floating-point type.  If A already has
% an integer type, then it is not modified.  Otherwise, A is converted to
% assumedtype, which can be 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32' or 'uint64'.  The default is 'uint64'.
%
% The input matrices must be real, and may be either GraphBLAS and/or
% MATLAB matrices, in any combination.  C is returned as a GraphBLAS
% matrix.  The type of C is the same as A, after any conversion to
% assumedtype, if needed.
%
% Example:
%
%   TODO example
%
% See also GrB/bitor, GrB/bitand, GrB/bitxor, GrB/bitcmp, GrB/bitget,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

atype = GrB.type (A) ;
btype = GrB.type (B) ;

if (contains (atype, 'complex') || contains (btype, 'complex'))
    error ('inputs must be real') ;
end

if (isequal (atype, 'logical') || isequal (btype, 'logical'))
    error ('inputs must not be logical') ;
end

if (nargin < 3)
    assumedtype = 'uint64' ;
end

if (~contains (assumedtype, 'int'))
    error ('assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

if (isequal (atype, 'double') || isequal (atype, 'single'))
    A = GrB (A, assumedtype) ;
    atype = assumedtype ;
end

if (~isequal (btype, 'int8'))
    % convert B to int8, and ensure all values are in range -64:64
    % ensure all entries in B are <= 64
    B = GrB.emult ('min', B, GrB.expand (GrB (64, btype), B)) ;
    if (GrB.issigned (B))
        % ensure all entries in B are >= -64
        B = GrB.emult ('max', B, GrB.expand (GrB (-64, btype), B)) ;
    end
    B = GrB (B, 'int8') ;
end

if (isscalar (B))
    % expand the scalar B to the pattern of A
    B = GrB.expand (B, A) ;
elseif (isscalar (A))
    % expand the scalar A to the pattern of B
    A = GrB.expand (A, B) ;
else
    % expand B by padding it with zeros from the pattern of A 
    B = GrB.eadd (['1st.int8'], B, GrB.expand (GrB (0, 'int8'), A)) ;
end

C = GrB.emult (['bitshift.' atype], A, B) ;

% recast C back to the original type of A
if (~isequal (ctype, GrB.type (C)))
    C = GrB (C, ctype) ;
end

