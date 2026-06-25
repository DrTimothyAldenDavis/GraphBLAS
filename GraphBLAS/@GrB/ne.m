function C = ne (A, B)
%A ~= B not equal.
% C = (A ~= B) compares A and B element-by-element.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/lt, GrB/le, GrB/gt, GrB/ge, GrB/eq.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A~=0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B~=0, otherwise C is a subset of A.
% A matrix, B matrix:  C is sparse, with the pattern of A+B.
% Zeroes are then dropped from C after it is computed.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[am, an, atype] = gbmex_size (A) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  C is sparse.
        C = GrB (gbmex_eunion (ghb, A, 0, '~=', B, 0)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) ~= 0)
            % since a ~= 0, entries not present in B result in a true
            % value, so the result is full.  Expand A to a full matrix.
            a = gb_scalar_to_full (bm, bn, ctype, gb_fmt (B), A) ;
            b = GrB (gbmex_full (ghb, B, ctype)) ;
            C = GrB (gbmex_emult (ghb, a, '~=', b)) ;
        else
            % since a == 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B ~= 0, then convert to true.
            C = GrB (B, 'logical') ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) ~= 0)
            % since b ~= 0, entries not present in A result in a true
            % value, so the result is full.  Expand B to a full matrix.
            a = GrB (gbmex_full (ghb, A, ctype)) ;
            b = gb_scalar_to_full (am, an, ctype, gb_fmt (A), B) ;
            C = GrB (gbmex_emult (ghb, a, '~=', b)) ;
        else
            % since b == 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  Simply
            % typecast A to logical.  Explicit zeroes in A become explicit
            % false entries.  Any other explicit entries not equal to zero
            % become true.
            C = GrB (A, 'logical') ;
        end
    else
        % both A and B are matrices.  C is sparse.
        C = GrB (gbmex_eunion (ghb, A, 0, '~=', B, 0)) ;
    end
end

