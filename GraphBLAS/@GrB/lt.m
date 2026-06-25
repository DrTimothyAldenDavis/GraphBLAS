function C = lt (A, B)
%A < B less than.
% C = (A < B) compares A and B element-by-element.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/le, GrB/gt, GrB/ge, GrB/ne, GrB/eq.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A<0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B>0, otherwise C is a subset of A.
% A matrix, B matrix:  C has the pattern of the set union, A+B.

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
        % both A and B are scalars
        C = gzb_eunion (ghb, A, 0, '<', B, 0) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) < 0)
            if (~gb_issigned (btype))
                % a < 0, and B has an unsigned type.  C is all true.
                C = gb_scalar_to_full (bm, bn, 'logical', gb_fmt (B), true) ;
            else
                % since a < 0, entries not present in B result in a true
                % value, so the result is full.  Expand A to full.
                a = gb_scalar_to_full (bm, bn, ctype, gb_fmt (B), A) ;
                b = gzb_full (ghb, B, ctype) ;
                C = gzb_emult (ghb, a, '<', b) ;
            end
        else
            % since a >= 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B > a, then convert to true.
            C = gzb_apply (ghb, '1.logical', gzb_select (ghb, B, '>', A)) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        b = gb_scalar (B) ;
        if (b < 0 && ~gb_issigned (atype))
            % b is negative, and A has an unsigned type.  C is all false.
            C = gzb (ghb, am, an, 'logical') ;
        elseif (b > 0)
            % since b > 0, entries not present in A result in a true
            % value, so the result is full.  Expand B to a full matrix.
            a = gzb_full (ghb, A, ctype) ;
            b = gb_scalar_to_full (am, an, ctype, gb_fmt (A), B) ;
            C = gzb_emult (ghb, a, '<', b) ;
        else
            % since b <= 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  Select all
            % entries in A < b, then convert to true.
            C = gzb_apply (ghb, '1.logical', gzb_select (ghb, A, '<', B)) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = gzb_eunion (ghb, A, 0, '<', B, 0) ;
    end
end

