function C = gb_lt (ghb, A, B)
%GB_LT implements "<" and ">" for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

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
            bfmt = gb_fmt (B) ;
            if (~gb_issigned (btype))
                % a < 0, and B has an unsigned type.  C is all true.
                C = gb_scalar_to_full (ghb, bm, bn, 'logical', bfmt, true) ;
            else
                % since a < 0, entries not present in B result in a true
                % value, so the result is full.  Expand A to full.
                a = gb_scalar_to_full (ghb, bm, bn, ctype, bfmt, A) ;
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
            b = gb_scalar_to_full (ghb, am, an, ctype, gb_fmt (A), B) ;
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

