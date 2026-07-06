function C = gb_le (ghb, A, B)
%GB_LE implements "<=" and ">=" for GrB and GhB.  Not user-callable.

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
        % both A and B are scalars.  C is full.
        a = gzb_full (ghb, A, ctype) ;
        b = gzb_full (ghb, B, ctype) ;
        C = gzb_emult (ghb, a, '<=', b) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) <= 0)
            % since a <= 0, entries not present in B result in a true
            % value, so the result is full.  Expand A to a full matrix.
            a = gb_scalar_to_full (ghb, bm, bn, ctype, gb_fmt (B), A) ;
            b = gzb_full (ghb, B, ctype) ;
            C = gzb_emult (ghb, a, '<=', b) ;
        else
            % since a > 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B >= a, then convert to true.
            C = gzb_apply (ghb, '1.logical', gzb_select (ghb, B, '>=', A)) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) >= 0)
            % since b >= 0, entries not present in A result in a true
            % value, so the result is full.  Expand B to a full matrix.
            b = gb_scalar_to_full (ghb, am, an, ctype, gb_fmt (A), B) ;
            a = gzb_full (ghb, A, ctype) ;
            C = gzb_emult (ghb, a, '<=', b) ;
        else
            % since b < 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  select all
            % entries in A <= b, then convert to true.
            C = gzb_apply (ghb, '1.logical', gzb_select (ghb, A, '<=', B)) ;
        end
    else
        % both A and B are matrices.  C is full.
        a = gzb_full (ghb, A, ctype) ;
        b = gzb_full (ghb, B, ctype) ;
        C = gzb_emult (ghb, a, '<=', b) ;
    end
end

