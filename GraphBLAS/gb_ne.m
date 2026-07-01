function C = gb_ne (ghb, A, B)
%GB_NE implements GrB/ne and GhB/ne.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[am, an, atype] = gbmex_size (A) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  C is sparse.
        C = gzb_eunion (ghb, A, 0, '~=', B, 0) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (ghb, A) ~= 0)
            % since a ~= 0, entries not present in B result in a true
            % value, so the result is full.  Expand A to a full matrix.
            a = gb_scalar_to_full (ghb, bm, bn, ctype, gb_fmt (B), A) ;
            b = gzb_full (ghb, B, ctype) ;
            C = gzb_emult (ghb, a, '~=', b) ;
        else
            % since a == 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B ~= 0, then convert to true.
            C = gzb (ghb, B, 'logical') ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (ghb, B) ~= 0)
            % since b ~= 0, entries not present in A result in a true
            % value, so the result is full.  Expand B to a full matrix.
            a = gzb_full (ghb, A, ctype) ;
            b = gb_scalar_to_full (ghb, am, an, ctype, gb_fmt (A), B) ;
            C = gzb_emult (ghb, a, '~=', b) ;
        else
            % since b == 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  Simply
            % typecast A to logical.  Explicit zeroes in A become explicit
            % false entries.  Any other explicit entries not equal to zero
            % become true.
            C = gzb (ghb, A, 'logical') ;
        end
    else
        % both A and B are matrices.  C is sparse.
        C = gzb_eunion (ghb, A, 0, '~=', B, 0) ;
    end
end

