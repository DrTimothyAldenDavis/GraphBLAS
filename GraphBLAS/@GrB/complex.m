function C = complex (A, B)
%COMPLEX cast to a built-in double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a built-in
% double complex matrix.  C is full if all entries in G are present,
% or sparse otherwse.
%
% With two inputs, C = complex (A,B) returns a matrix C = A + 1i*B,
% where A or B are real matrices (@GrB/built-in in any
% combination).  If A or B are nonzero scalars and the other input is a
% matrix, or if both A and B are scalars, C is full.
%
% To typecast the matrix G to a GraphBLAS double complex matrix
% instead, use C = GrB (G, 'complex') or C = GrB (G, 'double complex').
% To typecast the matrix G to a GraphBLAS single complex matrix, use
% C = GrB (G, 'single complex').
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices
% A and B, use C = A + 1i*B instead.
%
% Since sparse single complex matrices are not built-in, C is
% always returned as a double complex matrix (sparse or full).
%
% See also cast, GrB, GrB/double, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin == 1)

    % with a single input, A must be a GraphBLAS matrix (otherwise,
    % this overloaded method for GrB objects would not be called).
    % Convert A to a built-in double complex matrix C.
    C = gzb_cast (ghb, A, 'double complex') ;

else

    % with two inputs, A and B are real matrices (@GrB or built-in)
    % but at least one must be GrB or otherwise this overloaded method
    % would not be called).  The output is a double complex matrix.

    [am, an, atype] = gbmex_size (A) ;
    [bm, bn, btype] = gbmex_size (B) ;
    a_is_scalar = (am == 1) && (an == 1) ;
    b_is_scalar = (bm == 1) && (bn == 1) ;

    if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
        error ('GrB:error', 'inputs must be real') ;
    end

    if (a_is_scalar)
        if (b_is_scalar)
            % both A and B are scalars.  C is also a scalar.
            a = gzb_full (ghb, A, 'double') ;
            b = gzb_full (ghb, B, 'double') ;
            desc.kind = 'full' ;
            C = gzb_emult (ghb, 'cmplx.double', a, b, desc) ;
        else
            % A is a scalar, B is a matrix.  C is full, unless A == 0.
            if (gb_scalar (A) == 0)
                % C = 1i*B, so A = zero, C is sparse or full.
                desc.kind = 'builtin' ;
                C = gzb_apply2 (ghb, 'cmplx.double', 0, B, desc) ;
            else
                % expand A and B to full double matrices; C is full
                bfmt = gb_fmt (B) ;
                desc.kind = 'full' ;
                a = gb_scalar_to_full (bm, bn, 'double', bfmt, A) ;
                b = gzb_full (ghb, B, 'double') ;
                C = gzb_emult (ghb, 'cmplx.double', a, b, desc) ;
            end
        end
    else
        if (b_is_scalar)
            % A is a matrix, B is a scalar.  C is full, unless B == 0.
            if (gb_scalar (B) == 0)
                % C = complex (A); C is sparse or full
                C = gzb_cast (ghb, A, 'double.complex') ;
            else
                % expand A and B to full double matrices; C is full
                afmt = gb_fmt (A) ;
                desc.kind = 'full' ;
                a = gzb_full (ghb, A, 'double') ;
                b = gb_scalar_to_full (am, an, 'double', afmt, B) ;
                C = gzb_emult (ghb, 'cmplx.double', a, b, desc) ;
            end
        else
            % both A and B are matrices.  C is sparse or full.
            desc.kind = 'builtin' ;
            b = gzb_apply2 (ghb, B, '*', 1i) ;
            C = gzb_eadd (ghb, A, '+', b, desc) ;
        end
    end

end

% return C as a builtin MATLAB/Octave matrix
C = gbmex_builtin (C) ;

