function C = kronecker (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.KRONECKER sparse Kronecker product.
%
% syntax for a new matrix C:                        computation:
% C = GrB.kronecker (op, A, B, desc)                % C = op(A,B)
% C = GrB.kronecker (Cin, accum, op, A, B, desc)    % C = Cin + op(A,B)
% C = GrB.kronecker (Cin, M, op, A, B, desc)        % C = Cin ; C<M> = op(A,B)
% C = GrB.kronecker (Cin, M, accum, op, A, B, desc) % C = Cin ; C<M> += op(A,B)
%
% GrB.kronecker computes the Kronecker product T=kron(A,B), using any binary
% op (shown as op(A,B) in the computations listed above).
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  If desc.in1
% is 'transpose', then the input matrix B is transposed before applying the
% operator.
%
% See also kron, GrB/kron, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 3
            [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.kronecker (Cin, M, accum, op, A, B, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

