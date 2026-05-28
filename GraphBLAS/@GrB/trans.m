function C = trans (arg1, arg2, arg3, arg4, arg5)
%GRB.TRANS transpose a sparse matrix.
%
%   C = GrB.trans (A)
%   C = GrB.trans (A, desc)
%   C = GrB.trans (Cin, accum, A, desc)
%   C = GrB.trans (Cin, M, A, desc)
%   C = GrB.trans (Cin, M, accum, A, desc)
%
% The descriptor is optional.  If desc.in0 is 'transpose', then C<M>=A or
% C<M>=accum(C,A) is computed, since the default behavior is to transpose
% the input matrix.
%
% For complex matrices, GrB.trans computes the array transpose, not the
% matrix (complex conjugate) transpose.
%
% accum: an optional binary operator.
%       See 'help GrB.binopinfo' for available binary operators.
%
% See also GrB/transpose, GrB/ctranspose, GrB/conj, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 1
        [C_opaque, kind] = gbtrans (arg1) ;
    case 2
        [C_opaque, kind] = gbtrans (arg1, arg2) ;
    case 3
        [C_opaque, kind] = gbtrans (arg1, arg2, arg3) ;
    case 4
        [C_opaque, kind] = gbtrans (arg1, arg2, arg3, arg4) ;
    case 5
        [C_opaque, kind] = gbtrans (arg1, arg2, arg3, arg4, arg5) ;
end

C = gb_mexfunction_result (C_opaque, kind) ;

