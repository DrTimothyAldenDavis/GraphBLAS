function C = emult (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EMULT sparse element-wise 'multiplication'.
%
%   C = GrB.emult (op, A, B, desc)
%   C = GrB.emult (Cin, accum, op, A, B, desc)
%   C = GrB.emult (Cin, M, op, A, B, desc)
%   C = GrB.emult (Cin, M, accum, op, A, B, desc)
%
% GrB.emult computes the element-wise 'multiplication' T=A.*B.  The result
% T has the pattern of the intersection of A and B. The operator is used
% where A(i,j) and B(i,j) are present.  Otherwise the entry does not
% appear in T.
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% Cin, M, accum, and the optional descriptor desc are the same as all other
% GrB.methods; see GrB.mxm and GrB.descriptorinfo for more details.  For the
% binary operator, see GrB.binopinfo.
%
% See also GrB.eadd, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 3
        [C_opaque, kind] = gbemult (arg1, arg2, arg3) ;
    case 4
        [C_opaque, kind] = gbemult (arg1, arg2, arg3, arg4) ;
    case 5
        [C_opaque, kind] = gbemult (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C_opaque, kind] = gbemult (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C_opaque, kind] = gbemult (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

C = gb_mexfunction_result (C_opaque, kind) ;

