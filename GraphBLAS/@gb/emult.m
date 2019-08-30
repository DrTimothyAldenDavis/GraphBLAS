function Cout = emult (varargin)
%GB.EMULT sparse element-wise 'multiplication'
%
% Usage:
%
%   Cout = gb.emult (op, A, B, desc)
%   Cout = gb.emult (Cin, accum, op, A, B, desc)
%   Cout = gb.emult (Cin, M, op, A, B, desc)
%   Cout = gb.emult (Cin, M, accum, op, A, B, desc)
%
% gb.emult computes the element-wise 'multiplication' T=A.*B.  The result T
% has the pattern of the intersection of A and B. The operator is used
% where A(i,j) and B(i,j) are present.  Otherwise the entry does not
% appear in T.
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% Cin, M, accum, and the descriptor desc are the same as all other
% gb.methods; see gb.mxm and gb.descriptorinfo for more details.  For the
% binary operator, see gb.binopinfo.
%
% See also gb.eadd.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbemult (args {:})) ;
else
    Cout = gbemult (args {:}) ;
end

