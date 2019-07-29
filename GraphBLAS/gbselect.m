function Cout = gbselect (Cin, M, accum, op, A, thunk, desc)
%GBSELECT select a subset of entries from a sparse matrix
%
% Usage:
%
%   Cout = gbselect (Cin, M, accum, op, A, thunk, desc)
%
% TODO
% predefined operators, with their equivalent aliases
%    'tril'
%    triu
%    diag
%    offdiag
%    nonzero     '!=0'   '~=0'
%    eq0         '==0'
%    gt0         '>0'
%    ge0         '>=0'
%    lt0         '<0'
%    le0         '<=0'
%    nethunk     '!=thunk'   '~=thunk'
%    eqthunk     '==thunk'
%    gtthunk     '>thunk'
%    gethunk     '>=thunk'
%    ltthunk     '<thunk'
%    lethunk     '<=thunk'
%
% See also tril, triu, diag.
