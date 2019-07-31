function Cout = gbselect (Cin, M, accum, op, A, thunk, desc)
%GBSELECT select a subset of entries from a sparse matrix
%
% Usage:
%
%   Cout = gbselect (Cin, M, accum, op, A, thunk, desc)
%
% TODO
%
% predefined operators, with their equivalent aliases
%
%   tril
%   triu
%   diag
%   offdiag
%   nonzero     ne0         !=0   ~=0
%   eqzero      eq0         ==0
%   gtzero      gt0         >0
%   gezero      ge0         >=0
%   ltzero      lt0         <0
%   lezero      le0         <=0
%   nethunk     !=thunk     ~=thunk
%   eqthunk     ==thunk
%   gtthunk     >thunk
%   gethunk     >=thunk
%   ltthunk     <thunk
%   lethunk     <=thunk
%
% See also tril, triu, diag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbselect mexFunction not found; use gbmake to compile GraphBLAS') ;

