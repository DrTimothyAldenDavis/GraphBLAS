function c = edgecount (G,s,t)
%EDGECOUNT Determine the number of edges between two nodes.
%
% See also graph/edgecount, digraph/edgecount.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~iscell (s))
    s = { s } ;
end
if (~iscell (t))
    t = { t } ;
end

c = nnz (gb.extract (G, s, t)) ;

