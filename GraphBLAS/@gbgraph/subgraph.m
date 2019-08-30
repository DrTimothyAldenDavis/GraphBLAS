function S = subgraph (H, I)
%SUBGRAPH Extract a subgraph from a GraphBLAS gbgraph.
% S = subgraph (H, I) constructs a subgraph of the gbgraph H, induced by the
% nodes given by I.  If H has n nodes, I can either be a vector of indices in
% the range 1 to n, a logical vector of size n or less, or a cell array
% containing 2 or 3 scalars.
%
% With a logical vector I, S = subgraph (H, find(I)) is constructed.
%
% With a cell array of size 2, S = subgraph (H, {start, fini}) is the same as S
% = subgraph (H, start:fini).  If I is a cell array of size 3, S = subgraph (H,
% {start, inc, fini}) is the same as S = subgraph (H, start:inc:fini).  Using a
% cell array instead of the colon notation can be much faster and use less
% memory, particular when H is hypersparse.
%
% S is returned as a gbgraph, of same directed/undirected kind as H.
%
% Examples:
%
%   H = gbgraph (bucky)
%   S = subgraph (H, 1:10)
%   figure (1)
%   subplot (1,2,1) ; plot (H) ;
%   subplot (1,2,2) ; plot (S) ;
%
%   n = 2^40 ;
%   I = randperm (n, 5)
%   H = gb (n, n) ;
%   H (I,I) = magic (5) ;
%   H = gbgraph (H)
%   S = subgraph (H, { 1, I(2) })
%
% See also graph/subgraph, digraph/subgraph, gb/subsref.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (islogical (I))
    I = find (I) ;
end
if (~iscell (I))
    I = { I } ;
end
S = gbgraph (gb.extract (H, I, I), H.graphkind, false) ;

