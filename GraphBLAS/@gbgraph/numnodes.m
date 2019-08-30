function n = numnodes (H)
%NUMNODES Number of nodes in a gbgraph.
% n = numnodes (H) is the number of nodes in the graph H.
%
% See also graph/numnodes, gbgraph/numedges.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

n = size (H, 1) ;

