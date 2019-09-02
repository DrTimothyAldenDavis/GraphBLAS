function s = isdirected (H)
%ISDIRECTED determine if a gbgraph is undirected

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = isequal (H.graphkind, 'directed') ;

