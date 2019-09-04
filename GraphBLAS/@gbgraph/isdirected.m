function s = isdirected (G)
%ISDIRECTED determine if a gbgraph is undirected
% isdirected (G) is true if the gbgraph G is directed, and false otherwise.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = isequal (kind (G), 'directed') ;

