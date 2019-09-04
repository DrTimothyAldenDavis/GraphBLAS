function s = isundirected (G)
%ISUNDIRECTED determine if a gbgraph is undirected
% isundirected (G) is true if the gbgraph G is undirected, and false otherwise.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = isequal (kind (G), 'undirected') ;

