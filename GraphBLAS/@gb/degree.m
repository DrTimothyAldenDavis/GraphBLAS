function d = degree (G)
%DEGREE degree of a GraphBLAS matrix
% d = degree (G) is a vector of size n for an n-by-n GraphBLAS symmetric
% matrix G, equal to the d = sum (spones (G')).  It is intended for
% symmetric matrices (undirected graphs).  For directed graphs (G is
% unsymmetric), the result depends on the format of G.  If gb.format(G) is
% 'by row' then d = degree (G) is the row degree of G (or outdegree (G)),
% where d(i) is the number of entries in G(i,:).  If 'by col', then
% d = degree (G) is the column degree of G (or indegree(G)), which is
% the number of entries in G(:,i).
%
% See also gb/indegree, gb/outdegree, graph/degree.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isequal (gb.format (G), 'by row'))
    d = row_degree (G) ;
else
    d = col_degree (G) ;
end

