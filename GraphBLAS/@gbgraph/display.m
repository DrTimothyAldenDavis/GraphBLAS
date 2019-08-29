function display (H)
%DISPLAY display the contents of a GraphBLAS graph.
% display (G) displays the attributes and first few entries of a GraphBLAS
% sparse matrix object.  Use disp(G,3) to display all of the content of G.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

display@gb (H) ;
[e d] = numedges (H) ;
fprintf ('  GraphBLAS %s: %d nodes, %d edges', H.graphkind, numnodes (H), e)
if (d == 0)
    fprintf (' (no self edges).\n\n') ;
else
    fprintf (' (%d are self-edges).\n\n', d) ;
end

