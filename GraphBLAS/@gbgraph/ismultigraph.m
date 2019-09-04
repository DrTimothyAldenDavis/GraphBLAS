function s = ismultigraph (G)
%ISMULTIGRAPH returns false for a gbgraph.
% ismultigraph (G) always returns false, since GraphBLAS does not yet support
% multigraphs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = false ;

