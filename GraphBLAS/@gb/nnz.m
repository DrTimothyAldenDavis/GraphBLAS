function e = nnz (G)
%NNZ the number of entries in a GraphBLAS matrix.
% nnz (G) is the same as gb.nvals (G); some of the entries may actually
% be explicit zero-valued entries.  See 'help gb.nvals' for more details.
% To count the number of entries of G that have a nonzero value, use
% nnz (double (G)).  Use G = gb.prune (G) to remove explicit
% zeros from G.
%
% See also gb.nvals, nonzeros, size, numel.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = gbnvals (G.opaque) ;

