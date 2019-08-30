function e = nvals (X)
%GB.NVALS the number of entries in a GraphBLAS or MATLAB matrix.
% gb.nvals (X) is the number of explicit entries in a GraphBLAS or MATLAB
% matrix X.  Note that the entries in a GraphBLAS matrix can have any value,
% including zero.  MATLAB drops zero-valued entries from its sparse matrices.
% This cannot be done in a GraphBLAS matrix because of the different semirings
% that may be used.  In a shortest-path problem, for example, and edge with
% weight zero is very different from no edge at all.
%
% For a MATLAB sparse matrix S, gb.nvals (S) and nnz (S) are the same.  For a
% MATLAB full matrix F, gb.nvals (F) and numel (F) are the same.
%
% See also nnz, numel.

if (isa (X, 'gb'))
    e = gbnvals (X.opaque) ;
else
    e = gbnvals (X) ;
end

