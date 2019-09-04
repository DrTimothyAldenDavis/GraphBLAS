function s = isfull (A)
%GB.ISFULL determine if all entries are present.
% For a GraphBLAS matrix, or a MATLAB sparse matrix, gb.isfull (A) is true if
% numel (A) == nnz (A).  A can be a GraphBLAS matrix, or a MATLAB sparse or
% full matrix.  gb.isfull (A) is always true if A is a MATLAB full matrix.

if (isa (A, 'gb') || issparse (A))
    s = (numel (A) == nnz (A)) ;
else
    s = true ;
end

