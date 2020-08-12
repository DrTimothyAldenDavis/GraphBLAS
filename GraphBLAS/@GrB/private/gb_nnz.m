function e = gb_nnz (G)
%GB_NNZ the number of nonzeros in a GraphBLAS matrix.
% Implements e = nnz (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

try
    % TODO: gbselect doesn't work for bitmap yet
    e = gbnvals (G) - gbnvals (gbselect (G, '==0')) ;
catch
    e = gbnvals (G) ;
end

