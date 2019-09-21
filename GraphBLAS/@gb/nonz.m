function result = nonz (A, varargin)
%GB.NONZ count or query the nonzeros of a matrix.
% A GraphBLAS matrix can include explicit entries that have the value zero.
% These entries never appear in a MATLAB sparse matrix.  This function
% counts or queries the nonzeros of matrix, checking their value and treating
% explicit zeros the same as entries that do not appear in the pattern of A.
%
% e = gb.nonz (A)         number of nonzeros
% e = gb.nonz (A, 'all')  number of nonzeros
% e = gb.nonz (A, 'row')  number of rows with at least one nonzeros
% e = gb.nonz (A, 'col')  number of rows with at least one nonzeros
%
% X = gb.nonz (A, 'list')         list of values of unique nonzeros
% X = gb.nonz (A, 'all', 'list')  list of values of unique nonzeros
% I = gb.nonz (A, 'row', 'list')  list of rows with at least one nonzeros
% J = gb.nonz (A, 'col', 'list')  list of cols with at least one nonzeros
%
% d = gb.nonz (A, 'row', 'degree')
%   If A is m-by-n, then d is a sparse column vector of size m, with d(i) equal
%   to the number of nonzeros in A(i,:).  If A(i,:) has no nonzeros, then d(i)
%   is an implicit zero, not present in the pattern of d, so that
%   I = find (d) is the same I = gb.nonz (A, 'row', 'list').
%
% d = gb.nonz (A, 'col', 'degree')
%   If A is m-by-n, d is a sparse column vector of size n, with d(j) equal to
%   the number of nonzeros in A(:,j).  If A(:,j) has no nonzeros, then d(j)
%   is an implicit zero, not present in the pattern of d, so that
%   I = find (d) is the same I = gb.nonz (A, 'col', 'list').
%
% With an optional scalar argument as the last argument, the value of the
%   'zero' can be specified; d = gb.nonz (A, ..., id).  For example, to count
%   all entries in A not equal to one, use gb.nonz (A, 1)
%
% Example:
%
%   A = magic (5) ;
%   A (A < 10) = 0              % MATLAB full matrix with some explicit zeros
%   nnz (A)
%   gb.nonz (A)                 % same as nnz (A)
%   G = gb (A)                  % contains explicit zeros
%   gb.nonz (G)                 % same as nnz (A)
%   G (A > 18) = sparse (0)     % entries A>18 deleted, still has explicit zeros
%   gb.nonz (G)
%   gb.nonz (G, 'list')
%   S = double (G)              % MATLAB sparse matrix; no explicit zeros
%   gb.nonz (S)
%   gb.nonz (S, 'list')
%
% See also gb.nonz, nnz, gb/nnz, nonzeros, gb/nonzeros, gb.prune.
%
% See also gb.entries, nnz, gb/nnz, nonzeros, gb/nonzeros, gb.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% issparse (G) is overloaded for a GraphBLAS matrix, and always returns true.
% Here, we need to know if A is truly a MATLAB sparse matrix, not a GraphBLAS
% sparse matrix.
matlab_sparse = builtin ('issparse', A) ;

if (nargin > 1 && ~ischar (varargin {end}))
    id = gb_get_scalar (varargin {end}) ;
    if (id == 0 && matlab_sparse)
        % id is zero, and A is a MATLAB sparse matrix, so no need to prune.
        result = gb.entries (A, varargin {1:end-1}) ;
    else
        % id is nonzero, so it can appear in any matrix (GraphBLAS, MATLAB
        % sparse, or MATLAB full), so it must be pruned from A first.
        result = gb.entries (gb.prune (A, id), varargin {1:end-1}) ;
    end
else
    if (matlab_sparse)
        % id is not present so it defaults to zero, and A is MATLAB sparse
        % matrix, so no need to prune explicit zeros from A.
        result = gb.entries (A, varargin {:}) ;
    else
        % A is a GraphBLAS matrix, or a MATLAB full matrix, so zeros
        % must be pruned.  This does not prune explicit zeros in a MATLAB
        % full matrix.
        result = gb.entries (gb.prune (A), varargin {:}) ;
    end
end

