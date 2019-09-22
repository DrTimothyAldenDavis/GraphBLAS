function [C, I, J] = compact (A, id)
%GB.COMPACT remove empty rows and columns from a matrix.
% C = gb.compact (A) returns rows and columns from A that have no entries.
% It has no effect on a MATLAB full matrix, except to convert it to a
% GraphBLAS matrix, since all entries are present in a MATLAB full matrix.
%
% To remove rows and columns with no entries or only explicit zero entries,
% use C = gb.compact (A,0).  For a MATLAB sparse matrix, gb.compact (A,0)
% and gb.compact (A) are identical.
%
% To remove rows and colums with no entries, or with only entries equal to
% a particular scalar value, use C = gb.compact (A, id), where id is the
% scalar value.
%
% With two additional output arguments, [C,I,J] = gb.compact (A, ...),
% the indices of non-empty rows and columns of A are returned, so that
% C = A (I,J).  The lists I and J are returned in sorted order.
%
% Example:
%
%   n = 2^40 ;
%   H = gb (n,n)
%   I = sort (randperm (n, 4))
%   J = sort (randperm (n, 4))
%   A = magic (4) ;
%   H (I,J) = A
%   [C, I, J] = gb.compact (H)
%   H (I, J(1)) = 0
%   [C, I, J] = gb.compact (H, 0)
%   norm (C - A (:,2:end), 1)
%
% See also gb.entries, gb.nonz, gb.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin > 1)
    id = gb_get_scalar (id) ;
    if (~(id == 0 && builtin ('issparse', A)))
        A = gb.prune (A, id) ;
    end
end

S = gb.apply ('1.double', A) ;
I = find (gb.vreduce ('+', S)) ;
J = find (gb.vreduce ('+', S, struct ('in0', 'transpose'))) ;
C = gb.extract (A, { I }, { J }) ;

