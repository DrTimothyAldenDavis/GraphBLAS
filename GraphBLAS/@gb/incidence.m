function C = incidence (A, varargin)
%GB.INCIDENCE Graph incidence matrix.
% C = gb.incidence (A) is the graph incidence matrix of the square matrix A.  C
% is GraphBLAS matrix of size n-by-e, if A is n-by-n with e entries (not
% including diagonal entries).  The jth column of has 2 entries: C(s,j) = -1
% and C(t,j) = 1, where A(s,t) is an entry A.  Diagonal entries in A are
% ignored.   Optional string arguments can appear after A:
%
%   C = gb.incidence (A, ..., 'directed') constructs a matrix C of size n-by-e
%       where e = gb.entries (gb.offdiag (A)).  Any entry in the upper or lower
%       trianglar part of A results in a unique column of C.  The diagonal is
%       ignored.  This is the default. 
%
%   C = gb.incidence (A, ..., 'unsymmetric') is the same as 'directed'.
%
%   C = gb.incidence (A, ..., 'undirected') assumes A is symmetric, and only
%       creates columns of C based on entries in tril (A,-1).  The diagonal and
%       upper triangular part of A are ignored.
%
%   C = gb.incidence (A, ..., 'symmetric') is the same as 'undirected'.
%
%   C = gb.incidence (A, ..., 'lower') is the same as 'undirected'.
%
%   C = gb.incidence (A, ..., 'upper') is the same as 'undirected', except that
%       only entries in triu (A,1) are used.
%
%   C = gb.incidence (A, ..., type) construct C with the type 'double',
%       'single', 'int8', 'int16', 'int32', or 'int64'.  The default is
%       'double'.
%
% Examples:
%
%   A = sprand (5, 5, 0.5)
%   C = gb.incidence (A)
%
% See also graph/incidence, digraph/incidence.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m, n] = size (A) ;
if (m ~= n)
    gb_error ('A must be square') ;
end

% get the string options
kind = 'directed' ;
type = 'double' ;
for k = 1:nargin-1
    arg = lower (varargin {k}) ;
    switch arg
        case { 'directed', 'undirected', 'symmetric', 'unsymmetric', ...
            'lower', 'upper' }
            kind = arg ;
        case { 'double', 'single', 'int8', 'int16', 'int32', 'int64' }
            type = arg ;
        otherwise
            gb_error ('unknown option') ;
    end
end

if (isequal (kind, 'directed') || isequal (kind, 'unsymmetric'))
    % create the incidence matrix of a directed graph, using all of A;
    % except that diagonal entries are ignored.
    A = gb.select ('offdiag', A) ;
elseif (isequal (kind, 'upper'))
    % create the incidence matrix of an undirected graph, using only entries
    % in the strictly upper triangular part of A.
    A = triu (A, 1) ;
else
    % create the incidence matrix of an undirected graph, using only entries
    % in the strictly lower triangular part of A.
    A = tril (A, -1) ;
end

% build the incidence matrix
[i, j] = gb.extracttuples (A, struct ('kind', 'zero-based')) ;
e = length (i) ;
k = uint64 (0:e-1)' ;
x = ones (e, 1, type) ;
C = gb.build ([i ; j], [k ; k], [-x ; x], n, e) ;

