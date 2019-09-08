%===============================================================================
% OVERLOADED METHODS
%===============================================================================

function e = numel (G)
%NUMEL maximum number of entries in a GraphBLAS matrix.
[m n] = size (G) ;
e = m*n ;

function e = nnz (G)
%NNZ number of nonzeros in a GraphBLAS matrix.
% e = nnz (G) is the number of nonzero entries in a GraphBLAS matrix G.
% Explicit zero entries are ignored.
e = gb.nz (G) ;

function X = nonzeros (G)
%NONZEROS nonzeros of a GraphBLAS matrix.
% X = nonzeros (G) is the nonzero entries in a GraphBLAS matrix G.
% Explicit zero entries are ignored.
X = gb.nzs (G) ;

%===============================================================================
% STATIC METHODS: NZ vs VAL pairs of functions
%===============================================================================

% Each function below has a twin.  Functions with "nz" in their name return a
% result that depends on the values in the matrix A.  In particular, by default
% they ignore explicit entries in A that have a numerical value of zero.  These
% entries are treated as if they were not present at all.  Each of these
% functions can be given a scalar value that redefines the additive identity;
% if not present, the value of identity is zero.

% Functions with "val" in their name do not depend on the values of the entries
% present in A.  A GraphBLAS matrix can include explicit entries whose values
% happen to equal zero, or some other addititve identity.  These entries are
% not ignored by the "val" functions.

% Both sets of functions ignore entries not present in the data structure of A.

% All static methods have the form gb.*, and may be used on GraphBLAS matrices
% or MATLAB sparse or full matrices.  In the notation below, A is any matrix
% (GraphBLAS, or MATLAB sparse or full).  G is a GraphBLAS matrix.

% The term "entry" as used below is an entry in a GraphBLAS or MATLAB matrix
% that is explicitly present in the data structure.  All methods ignore entries
% not present.  A MATLAB full matrix has all of its entries present; for
% example gb.nval (F) == numel (F) for a MATLAB full matrix F.  A MATLAB sparse
% matrix S never stores explicit zero entries; for example, gb.nval (S),
% gb.nz (S), and nnz (S) are always the same for a MATLAB sparse matrix S.
% For a GraphBLAS matrix G, nnz (G) == gb.nz (G) <= gb.nval (G) <= numel (G).
% X is full vector.

%-------------------------------------------------------------------------------
% gb.nz / gb.nval
%-------------------------------------------------------------------------------

function e = nz (A, id)
%GB.NZ number of nonzeros in a matrix.
% e = gb.nz (A) is the number of entries in A not equal to zero;
% e = gb.nz (A,id) is the number of entries in A not equal to id.
%
% gb.nz (A) is the same as nnz (A) for any matrix (GraphBLAS or MATLAB).

function e = nval (A)
%GB.NVAL number of entries in a matrix.
% e = gb.nval (A) is the number of entries in A.

%-------------------------------------------------------------------------------
% gb.nzrow / gb.nvalrow
%-------------------------------------------------------------------------------

function r = nnzrow (A, id)
%GB.NNZROW number of rows with at least one nonzero entrie.
% r = gb.nnzrow (A) is the number of rows in A with at least one nonzero entry.
% r = gb.nnzrow (A,id) is the number of rows in A with at least one entry
% not equal to id.

function r = nvalrow (A)
%NVALROW number of rows of A with at least one entry.
% r = gb.nvalrow (A) is the number of rows in A with at least one entry.

%-------------------------------------------------------------------------------
% gb.nzrows / gb.valrows
%-------------------------------------------------------------------------------

function rlist = nzrows (A, id)
%GB.NZROWS list of nonzero rows of a matrix.
% rlist = gb.nzrows (A) is a list of row indices corresponding to rows of A
% that contain at least one nonzero value.
% rlist = gb.nzrows (A,id) returns a list of row indices corresponding to rows
% of A that contain at least one entry not equal to id.

function rlist = valrows (A)
%GB.VALROWS list of non-empty rows of a matrix.
% rlist = gb.valrows (A) is a list of row indices corresponding to rows of A
% that contain at least one entry.

%-------------------------------------------------------------------------------
% gb.nzcols / gb.valcols
%-------------------------------------------------------------------------------

function clist = nzcols (A, id)
%GB.NZCOLS list of nonzero columns of a matrix.
% clist = gb.nzcols (A) is a list of column indices corresponding to columns of
% A that contain at least one nonzero value.
% clist = gb.nzcols (A,id) returns a list of column indices corresponding to
% columns of A that contain at least one entry not equal to id.

function clist = valcols (A)
%GB.VALCOLS list of non-empty columns of a matrix.
% clist = gb.valcols (A) is a list of column indices corresponding to columns
% of A that contain at least one entry.

%-------------------------------------------------------------------------------
% gb.nzcompact / gb.valcompact
%-------------------------------------------------------------------------------

function [G, rlist, clist] = nzcompact (A, id)
%GB.NZCOMPACT removes all-zero rows and columns from a matrix.
if (nargin < 2)
    id = 0 ;
end
rlist = gb.nzrows (A, id) ;
clist = gb.nzcols (A, id) ;
G = A (rlist, clist) ;

function [G, rlist, clist] = valcompact (A)
%GB.VALCOMPACT removes empty rows and columns from a matrix.
rlist = gb.valrows (A, id) ;
clist = gb.valcols (A, id) ;
G = A (rlist, clist) ;

%-------------------------------------------------------------------------------
% gb.nzs / gb.vals
%-------------------------------------------------------------------------------

function X = nzs (A, id)
%NZS nonzeros in a matrix.
% X = gb.nzs (A) is a list of the nonzero values of A.
% X = gb.nzs (A,id) is a list of the entries of A not equal to id.
%
% X has length gb.nz (A), or gb.nz (A,id) if the id parameter is present.
% gb.nzs (A) is the same as nonzeros (A) for any matrix (GraphBLAS or MATLAB).

function X = vals (A)
%VALS values in a matrix.
% X = gb.vals (A) is a list of the entries of A.
% X has length gb.nval (A).

%-------------------------------------------------------------------------------
% gb.nzunique / gb.valunique
%-------------------------------------------------------------------------------

function X = nzunique (A, id)
%GB.NZUNIQUE unique nonzeros of a matrix
if (nargin < 2)
    id = 0 ;
X = unique (gb.nzs (A, id)) ;

function X = valunique (A)
%GB.VALUNIQUE unique values of a matrix
X = unique (gb.vals (A)) ;

%===============================================================================
% STATIC METHODS
%===============================================================================

function G = prune (A, s)
%GB.PRUNE remove explicit values from a matrix.
% G = gb.prune (A) removes any explicit zeros from A.
% G = gb.prune (A, s) removes entries equal to the given scalar s.
% The result is a GraphBLAS matrix.
