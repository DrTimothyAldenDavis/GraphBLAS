function C = horzcat (varargin)
%HORZCAT Horizontal concatenation.
% [A B] or [A,B] is the horizontal concatenation of A and B.
% A and B may be GraphBLAS or MATLAB matrices, in any combination.
% Multiple matrices may be concatenated, as [A, B, C, ...].
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also vercat, gb/vertcat.

% FUTURE: this will be much faster when it is a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% determine the size of each matrix and the size of the result
nmatrices = length (varargin) ;
nvals = zeros (1, nmatrices) ;
ncols = zeros (1, nmatrices) ;
A = varargin {1} ;
[m n] = size (A) ;
nvals (1) = gb.nvals (A) ;
ncols (1) = n ;
type = gb.type (A) ;
clear A
for k = 2:nmatrices
    B = varargin {k} ;
    [m2 n] = size (B) ;
    if (m ~= m2)
        error('Dimensions of arrays being concatenated are not consistent');
    end
    nvals (k) = gb.nvals (B) ;
    ncols (k) = n ;
    clear B ;
end
ncols = [0 cumsum(ncols)] ;
nvals = [0 cumsum(nvals)] ;
cnvals = nvals (end) ;
n = ncols (end) ;

% allocate the I,J,X arrays
I = zeros (cnvals, 1, 'int64') ;
J = zeros (cnvals, 1, 'int64') ;
X = zeros (cnvals, 1, type) ;

% fill the I,J,X arrays
d.kind = 'zero-based' ;
for k = 1:nmatrices
    [i, j, x] = gb.extracttuples (varargin {k}, d) ;
    noffset = int64 (ncols (k)) ;
    koffset = nvals (k) ;
    kvals = gb.nvals (varargin {k}) ;
    I ((koffset+1):(koffset+kvals)) = i ;
    J ((koffset+1):(koffset+kvals)) = j + noffset ;
    X ((koffset+1):(koffset+kvals)) = x ;
end

% build the output matrix
C = gb.build (I, J, X, m, n) ;

