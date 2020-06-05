function C = vertcat (varargin)
%VERTCAT Vertical concatenation.
% [A ; B] is the vertical concatenation of A and B.
% A and B may be GraphBLAS or MATLAB matrices, in any combination.
% Multiple matrices may be concatenated, as [A ; B ; C ; ...].
%
% If the matrices have different types, the type is determined
% according to the rules in GrB.optype.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB/horzcat, GrB.optype.

% FUTURE: this will be much faster when it is a mexFunction.
% The version below requires a sort in GrB.build.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% determine the size of each matrix and the size of the result
nmatrices = length (varargin) ;
nvals = zeros (1, nmatrices) ;
nrows = zeros (1, nmatrices) ;
A = varargin {1} ;
if (isobject (A))
    A = A.opaque ;
end
[m, n] = gbsize (A) ;
nvals (1) = gbnvals (A) ;
nrows (1) = m ;
type = gbtype (A) ;
clear A
for k = 2:nmatrices
    A = varargin {k} ;
    if (isobject (A))
        A = A.opaque ;
    end
    [m, n2] = gbsize (A) ;
    if (n ~= n2)
        gb_error ('Dimensions of arrays being concatenated are not consistent');
    end
    nvals (k) = gbnvals (A) ;
    nrows (k) = m ;
    type = gboptype (type, gbtype (A)) ;
    clear A ;
end
nrows = [0 cumsum(nrows)] ;
nvals = [0 cumsum(nvals)] ;
cnvals = nvals (end) ;
m = nrows (end) ;

% allocate the I,J,X arrays
I = zeros (cnvals, 1, 'int64') ;
J = zeros (cnvals, 1, 'int64') ;
X = zeros (cnvals, 1, type) ;

% fill the I,J,X arrays
desc.base = 'zero-based' ;
for k = 1:nmatrices
    A = varargin {k} ;
    if (isobject (A))
        A = A.opaque ;
    end
    [i, j, x] = gbextracttuples (A, desc) ;
    moffset = int64 (nrows (k)) ;
    k1 = nvals (k) + 1 ;
    k2 = nvals (k+1) ;
    I (k1:k2) = i + moffset ;
    J (k1:k2) = j ;
    X (k1:k2) = x ;
end

% build the output matrix
C = GrB (gbbuild (I, J, X, m, n, desc)) ;

