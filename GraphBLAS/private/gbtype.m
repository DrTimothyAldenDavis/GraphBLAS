function type = gbtype (X)
%GBTYPE type of a SuiteSparse:GraphBLAS or MATLAB sparse matrix
%
% Usage
%   type = gbtype (X)
%
% gbtype returns the type of a MATLAB or GraphBLAS sparse matrix.  The type
% is the same as class (X) if X is a MATLAB sparse matrix, unless the matrix
% is complex.  In that case, class (X) is 'double' but gbtype (X) is 'complex'.
%
% See also class.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbtype mexFunction not found; use gbmake to compile GraphBLAS') ;

