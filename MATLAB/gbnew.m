function gbnew (arg1, arg2, arg3)
%GBNEW create a new GraphBLAS sparse matrix
%
% Usage:
%
%   A = gbnew ;              empty 1-by-1 GraphBLAS double matrix
%   A = gbnew (X) ;          GraphBLAS copy of a MATLAB sparse X, same type
%   A = gbnew (type) ;       empty 1-by-1 GraphBLAS matrix of the given type
%   A = gbnew (X, type) ;    GraphBLAS typecasted copy of a MATLAB sparse X
%   A = gbnew (m, n) ;       empty m-by-n GraphBLAS double matrix
%   A = gbnew (m, n, type) ; empty m-by-n GraphBLAS matrix of the given type
%
% Creates a new GraphBLAS sparse matrix A of the specified type.
%
% The usage A = gbnew (m, n, type) is analgous to X = sparse (m, n), which
% creates an empty MATLAB sparse matrix X.  The type parameter is a string,
% which defaults to 'double' if not present.
%
% For the usage A = gbnew (X, type), X is either a MATLAB sparse matrix or a
% MATLAB struct that represents a GraphBLAS matrix.  A is created as a
% GraphBLAS struct that contains a copy of X, typecasted to the given type if
% the type string does not match the type of X.  If the type string is not
% present it defaults to 'double'.
% 
% Most of the valid type strings correspond to MATLAB class of the same name
% (see 'help class'), with the addition of the 'complex' type:
%
%   'double'    64-bit floating-point (real, not complex)
%   'single'    32-bit floating-point (real, not complex)
%   'logical'   8-bit boolean
%   'int8'      8-bit signed integer
%   'uint8'     8-bit unsigned integer
%   'int16'     16-bit signed integer
%   'uint16'    16-bit unsigned integer
%   'int32'     32-bit signed integer
%   'uint32'    32-bit unsigned integer
%   'int64'     64-bit signed integer
%   'uint64'    64-bit unsigned integer
%   'complex'   64-bit double complex.  In MATLAB, this is not a MATLAB class
%               name, but instead a property of a MATLAB sparse double matrix.
%               In GraphBLAS, 'complex' is treated as a type.
%
% See also gbsparse, sparse, class.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbnew mexFunction not found; use gbmake to compile GraphBLAS') ;

