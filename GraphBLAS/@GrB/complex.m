function C = complex (A, B)
%COMPLEX cast a GraphBLAS matrix to MATLAB sparse double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a MATLAB sparse
% complex matrix.
%
% With two inputs, C = complex (A,B) returns a MATLAB sparse matrix
% C = A + 1i*B, where A or B are real GraphBLAS matrices.
%
% To typecast the matrix G to a GraphBLAS sparse double complex matrix
% instead, use C = GrB (G, 'complex') or C = GrB (G, 'double complex').
% To typecast the matrix G to a GraphBLAS single complex matrix, use
% C = GrB (G, 'single complex').
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices
% A and B, use C = A + 1i*B instead.
%
% See also cast, GrB, GrB/double, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO

if (nargin == 1)

    % with a single input, A must be a GraphBLAS matrix (otherwise,
    % this overloaded method for GrB objects would not be called).
    % Convert A to a double complex matrix C.
    C = gbsparse (A.opaque, 'double complex') ;

else

    if (~isreal (A) || ~isreal (A))
        error ('inputs must be real') ;
    end

    % A or B must be a GraphBLAS matrix, or both, and thus C is a
    % GraphBLAS matrix.
    C = A + 1i * B ;

    % convert C to a MATLAB double complex matrix
    C = gbsparse (C.opaque, 'double complex') ;

end

