function C = mldivide (A, B)
% C = A\B, matrix left division
%
% If A is a scalar, then C = A.\B is computed; see 'help ldivide'.
%
% Otherwise, C is computed by first converting A and B to MATLAB sparse
% matrices, and the result is converted back to a GraphBLAS double or
% complex matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    C = rdivide (B, A) ;
else
    C = gb (builtin ('mldivide', double (A), double (B))) ;
end

