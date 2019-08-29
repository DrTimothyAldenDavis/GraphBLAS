function C = mrdivide (A, B)
% C = A/B, matrix right division
%
% If B is a scalar, then C = A./B is computed; see 'help rdivide'.
%
% Otherwise, C is computed by first converting A and B to MATLAB sparse
% matrices, and the result is converted back to a GraphBLAS double or complex
% matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (B))
    C = rdivide (A, B) ;
else
    C = gb (builtin ('mrdivide', double (A), double (B))) ;
end

