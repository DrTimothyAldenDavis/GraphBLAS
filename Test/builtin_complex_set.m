function builtin_complex = builtin_complex_set (builtin_complex)
%BUILTIN_COMPLEX_SET set a global flag to determine the GrB Complex type 
%
% builtin_complex = builtin_complex_set (builtin_complex)
%
% Sets the GraphBLAS_builtin_complex flag.  If true, the Complex ==
% GxB_FC64.  If false, the Complex GrB_Type is allocated as a user-defined
% type.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_builtin_complex
if (nargin > 0)
    GraphBLAS_builtin_complex = logical (builtin_complex) ;
elseif (isempty (GraphBLAS_builtin_complex))
    GraphBLAS_builtin_complex = true ;
end

builtin_complex = GraphBLAS_builtin_complex ;

