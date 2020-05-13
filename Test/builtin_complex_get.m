function builtin_complex = builtin_complex_get
%BUILTINT_COMPLEX get the flag that determines the GrB_Type Complex
%
% builtin_complex = builtin_complex_get
%
% Returns the boolean flag builtin_complex.  If true, GxB_FC64 is used,
% and set as the "user-defined" Complex type.  Otherwise, the Complex type is
% created as user-defined.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_builtin_complex
if (isempty (GraphBLAS_builtin_complex))
    builtin_complex = builtin_complex_set (true) ;
end
builtin_complex = GraphBLAS_builtin_complex ;

