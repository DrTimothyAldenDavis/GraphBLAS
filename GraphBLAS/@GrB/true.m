function C = true (arg1, arg2, arg3, arg4)
%TRUE a GraphBLAS logical matrix with all true values.
% C = true (m, n, 'like', G) or C = ones ([m n], 'like', G) constructs an
% m-by-n GraphBLAS logical matrix C with all entries equal to true.
%
% See also GrB/zeros, GrB/false, GrB/true.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (length (arg1) == 2)
    m = arg1 (1) ;
    n = arg1 (2) ;
else
    m = arg1 ;
    n = arg2 ;
end

C = GrB (gbsubassign (gbnew (m, n, 'logical'), true)) ;

