function C = gb_full (A, ctype)
%GB_FULL convert a GraphBLAS or MATLAB matrix to full, with typecast.
% C = gb_full (A, ctype) converts A to a full GraphBLAS matrix.
%
% ctype can be any MATLAB class, or 'single complex', or 'double complex'.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% TODO delete this function

if (nargin < 2)
    ctype = GrB.type (A) ;
end

C = full (A, ctype, GrB (0, ctype)) ;

