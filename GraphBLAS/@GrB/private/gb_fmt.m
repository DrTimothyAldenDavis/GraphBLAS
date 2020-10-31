function f = gb_fmt (A)
%GB_FMT return the format of A as a single string.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[f, s] = GrB.format (A) ;

if (~isempty (s))
    f = [s ' ' f] ;
end

