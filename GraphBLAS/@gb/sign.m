function C = sign (G)
%SIGN Signum function.
% For each element of a GraphBLAS matrix G, sign(G) returns 1 if the
% element is greater than zero, 0 if it equals zero, and -1 if it is less
% than zero.  The output C is a sparse GraphBLAS matrix, with no explicit
% zeros; any entry not present is implicitly zero.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = spones (gb.select ('>0', G)) - spones (gb.select ('<0', G)) ;

