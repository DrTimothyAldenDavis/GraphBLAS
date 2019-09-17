function gbtest43
%GBTEST43 error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

ok = true ;
G = gb (magic (5)) ;

try
    x = prod (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

fprintf ('gbtest43: all tests passed\n') ;

