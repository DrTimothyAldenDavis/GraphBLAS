function index = end (G, k, ndims)
%END Last index in an indexing expression for a GraphBLAS matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (ndims == 1)
    if (~isvector (G))
        error ('Linear indexing not supported') ;
    end
    index = length (G) ;
elseif (ndims == 2)
    s = size (G) ;
    index = s (k) ;
else
    error ('%dD indexing not supported', ndims) ;
end

