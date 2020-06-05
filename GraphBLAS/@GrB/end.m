function i = end (G, k, ndims)
%END Last index in an indexing expression for a GraphBLAS matrix.
%
% See also GrB/size, GrB/length.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% FUTURE: add linear indexing
% FUTURE: use hypersparse matrices to implement multidimensionl nD arrays

[m, n] = gbsize (G.opaque) ;

if (ndims == 1)
    % G must be a vector for 1D indexing
    if (m > 1 && n > 1)
        % G is a matrix; linear indexing G(:) for a matrix G is not
        % yet supported.
        error ('GrB:unsupported', 'Linear indexing not supported') ;
    end
    % i = length (G)
    if (m == 0 || n == 0)
        i = 0 ;
    else
        i = max (m, n) ;
    end
elseif (ndims == 2)
    if (k == 1)
        i = m ;
    else
        i = n ;
    end
else
    error ('GrB:unsupported', '%dD indexing not supported', ndims) ;
end

