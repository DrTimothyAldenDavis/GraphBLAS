function result = ishermitian (G, option)
%ISHERMITIAN Determine if a GraphBLAS matrix is real symmetric or
% complex Hermitian.
%
% See also issymetric.

% FUTURE: this will be much faster.  See CHOLMOD/MATLAB/spsym.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (m ~= n)
    result = false ;
else
    if (nargin < 2)
        option = 'nonskew' ;
    end
    if (isequal (option, 'skew'))
        result = (norm (G + G', 1) == 0) ;
    else
        result = (norm (G - G', 1) == 0) ;
    end
end

