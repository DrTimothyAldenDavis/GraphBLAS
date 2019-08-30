function s = issymmetric (G, option)
%ISHERMITIAN Determine if a GraphBLAS matrix is symmetric.
%
% See also ishermitian.

% FUTURE: this will be much faster.  See CHOLMOD/MATLAB/spsym.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% TODO use isequal, not norm

[m n] = size (G) ;
if (m ~= n)
    s = false ;
else
    if (nargin < 2)
        option = 'nonskew' ;
    end
    if (isequal (option, 'skew'))
        s = (norm (G + G.', 1) == 0) ;
    else
        s = (norm (G - G.', 1) == 0) ;
    end
end

