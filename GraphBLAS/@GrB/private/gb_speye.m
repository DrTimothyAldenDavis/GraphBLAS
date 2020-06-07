function C = gb_speye (varargin)
%GB_SPEYE Sparse identity matrix, of any type supported by GraphBLAS.
% Implements C = GrB.eye (...) and GrB.speye (...).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% get the type
type = 'double' ;
nargs = nargin ;
if (nargs > 1 && ischar (varargin {nargs}))
    type = varargin {nargs} ;
    nargs = nargs - 1 ;
end

% get the size
if (nargs == 0)
    m = 1 ;
    n = 1 ;
elseif (nargs == 1)
    % C = gb_speye (n) or gb_speye ([m n])
    arg1 = varargin {1} ;
    if (length (arg1) == 1)
        % C = gb_speye (n)
        m = arg1 ;
        n = m ;
    elseif (length (arg1) == 2)
        % C = gb_speye ([m n])
        m = arg1 (1) ;
        n = arg1 (2) ;
    else
        error ('GrB:unsupported', 'only 2D arrays supported') ;
    end
elseif (nargs == 2)
    % C = gb_speye (m,n)
    m = varargin {1} ;
    n = varargin {2} ;
else
    error ('GrB:unsupported', 'only 2D arrays supported') ;
end

% construct the m-by-n identity matrix of the given type
m = max (m, 0) ;
n = max (n, 0) ;
mn = min (m, n) ;
I = int64 (0) : int64 (mn-1) ;
desc.base = 'zero-based' ;

if (isequal (type, 'single complex'))
    X = complex (ones (mn, 1, 'single')) ;
elseif (contains (type, 'complex'))
    X = complex (ones (mn, 1, 'double')) ;
else
    X = ones (mn, 1, type) ;
end

C = gbbuild (I, I, X, m, n, '1st', type, desc) ;

