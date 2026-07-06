function [I,J,X] = gb_find (ghb, G_arg, k, search)
%GB_FIND: implements GrB/find and GhB/find.  Not user-callable

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FIXME: ghb not needed

if (gb_is_grb (G_arg))
    G_arg = struct (G_arg) ;
end

% prune explicit zeros
gbmex_wait (G_arg) ;
desc.format = gbmex_format (G_arg) ;
G = gzb_select (1, G_arg, 'nonzero', desc) ;

if (nargin > 2)
    k = ceil (double (gb_get_scalar (k))) ;
    if (k < 1)
        error ('GrB:error', 'k must be positive') ;
    end
    if (~isequal (gbmex_format (G), 'by col'))
        % find (G, k) assumes the matrix is stored by column, so reformat G
        % if it is stored by row.
        G = gzb (1, G, 'by col') ;
    end
end

[m, n] = gbmex_size (G) ;
gbmex_wait (G) ;

if (nargout == 3)
    [I, J, X] = gbmex_extracttuples (ghb, G) ;  % FIXME remove ghb
    if (m == 1)
        I = I' ;
        J = J' ;
        X = X' ;
    end
elseif (nargout == 2)
    [I, J] = gbmex_extracttuples (ghb, G) ;
    if (m == 1)
        I = I' ;
        J = J' ;
    end
else
    if (m == 1)
        % extract indices from a row vector
        [~, I] = gbmex_extracttuples (ghb, G) ;
        I = I' ;
    elseif (n == 1)
        % extract indices from a column vector
        I = gbmex_extracttuples (ghb, G) ;
    else
        % extract linear indices from a matrix
        [I, J] = gbmex_extracttuples (ghb, G) ;
        % use the built-in sub2ind to convert the 2D indices to 1D indices
        I = sub2ind ([m n], I, J) ;
    end
end

if (nargin > 2)
    % find (G, k, ...): get the first or last k entries
    if (nargin < 4)
        search = 'first' ;
    end
    n = length (I) ;
    if (k >= n)
        % output already has all k first or last entries;
        % nothing more to do
    elseif (isequal (search, 'first'))
        % find (G, k, 'first'): get the first k entries
        I = I (1:k) ;
        if (nargout > 1)
            J = J (1:k) ;
        end
        if (nargout > 2)
            X = X (1:k) ;
        end
    elseif (isequal (search, 'last'))
        % find (G, k, 'last'): get the last k entries
        I = I (n-k+1:n) ;
        if (nargout > 1)
            J = J (n-k+1:n) ;
        end
        if (nargout > 2)
            X = X (n-k+1:n) ;
        end
    else
        error ('GrB:error', ...
            'invalid search option; must be ''first'' or ''last''') ;
    end
end

