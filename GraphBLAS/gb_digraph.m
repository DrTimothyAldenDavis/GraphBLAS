function DiGraph = gb_digraph (ghb, G_arg, option)
%GB_DIGRAPH implements DiGraph for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gbmex_size (G_arg) ;
if (m ~= n)
    error ('GrB:error', 'G must be square') ;
end

% get the string options
omitself = false ;
if (nargin > 2)
    if (isequal (lower (option), 'omitselfloops'))
        omitself = true ;
    else
        error ('GrB:error', 'unknown option') ;
    end
end

% apply the options
if (omitself)
    % ignore diagonal entries of G
    G = gzb_select (ghb, 'offdiag', G_arg, 0) ;
else
    % use G_arg as-is
    G = G_arg ;
end

% construct the graph
switch (type)
    case { 'single', 'logical' }
        gtype = type ;
    otherwise
        gtype = 'double' ;
end

% construct the digraph
switch (type)

    case { 'single' }

        % The digraph(...) function can accept x as single, but not
        % from a sparse matrix.  So extract the tuples of G first.
        gbmex_wait (G) ;
        [i, j, x] = gbmex_extracttuples (ghb, G) ;
        DiGraph = digraph (i, j, x, n) ;

    case { 'logical' }

        % The digraph(...) function allows for logical
        % adjacency matrices (no edge weights are created).
        DiGraph = digraph (gbmex_builtin (gzb_cast (ghb, G, 'logical'))) ;

    otherwise

        % typecast to double
        DiGraph = digraph (gbmex_builtin (gzb_cast (ghb, G, 'double'))) ;
end

