function Graph = gb_graph (ghb, G_arg, varargin)
%GB_GRAPH: implements GrB.graph and GhB.graph.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gbmex_size (G_arg) ;
if (m ~= n)
    error ('GrB:error', 'G must be square') ;
end

% get the string options
side = 'lower' ;
omitself = false ;
for k = 1:nargin-2
    arg = lower (varargin {k}) ;
    switch arg
        case { 'upper', 'lower' }
            side = arg ;
        case { 'omitselfloops' }
            omitself = true ;
        otherwise
            error ('GrB:error', 'unknown option') ;
    end
end

% apply the options
if (omitself)
    % ignore diagonal entries of G
    if (isequal (side, 'upper'))
        G = gzb_select (ghb, 'triu', G_arg, 1) ;
    else
        G = gzb_select (ghb, 'tril', G_arg, -1) ;
    end
else
    % include diagonal entries of G
    if (isequal (side, 'upper'))
        G = gzb_select (ghb, 'triu', G_arg, 0) ;
    else
        G = gzb_select (ghb, 'tril', G_arg, 0) ;
    end
end

switch (type)

    case { 'single' }

        % The graph(...) function can accept x as single, but not from a
        % built-in sparse matrix.  So extract the tuples of G first.
        gbmex_wait (G) ;
        [i, j, x] = gbmex_extracttuples (ghb, G) ;
        Graph = graph (i, j, x, n) ;

    case { 'logical' }

        % The digraph(...) function allows for logical
        % adjacency matrices (no edge weights are created).
        Graph = graph (gbmex_builtin (gzb_cast (ghb, G, 'logical')), side) ;

    otherwise

        % typecast to double
        Graph = graph (gbmex_builtin (gzb_cast (ghb, G, 'double')), side) ;

end

