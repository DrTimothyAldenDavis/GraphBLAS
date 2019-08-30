function f = format (arg)
%GB.FORMAT get/set the default GraphBLAS matrix format.
%
% In its ANSI C interface, SuiteSparse:GraphBLAS stores its matrices by
% row, by default, since that format tends to be fastest for graph
% algorithms, but it can also store its matrices by column.  MATLAB sparse
% and dense sparse matrices are always stored by column.  For better
% compatibility with MATLAB sparse matrices, the default for the MATLAB
% interface for SuiteSparse:GraphBLAS is to store matrices by column.  This
% has performance implications, and algorithms should be designed
% accordingly.  The default format can be can changed via:
%
%   gb.format ('by row')
%   gb.format ('by col')
%
% which changes the format of all subsequent GraphBLAS matrices.  Existing
% gb matrices are not affected.
%
% The current default global format can be queried with
%
%   f = gb.format ;
%
% which returns the string 'by row' or 'by col'.
%
% Since MATLAB sparse and dense matrices are always 'by col', converting
% them to a gb matrix 'by row' requires an internal transpose of the
% format.  That is, if A is a MATLAB sparse or dense matrix,
%
%   gb.format ('by row')
%   G = gb (A)
%
% Constructs a double gb matrix G that is held by row, but this takes more
% work than if G is held by column:
%
%   gb.format ('by col')
%   G = gb (A)
%
% If a subsequent algorithm works better with its matrices held by row,
% then this transformation can save significant time in the long run.
% Graph algorithms tend to be faster with their matrices held by row, since
% the edge (i,j) is typically the entry G(i,j) in the matrix G, and most
% graph algorithms need to know the outgoing edges of node i.  This is
% G(i,:), which is very fast if G is held by row, but very slow if G is
% held by column.
%
% When the gb.format (f) is changed, all subsequent matrices are created in
% the given format f.  All prior matrices created before gb.format (f) are
% kept in their same format; this setting only applies to new matrices.
% Operations on matrices can be done with any mix of with different
% formats.  The format only affects time and memory usage, not the results.
%
% This setting is reset to 'by col', by 'clear all' or by gb.clear.
%
% To query the format for a given GraphBLAS matrix G, use the following
% (which does not affect the global format setting):
%
%   f = gb.format (G)
%
% Use G = gb (G, 'by row') or G = gb (G, 'by col') to change the format of G.
%
% Examples:
%
%   A = sparse (rand (4))
%   gb.format ('by row') ;
%   G = gb (A)
%   gb.format (G)
%   gb.format ('by col') ;      % set the default format to 'by col'
%   G = gb (A)
%   gb.format (G)               % query the format of G
%
% See also gb.

if (nargin == 0)
    % f = gb.format ; get the global format
    f = gbformat ;
elseif (nargin == 1)
    if (isa (arg, 'gb'))
        % f = gb.format (G) ; get the format of the matrix G
        f = gbformat (arg.opaque) ;
    else
        % f = gb.format (f) ; set the global format for all future matrices
        f = gbformat (arg) ;
    end
else
    error ('usage: f = gb.format or f = gb.format (f)') ;
end

