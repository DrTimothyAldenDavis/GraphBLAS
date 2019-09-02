function s = isbycol (X)
%GB.ISBYCOL True if X is stored by column, false if by column.
% s = gb.isbycol (X) is true if X is stored by column, false if by row.
% X may be a GraphBLAS matrix or MATLAB matrix (sparse or full).  MATLAB
% matrices are always stored by column.
%
% See also gb.isbyrow, gb.format.

s = isequal (gb.format (X), 'by col')  ;

