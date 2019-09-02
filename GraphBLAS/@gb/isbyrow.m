function s = isbyrow (X)
%GB.ISBYROW True if X is stored by row, false if by column.
% s = gb.isbyrow (X) is true if X is stored by row, false if by column.
% X may be a GraphBLAS matrix or MATLAB matrix (sparse or full).  MATLAB
% matrices are always stored by column.
%
% See also gb.isbycol, gb.format.

s = isequal (gb.format (X), 'by row')  ;

