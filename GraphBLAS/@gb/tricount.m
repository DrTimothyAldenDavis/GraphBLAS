function s = tricount (A, check)
%GB.TRICOUNT count triangles in a matrix.
% s = gb.tricount (A) counts the number of triangles in the matrix A.
% spones (A) must be symmetric; results are undefined if spones (A) is
% unsymmetric.  Diagonal entries are ignored.
%
% To check the input matrix A, use gb.tricount (A, 'check').  This check takes
% additional time so by default the input is not checked.
%
% See also gb.ktruss.

[m, n] = size (A) ;
if (m ~= n)
    gb_error ('A must be square') ;
end
if (nargin < 2)
    check = false ;
else
    check = isequal (check, 'check') ;
end

int_type = 'int64' ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
end
A = spones (A, int_type) ;

if (check && ~issymmetric (A))
    gb_error ('spones (A) must be symmetric') ;
end

C = gb (n, n, int_type, gb.format (A)) ;
L = tril (A, -1) ;
U = triu (A, 1) ;

% Inside GraphBLAS, the methods below are identical.  For example, L stored by
% row is the same data structure as U stored by column.

if (gb.isbyrow (A))
    % C<L> = L*U'
    C = gb.mxm (C, L, '+.*', L, U, struct ('in1', 'transpose')) ;
else
    % C<U> = L'*U
    C = gb.mxm (C, U, '+.*', L, U, struct ('in0', 'transpose')) ;
end

s = full (double (gb.reduce ('+.int64', C))) ;

