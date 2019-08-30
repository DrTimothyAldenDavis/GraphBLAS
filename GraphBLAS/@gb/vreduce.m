function Cout = vreduce (varargin)
%GB.REDUCE reduce a matrix to a vector
%
% Usage:
%
%   Cout = gb.vreduce (monoid, A)
%   Cout = gb.vreduce (monoid, A, desc)
%   Cout = gb.vreduce (Cin, M, monoid, A)
%   Cout = gb.vreduce (Cin, M, monoid, A, desc)
%   Cout = gb.vreduce (Cin, accum, monoid, A)
%   Cout = gb.vreduce (Cin, accum, monoid, A, desc)
%   Cout = gb.vreduce (Cin, M, accum, monoid, A)
%   Cout = gb.vreduce (Cin, M, accum, monoid, A, desc)
%
% The monoid and A arguments are required.  All others are optional.
% The valid monoids are: '+', '*', 'max', and 'min' for all but the
% 'logical' type, and '|', '&', 'xor', and 'ne' for the 'logical' type.
% See 'help gb.monoidinfo' for more details.
%
% By default, each row of A is reduced to a scalar.  If Cin is not present,
% Cout (i) = reduce (A (i,:)).  In this case, Cin and Cout are column
% vectors of size m-by-1, where A is m-by-n.  If desc.in0 is 'transpose',
% then A.' is reduced to a column vector; Cout (j) = reduce (A (:,j)).
% In this case, Cin and Cout are column vectors of size n-by-1, if A is
% m-by-n.
%
% See also gb.reduce, sum, prod, max, min.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbvreduce (args {:})) ;
else
    Cout = gbvreduce (args {:}) ;
end

