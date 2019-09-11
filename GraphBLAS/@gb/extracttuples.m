function [I,J,X] = extracttuples (varargin)
%GB.EXTRACTTUPLES extract a list of entries from a matrix.
%
% Usage:
%
%   [I,J,X] = gb.extracttuples (A, desc)
%
% gb.extracttuples extracts all entries from either a MATLAB matrix or a
% GraphBLAS matrix.  If A is a MATLAB sparse or dense matrix,
% [I,J,X] = gb.extracttuples (A) is identical to [I,J,X] = find (A).
%
% The descriptor is optional.  d.base is a string, equal to 'default',
% 'zero-based', or 'one-based'.  This parameter determines the type of
% output for I and J.  The default is one-based, to be more compatible
% with MATLAB.  If one-based, then I and J are returned as MATLAB double
% column vectors, containing 1-based indices.  The indices in I are in
% the range 1 to m, and the indices in J are in the range 1 to n, if A is
% m-by-n.  This usage is identical to [I,J,X] = find (A) for a MATLAB
% sparse or dense matrix.  The array X has the same type as A (double,
% single, int8, ..., uint8, or (in the future) complex).
%
% The default is 'one based', but 'zero based' is faster and uses less
% memory.  In this case, I and J are returned as int64 arrays.  Entries
% in I are in the range 0 to m-1, and entries in J are in the range 0 to
% n-1.
%
% This function corresponds to the GrB_*_extractTuples_* functions in
% GraphBLAS.
%
% See also find, gb/build.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args, ~] = gb_get_args (varargin {:}) ;
if (nargout == 3)
    [I, J, X] = gbextracttuples (args {:}) ;
elseif (nargout == 2)
    [I, J] = gbextracttuples (args {:}) ;
else
    I = gbextracttuples (args {:}) ;
end

