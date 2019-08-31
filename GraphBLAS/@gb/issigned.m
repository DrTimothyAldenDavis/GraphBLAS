function s = issigned (type)
%GB.ISSIGNED Determine if a type is signed or unsigned.
% s = gb.issigned (type) returns true if type is the string 'double',
% 'single', 'int8', 'int16', 'int32', or 'int64'.
%
% See also isinteger, isreal, isnumeric, isfloat.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

s = isequal (type, 'double') || isequal (type, 'single') || ...
    isequal (type (1:3), 'int') ;

