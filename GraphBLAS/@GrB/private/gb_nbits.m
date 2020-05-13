function n = gb_nbits (a)
%GB_NBITS return number of bits in a type
% n = gb_nbits (a) returns the number of bits in a type.  For example,
% gb_nbits ('int32') is 32.
%
% The input a can be a string with a type, or a matrix (whose type is
% queried).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% get the type
if (ischar (a))
    atype = a ;
else
    atype = GrB.type (a) ;
end

% get the number of bits in the type
switch (atype)
    case { 'logical' }
        n = 1 ;
    case { 'int8', 'uint8' }
        n = 8 ;
    case { 'int16', 'uint16' }
        n = 16 ;
    case { 'int32', 'uint32', 'single' }
        n = 32 ;
    case { 'int64', 'uint64', 'double', 'single complex' }
        n = 64 ;
    case { 'double complex' }
        n = 128 ;
    otherwise
        error ('unknown type') ;
end


