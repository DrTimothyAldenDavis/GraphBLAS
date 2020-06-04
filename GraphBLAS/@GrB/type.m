function s = type (X)
%GRB.TYPE get the type of a MATLAB or GraphBLAS matrix.
% s = GrB.type (X) returns the type of a GraphBLAS matrix X as a string:
% 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
% 'uint32', 'uint64', 'single', 'double', 'single complex', and 'double
% complex' (the latter can be specified as just 'complex').  Note that
% 'complex' is treated as a type, not an attribute, which differs from the
% MATLAB convention.
%
% If X is not a GraphBLAS matrix, GrB.type (X) is the same as class (X),
% except when X is a MATLAB single complex or double complex matrix, which
% case GrB.type (X) is 'single complex' or 'double complex', respectively.
%
% See also class, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (X))
    X = X.opaque ;
end

s = gbtype (X) ;

