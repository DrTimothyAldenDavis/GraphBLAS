function s = isa (G, classname)
%ISA Determine if a GraphBLAS matrix is of specific class.
%
% For any GraphBLAS matrix G, isa (G, 'GrB'), isa (G, 'numeric'), and isa
% (G, 'object') are always true (even if G is logical, since many
% semirings are defined for that type).
%
% isa (G, 'float') is the same as isfloat (G), and is true if the GrB
% matrix G has type 'double', 'single', 'single complex', or 'double
% complex'.
%
% isa (G, 'integer') is the same as isinteger (G), and is true if the GrB
% matrix G has type 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
% 'uint32', or 'uint64'.
%
% isa (G, classname) is true if the classname matches the type of G.
%
% See also GrB.type, GrB/isnumeric, GrB/islogical, ischar, iscell,
% isstruct, GrB/isfloat, GrB/isinteger, isobject, isjava, GrB/issparse,
% GrB/isreal, class.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isequal (classname, 'GrB') || isequal (classname, 'numeric'))

    % all GraphBLAS matrices are numeric, and have class name 'GrB'
    s = true ;

else

    type = gbtype (G.opaque) ;

    switch (classname)

        case { 'float' }

            % GraphBLAS double, single, and complex matrices are 'float'
            s = contains (type, 'double') || contains (type, 'single') ;

        case { 'integer' }

            % GraphBLAS int* and uint* matrices are 'integer'
            s = contains (type, 'int') ;

        case { 'single complex', 'double complex', 'complex' }

            s = contains (type, 'complex') ;

        case { type }

            % specific cases, such as isa (G, 'double')
            s = true ;

        otherwise

            % catch-all for cases not handled above
            s = builtin ('isa', G, classname) ;

    end

end

