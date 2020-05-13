function C = sqrt (G)
%SQRT Square root.
% C = sqrt (G) is the square root of the elements of the GraphBLAS matrix
% G.
%
% See also GrB.apply, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

switch (GrB.type (G))

    case {'logical'}

        % logical
        C = G ;

    case {'uint8', 'uint16', 'uint32', 'uint64'}

        % unsigned integer
        C = GrB.apply ('sqrt', GrB (G, 'double')) ;

    case {'int8', 'int16', 'int32', 'int64', 'double'}

        % signed integer, or double
        if (min (G, [ ], 'all') < 0)
            C = GrB.apply ('sqrt', GrB (G, 'double complex')) ;
        else
            C = GrB.apply ('sqrt', GrB (G, 'double')) ;
        end

    case {'single'}

        % single
        if (min (G, [ ], 'all') < 0)
            C = GrB.apply ('sqrt', GrB (G, 'single complex')) ;
        else
            C = GrB.apply ('sqrt', GrB (G, 'single')) ;
        end

    otherwise

        % single complex or double complex
        C = GrB.apply ('sqrt', G) ;

end

% so that realsqrt gets the right result
if (~isreal (C) && nnz (imag (C) == 0))
    C = real (C) ;
end

