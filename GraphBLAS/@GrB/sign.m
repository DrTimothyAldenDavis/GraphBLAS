function C = sign (G)
%SIGN Signum function.
% C = sign (G) computes the signum function for each entry in the
% GraphBLAS matrix G.  For each element of G, sign(G) returns 1 if the
% element is greater than zero, 0 if it equals zero, and -1 if it is less
% than zero.  For the complex case, C = G ./ abs (G).
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% FUTURE: this would be faster if GraphBLAS had a built-in unary
% GxB_SIGN_[TYPE] operator.

G = G.opaque ;
type = gbtype (G) ;

% prune explicit zeros
G = gbselect (G, 'nonzero') ;

switch (type)

    case { 'double', 'single' }

        % single or double real
        C = GrB (gbemult ('copysign', gb_spones (G, type), G)) ;

    case { 'int8', 'int16', 'int32', 'int64' }

        % signed integer
        Pos = gb_spones (gbselect (G, '>0'), type) ;
        Neg = gb_spones (gbselect (G, '<0'), type) ;
        C = GrB (gbeadd (Pos, '+', gbapply ('-', Neg))) ;

    case { 'single complex', 'double complex' }

        % single or double complex
        C = GrB (gbemult (G, '/', gbapply ('abs', G))) ;

    case { 'uint8', 'uint16', 'uint32', 'uint64' }

        % unsigned integer
        C = GrB (gb_spones (G, type)) ;

    case { 'logical' }

        % logical
        C = GrB (G) ;

end

