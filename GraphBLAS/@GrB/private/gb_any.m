function C = gb_any (G, option)
%GB_ANY True if any element of a GraphBLAS matrix is nonzero or true.
% Implements C = any (G, option)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

switch (option)

    case { 'default' }

        % C = any (G)
        if (gb_isvector (G))
            % C = any (G) for a vector G results in a scalar C
            C = GrB (gbreduce ('|.logical', G)) ;
        else
            % C = any (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            desc.in0 = 'transpose' ;
            C = GrB (gbtrans (gbvreduce ('|.logical', G, desc))) ;
        end

    case { 'all' }

        % C = any (G, 'all'), reducing all entries to a scalar
        C = GrB (gbreduce ('|.logical', G)) ;

    case { 1 }

        % C = any (G, 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        desc.in0 = 'transpose' ;
        C = GrB (gbtrans (gbvreduce ('|.logical', G, desc))) ;

    case { 2 }

        % C = any (G, 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = GrB (gbvreduce ('|.logical', G)) ;

    otherwise

        gb_error ('unknown option') ;

end

