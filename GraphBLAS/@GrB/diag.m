function C = diag (A, k)
%DIAG Diagonal matrices and diagonals of a GraphBLAS matrix.
% C = diag (v,k) when v is a GraphBLAS vector with n components is a
% square sparse GarphBLAS matrix of dimension n+abs(k), with the elements
% of v on the kth diagonal. The main diagonal is k = 0; k > 0 is above
% the diagonal, and k < 0 is below the main diagonal.  C = diag (v) is
% the same as C = diag (v,0).
%
% c = diag (A,k) when A is a GraphBLAS matrix returns a GraphBLAS column
% vector c formed the entries on the kth diagonal of A.  The main
% diagonal is c = diag(A).
%
% The GraphBLAS diag function always constructs a GraphBLAS sparse
% matrix, unlike the the MATLAB diag, which always constructs a MATLAB
% full matrix.
%
% A may be a MATLAB or GraphBLAS matrix.  To use this function for a
% MATLAB sparse matrix A, use C = diag (A, GrB (k)) ;
%
% Examples:
%
%   C1 = diag (GrB (1:10, 'uint8'), 2)
%   C2 = sparse (diag (1:10, 2))
%   nothing = double (C1-C2)
%
%   A = magic (8)
%   full (double ([diag(A,1) diag(GrB(A),1)]))
%
%   m = 5 ;
%   f = ones (2*m,1) ;
%   A = diag(-m:m) + diag(f,1) + diag(f,-1)
%   G = diag(GrB(-m:m)) + diag(GrB(f),1) + diag(GrB(f),-1)
%   nothing = double (A-G)
%
% See also GrB/diag, spdiags, GrB/tril, GrB/triu, GrB.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

if (nargin < 2)
    k = 0 ;
elseif (isobject (k))
    k = gb_get_scalar (k) ;
end
[am, an] = gbsize (A) ;
a_is_vector = (am == 1) || (an == 1) ;
desc.base = 'zero-based' ;

if (a_is_vector)

    % C = diag (v,k) is an m-by-m matrix if v is a vector
    n = am * an ;
    m = n + abs (k) ;

    if (am == 1)
        % A is a row vector
        if (k >= 0)
            [~, I, X] = gbextracttuples (A, desc) ;
            J = I + int64 (k) ;
        else
            [~, J, X] = gbextracttuples (A, desc) ;
            I = J - int64 (k) ;
        end
    else
        % A is a column vector
        if (k >= 0)
            [I, ~, X] = gbextracttuples (A, desc) ;
            J = I + int64 (k) ;
        else
            [J, ~, X] = gbextracttuples (A, desc) ;
            I = J - int64 (k) ;
        end
    end

    C = GrB (gbbuild (I, J, X, m, m, desc)) ;

else

    % C = diag (A,k) is a column vector formed from the elements of the kth
    % diagonal of A

    if (k >= 0)
        m = min (an-k, am) ;
    else
        m = min (an, am+k) ;
    end

    if (m == 0)

        % A does not have a kth diagonal so C is empty
        I = [ ] ;

    else

        % extract the kth diagonal from A and convert into a column vector
        C = gbselect ('diag', A, k) ;
        if (k >= 0)
            [I, ~, X] = gbextracttuples (C, desc) ;
        else
            [~, I, X] = gbextracttuples (C, desc) ;
        end

    end

    if (length (I) == 0)
        % A does not have a kth diagonal, or diag (A,k) has no entries
        C = GrB (gbnew (m, 1, gbtype (A))) ;
    else
        C = GrB (gbbuild (I, int64 (0), X, m, 1, desc)) ;
    end

end

