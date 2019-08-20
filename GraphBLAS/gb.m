classdef gb
%GB GraphBLAS sparse matrices for MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Visit http://graphblas.org for more
% details and resources.  See also the SuiteSparse:GraphBLAS User Guide in this
% package.
%
% The MATLAB gb class represents a GraphBLAS sparse matrix.  The gb method
% creates a GraphBLAS sparse matrix from a MATLAB matrix.  Other methods also
% generate gb matrices.  For example G = gb.subassign (C, M, A) constructs a
% GraphBLAS matrix G, which is the result of C<M>=A in GraphBLAS notation (like
% C(M)=A(M) in MATLAB).  The matrices used any gb.method may be MATLAB matrices
% (sparse or dense) or GraphBLAS sparse matrices, in any combination.
%
% The gb constructor creates a GraphBLAS matrix.  The input X may be any
% MATLAB or GraphBLAS matrix:
%
%   G = gb (X) ;            GraphBLAS copy of a matrix X, same type
%   G = gb (X, type) ;      GraphBLAS typecasted copy of matrix X
%   G = gb (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   G = gb (m, n, type) ;   m-by-n GraphBLAS matrix of given type, no entries
%
% The m and n parameters above are MATLAB scalars.  The type is a string.  The
% default format is by column, to match the format used in MATLAB (see
% gb.format to get/set the default GraphBLAS format).
%
% The usage G = gb (m, n, type) is analgous to X = sparse (m, n), which creates
% an empty MATLAB sparse matrix X.  The type parameter is a string, which
% defaults to 'double' if not present.
%
% For the usage G = gb (X, type), X is either a MATLAB sparse or dense matrix,
% or a GraphBLAS sparse matrix object.  G is created as a GraphBLAS sparse
% matrix object that contains a copy of X, typecasted to the given type if the
% type string does not match the type of X.  If the type string is not present
% it defaults to 'double'.
%
% Most of the valid type strings correspond to MATLAB class of the same name
% (see 'help class'), with the addition of the 'complex' type:
%
%   'double'    64-bit floating-point (real, not complex)
%   'single'    32-bit floating-point (real, not complex)
%   'logical'   8-bit boolean
%   'int8'      8-bit signed integer
%   'int16'     16-bit signed integer
%   'int32'     32-bit signed integer
%   'int64'     64-bit signed integer
%   'uint8'     8-bit unsigned integer
%   'uint16'    16-bit unsigned integer
%   'uint32'    32-bit unsigned integer
%   'uint64'    64-bit unsigned integer
%   'complex'   64-bit double complex.  In MATLAB, this is not a MATLAB
%               class name, but instead a property of a sparse double matrix.
%               FUTURE: In GraphBLAS, 'complex' will be treated as a type, but
%               complex matrices are not yet supported.
%
% Operations on integer values differ from MATLAB.  In MATLAB, uint9(255)+1 is
% 255, since the arithmetic saturates.  This is not possible in matrix
% operations such as C=A*B, since saturation of integer arithmetic would render
% most of the monoids useless.  GraphBLAS instead computes a result modulo the
% word size, so that gb(uint8(255))+1 is zero.  However, new unary and binary
% operators could be added so that element-wise operations saturate.  The C
% interface allows for arbitrary creation of user-defined operators, so this
% could be added in the future.
%
% Methods for the gb class:
%
%   These methods operate on GraphBLAS matrices only, and they overload the
%   existing MATLAB functions of the same name.
%
%       S = sparse (G)          makes a copy of a gb matrix
%       F = full (G)            adds explicit zeros to a gb matrix
%       F = full (G,id)         adds explicit identity values to a gb matrix
%       S = double (G)          cast gb matrix to MATLAB sparse double matrix
%       C = logical (G)         cast gb matrix to MATLAB sparse logical matrix
%       C = complex (G)         cast gb matrix to MATLAB sparse complex (FUTURE)
%       C = single (G)          cast gb matrix to MATLAB full single matrix
%       C = int8 (G)            cast gb matrix to MATLAB full int8 matrix
%       C = int16 (G)           cast gb matrix to MATLAB full int16 matrix
%       C = int32 (G)           cast gb matrix to MATLAB full int32 matrix
%       C = int64 (G)           cast gb matrix to MATLAB full int64 matrix
%       C = uint8 (G)           cast gb matrix to MATLAB full uint8 matrix
%       C = uint16 (G)          cast gb matrix to MATLAB full uint16 matrix
%       C = uint32 (G)          cast gb matrix to MATLAB full uint32 matrix
%       C = uint64 (G)          cast gb matrix to MATLAB full uint64 matrix
%       C = cast (G,...)        cast gb matrix to MATLAB matrix (as above)
%       [I,J,X] = find (G)      extract all entries from a gb matrix
%       X = nonzeros (G)        extract all entries from a gb matrix
%       C = spones (G)          return pattern of gb matrix
%       disp (G, level)         display a gb matrix G
%       display (G)             display a gb matrix G; same as disp(G,2)
%       mn = numel (G)          m*n for an m-by-n gb matrix G
%       e = nnz (G)             number of entries in a gb matrix G
%       e = nzmax (G)           number of entries in a gb matrix G
%       [m n] = size (G)        size of a gb matrix G
%       n = length (G)          length of a gb vector
%       s = isempty (G)         true if any dimension of G is zero
%       s = issparse (G)        true for any gb matrix G
%       s = ismatrix (G)        true for any gb matrix G
%       s = isvector (G)        true if m=1 or n=1, for an m-by-n gb matrix G
%       s = isscalar (G)        true if G is a 1-by-1 gb matrix
%       s = isnumeric (G)       true for any gb matrix G (even logical)
%       s = isfloat (G)         true if gb matrix is double, single, or complex
%       s = isreal (G)          true if gb matrix is not complex
%       s = isinteger (G)       true if gb matrix is int8, int16, ..., uint64
%       s = islogical (G)       true if gb matrix is logical
%       s = isa (G, classname)  determine if a gb matrix is of a specific class
%       C = diag (G,k)          diagonal matrices and diagonals of a gb matrix G
%       L = tril (G,k)          lower triangular part of gb matrix G
%       U = triu (G,k)          upper triangular part of gb matrix G
%       C = kron (A,B)          Kronecker product
%       C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%       C = reshape (G, ...)    reshape a GraphBLAS matrix
%       C = abs (G)             absolute value
%       C = sign (G)            signum function
%       s = istril (G)          true if G is lower triangular
%       s = istriu (G)          true if G is upper triangular
%       s = isbanded (G,...)    true if G is banded
%       s = isdiag (G)          true if G is diagonal
%       s = ishermitian (G)     true if G is Hermitian
%       s = issymmetric (G)     true if G is symmetric
%       [lo,hi] = bandwidth (G) determine the lower & upper bandwidth of G
%       C = sqrt (G)            element-wise square root
%       C = sum (G, option)     reduce via sum, to vector or scalar
%       C = prod (G, option)    reduce via product, to vector or scalar
%       s = norm (G, kind)      1-norm or inf-norm of a gb matrix
%       C = max (G, ...)        reduce via max, to vector or scalar
%       C = min (G, ...)        reduce via min, to vector or scalar
%       C = any (G, ...)        reduce via '|', to vector or scalar
%       C = all (G, ...)        reduce via '&', to vector or scalar
%       C = eps (G)             floating-point spacing
%       C = ceil (G)            round towards infinity
%       C = floor (G)           round towards -infinity
%       C = round (G)           round towards nearest
%       C = fix (G)             round towards zero
%       C = isfinite (G)        test if finite
%       C = isinf (G)           test if infinite
%       C = isnan (G)           test if NaN
%       C = spfun (fun, G)      evaluate a function on the entries of G
%       p = amd (G)             approximate minimum degree ordering
%       p = colamd (G)          column approximate minimum degree ordering
%       p = symamd (G)          approximate minimum degree ordering
%       p = symrcm (G)          reverse Cuthill-McKee ordering
%       [...] = dmperm (G)      Dulmage-Mendelsohn permutation
%       C = conj (G)            complex conjugate
%       C = real (G)            real part of a complex GraphBLAS matrix
%       [V, ...] = eig (G,...)  eigenvalues and eigenvectors
%
%   operator overloading:
%
%       C = plus (A, B)         C = A + B
%       C = minus (A, B)        C = A - B
%       C = uminus (A)          C = -A
%       C = uplus (A)           C = +A
%       C = times (A, B)        C = A .* B
%       C = mtimes (A, B)       C = A * B
%       C = rdivide (A, B)      C = A ./ B
%       C = ldivide (A, B)      C = A .\ B
%       C = mrdivide (A, B)     C = A / B
%       C = mldivide (A, B)     C = A \ B
%       C = power (A, B)        C = A .^ B
%       C = mpower (A, B)       C = A ^ B
%       C = lt (A, B)           C = A < B
%       C = gt (A, B)           C = A > B
%       C = le (A, B)           C = A <= B
%       C = ge (A, B)           C = A >= B
%       C = ne (A, B)           C = A ~= B
%       C = eq (A, B)           C = A == B
%       C = and (A, B)          C = A & B
%       C = or (A, B)           C = A | B
%       C = not (A)             C = ~A
%       C = ctranspose (A)      C = A'
%       C = transpose (A)       C = A.'
%       C = horzcat (A, B)      C = [A , B]
%       C = vertcat (A, B)      C = [A ; B]
%       C = subsref (A, I, J)   C = A (I,J) or C = A (M)
%       C = subsasgn (A, I, J)  C (I,J) = A
%       index = end (A, k, n)   for object indexing, A(1:end,1:end)
%
% Static Methods:
%
%       The Static Methods for the gb class can be used on input matrices of
%       any kind: GraphBLAS sparse matrices, MATLAB sparse matrices, or MATLAB
%       dense matrices, in any combination.  The output matrix Cout is a
%       GraphBLAS matrix, by default, but can be optionally returned as a
%       MATLAB sparse or dense matrix.  The static methods divide into two
%       categories: those that perform basic functions, and the GraphBLAS
%       operations that use the mask/accum.
%
%   GraphBLAS basic functions:
%
%       gb.clear                    clear GraphBLAS workspace and settings
%       gb.descriptorinfo (d)       list properties of a descriptor
%       gb.unopinfo (op, type)      list properties of a unary operator
%       gb.binopinfo (op, type)     list properties of a binary operator
%       gb.monoidinfo (op, type)    list properties of a monoid
%       gb.semiringinfo (s, type)   list properties of a semiring
%       t = gb.threads (t)          set/get # of threads to use in GraphBLAS
%       c = gb.chunk (c)            set/get chunk size to use in GraphBLAS
%       e = gb.nvals (A)            number of entries in a matrix
%       G = gb.empty (m, n)         return an empty GraphBLAS matrix
%       s = gb.type (X)             get the type of a MATLAB or gb matrix X
%       f = gb.format (f)           set/get matrix format to use in GraphBLAS
%       C = gb.expand (scalar, S)   expand a scalar (C = scalar*spones(S))
%       C = gb.eye                  identity matrix of any type
%       C = gb.speye                identity matrix (of type 'double')
%
%       G = gb.build (I, J, X, m, n, dup, type, desc)
%                           build a GraphBLAS matrix from a list of entries
%       [I,J,X] = gb.extracttuples (A, desc)
%                           extract all entries from a matrix
%
%   GraphBLAS operations (as Static methods) with Cout, mask M, and accum:
%
%       Cout = gb.mxm (Cin, M, accum, semiring, A, B, desc)
%                           sparse matrix-matrix multiplication over a semiring
%       Cout = gb.select (Cin, M, accum, op, A, thunk, desc)
%                           select a subset of entries from a matrix
%       Cout = gb.assign (Cin, M, accum, A, I, J, desc)
%                           sparse matrix assignment, such as C(I,J)=A
%       Cout = gb.subassign (Cin, M, accum, A, I, J, desc)
%                           sparse matrix assignment, such as C(I,J)=A
%       Cout = gb.vreduce (Cin, M, accum, op, A, desc)
%                           reduce a matrix to a vector
%       Cout = gb.reduce (Cin, accum, op, A, desc)
%                           reduce a matrix to a scalar
%       Cout = gb.gbkron (Cin, M, accum, op, A, B, desc)
%                           Kronecker product
%       Cout = gb.gbtranspose (Cin, M, accum, A, desc)
%                           transpose a matrix
%       Cout = gb.eadd (Cin, M, accum, op, A, B, desc)
%                           element-wise addition
%       Cout = gb.emult (Cin, M, accum, op, A, B, desc)
%                           element-wise multiplication
%       Cout = gb.apply (Cin, M, accum, op, A, desc)
%                           apply a unary operator
%       Cout = gb.extract (Cin, M, accum, A, I, J, desc)
%                           extract submatrix, like C=A(I,J) in MATLAB
%
%       GraphBLAS operations (with Cout, Cin arguments) take the following form:
%
%           C<#M,replace> = accum (C, operation (A or A', B or B'))
%
%       C is both an input and output matrix.  In this MATLAB interface to
%       GraphBLAS, it is split into Cin (the value of C on input) and Cout (the
%       value of C on output).  M is the optional mask matrix, and #M is either
%       M or !M depending on whether or not the mask is complemented via the
%       desc.mask option.  The replace option is determined by desc.out; if
%       present, C is cleared after it is used in the accum operation but
%       before the final assignment.  A and/or B may optionally be transposed
%       via the descriptor fields desc.in0 and desc.in1, respectively.  See
%       gb.descriptorinfo for more details.
%
%       accum is optional; if not is not present, then the operation becomes
%       C<...> = operation(A,B).  Otherwise, C = C + operation(A,B) is computed
%       where '+' is the accum operator.  It acts like a sparse matrix addition
%       (see gb.eadd), in terms of the structure of the result C, but any
%       binary operator can be used.
%
%       The mask M acts like MATLAB logical indexing.  If M(i,j)=1 then C(i,j)
%       can be modified; if zero, it cannot be modified by the operation.
%
% See also sparse, doc sparse, and https://twitter.com/DocSparse .

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

properties (SetAccess = private, GetAccess = private)
    % The struct contains the entire opaque content of a GraphBLAS GrB_Matrix.
    opaque = [ ] ;
end

%===============================================================================
methods %=======================================================================
%===============================================================================

%===============================================================================
% gb: construct a GraphBLAS sparse matrix object
%===============================================================================

    function G = gb (varargin)
    %GB GraphBLAS constructor: create a GraphBLAS sparse matrix.
    %
    %   G = gb (X) ;          gb copy of a matrix X, same type
    %   G = gb (X, type) ;    gb typecasted copy of a matrix X
    %   G = gb (m, n) ;       empty m-by-n gb double matrix
    %   G = gb (m, n, type) ; empty m-by-n gb matrix of given type
    %
    % See also sparse.
    if (nargin == 1 && ...
        (isstruct (varargin {1}) && isfield (varargin {1}, 'GraphBLAS')))
        % G = gb (X), where the input X is a GraphBLAS struct as returned by
        % another gb* function, but this usage is not meant for the end-user.
        % It is only used in gb.m.  See for example mxm below, which uses G =
        % gb (gbmxm (args)), and the typecasting methods, C = double (G), etc.
        % The output of gb is a GraphBLAS object.
        G.opaque = varargin {1} ;
    else
        if (isa (varargin {1}, 'gb'))
            % extract the contents of the gb object as its opaque struct so
            % the gbnew function can access it.
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

%===============================================================================
% basic methods ================================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % sparse: make a copy of a GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function S = sparse (G)
    %SPARSE make a copy of a GraphBLAS sparse matrix.
    % Since G is already sparse, S = sparse (G) simply makes a copy of G.
    % Explicit zeros are not removed.
    %
    % See also issparse, full, gb.type, gb.
    S = gb (G) ;
    end

    %---------------------------------------------------------------------------
    % full: convert a matrix into a GraphBLAS dense matrix
    %---------------------------------------------------------------------------

    function F = full (X, type, identity)
    %FULL convert a matrix into a GraphBLAS dense matrix.
    % F = full (X, type, identity) converts the matrix X into a GraphBLAS dense
    % matrix F of the given type, by inserting identity values.  The type may
    % be any GraphBLAS type: 'double', 'single', 'logical', 'int8' 'int16'
    % 'int32' 'int64' 'uint8' 'uint16' 'uint32' 'uint64', or in the future,
    % 'complex'.  If not present, the type defaults to the same type as G, and
    % the identity defaults to zero.  X may be any matrix (GraphBLAS, MATLAB
    % sparse or full).  To use this method for a MATLAB matrix A, use a
    % GraphBLAS identity value such as gb(0), or use F = full (gb (A)).  Note
    % that issparse (F) is true, since issparse (G) is true for any GraphBLAS
    % matrix G.
    %
    % Examples:
    %
    %   G = gb (sprand (5, 5, 0.5))         % GraphBLAS sparse matrix
    %   F = full (G)                        % add explicit zeros
    %   F = full (G, 'double', inf)         % add explicit inf's
    %
    %   A = speye (2) ;
    %   F = full (A, 'double', gb(0)) ;     % full gb matrix F, from A
    %   F = full (gb (A)) ;                 % same matrix F
    %
    % See also issparse, sparse, cast, gb.type, gb.
    if (isa (X, 'gb'))
        X = X.opaque ;
    end
    if (nargin < 2)
        type = gbtype (X) ;
    end
    if (nargin < 3)
        identity = 0 ;
    end
    if (isa (identity, 'gb'))
        identity = identity.opaque ;
    end
    F = gb (gbfull (X, type, identity, struct ('kind', 'gb'))) ;
    end

    %---------------------------------------------------------------------------
    % double, logical, complex: typecast to a MATLAB sparse matrix
    %---------------------------------------------------------------------------

    % These methods typecast the GraphBLAS matrix G to a MATLAB sparse matrix,
    % either double, double complex, or logical.

    function C = double (G)
    %DOUBLE cast a GraphBLAS sparse matrix to a MATLAB sparse double matrix.
    % C = double (G) typecasts the GraphBLAS matrix G into a MATLAB sparse
    % double matrix C, either real or complex.
    %
    % To typecast the matrix G to a GraphBLAS sparse double (real) matrix
    % instead, use C = gb (G, 'double').
    %
    % To typecast the matrix G to a GraphBLAS sparse double (complex) matrix
    % instead, use C = gb (G, 'complex') (... not yet supported).
    %
    % See also cast, gb, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint16, uint32, and uint64.

    if (isreal (G))
        C = gbsparse (G.opaque, 'double') ;
    else
        C = gbsparse (G.opaque, 'complex') ;
    end
    end

    function C = logical (G)
    %LOGICAL typecast a GraphBLAS sparse matrix to MATLAB sparse logical matrix.
    % C = logical (G) typecasts the GraphBLAS matrix G to into a MATLAB
    % sparse logical matrix.
    %
    % To typecast the matrix G to a GraphBLAS sparse logical matrix instead,
    % use C = gb (G, 'logical').
    %
    % See also cast, gb, double, complex, single, int8, int16, int32, int64,
    % uint8, uint16, uint32, and uint64.
    C = gbsparse (G.opaque, 'logical') ;
    end

    function C = complex (A,B)
    %COMPLEX cast a GraphBLAS matrix to MATLAB sparse double complex matrix.
    % C = complex (G) will typecast the GraphBLAS matrix G to into a MATLAB
    % sparse logical matrix.
    %
    % To typecast the matrix G to a GraphBLAS sparse complex matrix instead,
    % use C = gb (G, 'complex').
    %
    % See also cast, gb, double, single, logical, int8, int16, int32, int64,
    % uint8, uint16, uint32, and uint64.
    error ('complex type not yet supported') ;
    end

    %---------------------------------------------------------------------------
    % single, int8, int16, int32, int64, uint8, uint16, uint32, uint64
    %---------------------------------------------------------------------------

    % These methods typecast the GraphBLAS matrix G to a MATLAB full matrix of
    % the same type, since MATLAB does not support sparse matrices of these
    % types.   Explicit zeros are inserted for entries that do not appear in G.

    function C = single (G)
    %SINGLE cast a GraphBLAS matrix to MATLAB full single matrix.
    % C = single (G) typecasts the gb matrix G to a MATLAB full single matrix.
    % The result C is full since MATLAB does not support sparse single matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse single matrix instead,
    % use C = gb (G, 'single').
    %
    % See also gb, double, complex, logical, int8, int16, int32, int64, uint8,
    % uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'single') ;
    end

    function C = int8 (G)
    %INT8 cast a GraphBLAS matrix to MATLAB full int8 matrix.
    % C = int8 (G) typecasts the gb matrix G to a MATLAB full int8 matrix.
    % The result C is full since MATLAB does not support sparse int8 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse int8 matrix instead,
    % use C = gb (G, 'int8').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'int8') ;
    end

    function C = int16 (G)
    %INT16 cast a GraphBLAS matrix to MATLAB full int16 matrix.
    % C = int16 (G) typecasts the gb matrix G to a MATLAB full int16 matrix.
    % The result C is full since MATLAB does not support sparse int16 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse int16 matrix instead,
    % use C = gb (G, 'int16').
    %
    % See also gb, double, complex, single, logical, int8, int32, int64,
    % uint8, uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'int16') ;
    end

    function C = int32 (G)
    %INT32 cast a GraphBLAS matrix to MATLAB full int32 matrix.
    % C = int32 (G) typecasts the gb matrix G to a MATLAB full int32 matrix.
    % The result C is full since MATLAB does not support sparse int32 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse int32 matrix instead,
    % use C = gb (G, 'int32').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'int32') ;
    end

    function C = int64 (G)
    %INT64 cast a GraphBLAS matrix to MATLAB full int64 matrix.
    % C = int64 (G) typecasts the gb matrix G to a MATLAB full int64 matrix.
    % The result C is full since MATLAB does not support sparse int64 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse int64 matrix instead,
    % use C = gb (G, 'int64').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, uint8,
    % uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'int64') ;
    end

    function C = uint8 (G)
    %UINT8 cast a GraphBLAS matrix to MATLAB full uint8 matrix.
    % C = uint8 (G) typecasts the gb matrix G to a MATLAB full uint8 matrix.
    % The result C is full since MATLAB does not support sparse uint8 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse uint8 matrix instead,
    % use C = gb (G, 'uint8').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint16, uint32, and uint64.
    C = gbfull (G.opaque, 'uint8') ;
    end

    function C = uint16 (G)
    %UINT16 cast a GraphBLAS matrix to MATLAB full uint16 matrix.
    % C = uint16 (G) typecasts the gb matrix G to a MATLAB full uint16 matrix.
    % The result C is full since MATLAB does not support sparse uint16 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse uint16 matrix instead,
    % use C = gb (G, 'uint16').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint32, and uint64.
    C = gbfull (G.opaque, 'uint16') ;
    end

    function C = uint32 (G)
    %UINT32 cast a GraphBLAS matrix to MATLAB full uint32 matrix.
    % C = uint32 (G) typecasts the gb matrix G to a MATLAB full uint32 matrix.
    % The result C is full since MATLAB does not support sparse uint32 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse uint32 matrix instead,
    % use C = gb (G, 'uint32').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint16, and uint64.
    C = gbfull (G.opaque, 'uint32') ;
    end

    function C = uint64 (G)
    %UINT64 cast a GraphBLAS matrix to MATLAB full uint64 matrix.
    % C = uint64 (G) typecasts the gb matrix G to a MATLAB full uint64 matrix.
    % The result C is full since MATLAB does not support sparse uint64 matrices.
    %
    % To typecast the matrix G to a GraphBLAS sparse uint64 matrix instead,
    % use C = gb (G, 'uint64').
    %
    % See also gb, double, complex, single, logical, int8, int16, int32, int64,
    % uint8, uint16, and uint32.
    C = gbfull (G.opaque, 'uint64') ;
    end

    %---------------------------------------------------------------------------
    % cast: cast a GraphBLAS matrix
    %---------------------------------------------------------------------------

    % C = cast (G, newclass)
    % C = cast (G, 'like', Y)

    % This method is implicitly defined.  cast (G, class) works for a GraphBLAS
    % matrix G and the MATLAB classes 'double', 'single', 'logical', 'int8',
    % 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', and 'uint64'.

    %---------------------------------------------------------------------------
    % nonzeros: extract entries from a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function X = nonzeros (G)
    %NONZEROS extract entries from a GraphBLAS matrix.
    % X = nonzeros (G) extracts the entries from a GraphBLAS matrix G.  X has
    % the same type as G ('double', 'single', 'int8', ...).  If G contains
    % explicit entries with a value of zero, these are dropped from X.  Use
    % gb.extracttuples to return those entries.
    %
    % See also gb.extracttuples, find.
    T = gbselect ('nonzero', G.opaque, struct ('kind', 'gb')) ;
    X = gbextractvalues (T) ;
    end

    %---------------------------------------------------------------------------
    % find: extract entries from a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function [I, J, X] = find (G)
    %FIND extract entries from a GraphBLAS matrix.
    % [I, J, X] = find (G) extracts the entries from a GraphBLAS matrix G.
    % X has the same type as G ('double', 'single', 'int8', ...).  I and J are
    % returned as 1-based indices, the same as [I,J,X] = find (S) for a MATLAB
    % matrix.  Use gb.extracttuples to return I and J as zero-based.  Linear 1D
    % indexing (I = find (S) for the MATLAB matrix S) and find (G, k, ...) are
    % not supported.  G may contain explicit entries, but these are dropped
    % from the output [I,J,X].  Use gb.extracttuples to return those entries.
    %
    % See also sparse, gb.build, gb.extracttuples.
    T = gbselect ('nonzero', G.opaque, struct ('kind', 'gb')) ;
    if (nargout == 3)
        [I, J, X] = gbextracttuples (T) ;
    elseif (nargout == 2)
        [I, J] = gbextracttuples (T) ;
    else
        I = gbextracttuples (T) ;
    end
    end

    %---------------------------------------------------------------------------
    % spones: return pattern of GraphBLAS matrix
    %---------------------------------------------------------------------------

    function C = spones (G, type)
    %SPONES return pattern of GraphBLAS matrix.
    % The behavior of spones (G) for a gb matrix differs from spones (S) for a
    % MATLAB matrix S.  An explicit entry G(i,j) that has a value of zero is
    % converted to the explicit entry C(i,j)=1; these entries never appear in
    % spones (S) for a MATLAB matrix S.  C = spones (G) returns C as the same
    % type as G.  C = spones (G,type) returns C in the requested type
    % ('double', 'single', 'int8', ...).  For example, use C = spones (G,
    % 'logical') to return the pattern of G as a sparse logical matrix.
    if (nargin == 1)
        C = gb.apply ('1', G) ;
    else
        if (~ischar (type))
            error ('type must be a string') ;
        end
        C = gb.apply (['1.' type], G) ;
    end
    end

    %---------------------------------------------------------------------------
    % disp: display the contents of a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function disp (G, level)
    %DISP display the contents of a GraphBLAS object.
    % disp (G, level) displays the GraphBLAS sparse matrix G.  Level controls
    % how much is printed; 0: none, 1: terse, 2: a few entries, 3: all.  The
    % default is 2 if level is not present.
    if (nargin < 2)
        level = 2 ;
    end
    if (level > 0)
        name = inputname (1) ;
        if (~isempty (name))
            fprintf ('\n%s =\n', name) ;
        end
    end
    gbdisp (G.opaque, level) ;
    end

    %---------------------------------------------------------------------------
    % display: display the contents of a GraphBLAS matrix.
    %---------------------------------------------------------------------------

    function display (G)
    %DISPLAY display the contents of a GraphBLAS object.
    % display (G) displays the attributes and first few entries of a GraphBLAS
    % sparse matrix object.  Use disp(G,3) to display all of the content of G.
    name = inputname (1) ;
    if (~isempty (name))
        fprintf ('\n%s =\n', name) ;
    end
    gbdisp (G.opaque, 2) ;
    end

    %---------------------------------------------------------------------------
    % numel: number of elements in a GraphBLAS matrix, m * n
    %---------------------------------------------------------------------------

    function result = numel (G)
    %NUMEL the maximum number of entries a GraphBLAS matrix can hold.
    % numel (G) is m*n for the m-by-n GraphBLAS matrix G.
    [m, n] = size (G) ;
    result = m*n ;
    end

    %---------------------------------------------------------------------------
    % nnz: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function e = nnz (G)
    %NNZ the number of entries in a GraphBLAS matrix.
    % nnz (G) is the same as gb.nvals (G); some of the entries may actually be
    % explicit zero-valued entries.  See 'help gb.nvals' for more details.
    % To count the number of entries of G that have a nonzero value, use
    % nnz (sparse (G)).
    %
    % See also gb.nvals, nonzeros, size, numel.
    e = gbnvals (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % nzmax: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function e = nzmax (G)
    %NZMAX the number of entries in a GraphBLAS matrix.
    e = max (gbnvals (G.opaque), 1) ;
    end

    %---------------------------------------------------------------------------
    % size: number of rows and columns in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function [arg1 n] = size (G, dim)
    %SIZE the dimensions of a GraphBLAS matrix.
    % [m n] = size (G) is the size of an m-by-n GraphBLAS sparse matrix.
    if (nargout <= 1)
        arg1 = gbsize (G.opaque) ;
        if (nargin == 2)
            arg1 = arg1 (dim) ;
        end
    else
        [arg1 n] = gbsize (G.opaque) ;
    end
    end

    %---------------------------------------------------------------------------
    % length: length of a GraphBLAS vector
    %---------------------------------------------------------------------------

    function n = length (G)
    %LENGTH the length of a GraphBLAS vector.
    % length (G) is the length of the vector G.  For matrices, it is
    % max (size (G)) if G is non-empty, or zero if G has any zero dimension.
    [m n] = size (G) ;
    if (m == 0 | n == 0)
        n = 0 ;
    else
        n = max (m, n) ;
    end
    end

    %---------------------------------------------------------------------------
    % isempty: true if any dimension of G is zero
    %---------------------------------------------------------------------------

    function s = isempty (G)
    %ISEMPTY true for empty array.
    [m n] = size (G) ;
    s = (m == 0) | (n == 0) ;
    end

    %---------------------------------------------------------------------------
    % issparse: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = issparse (G)
    %ISSPARSE always true for any GraphBLAS matrix.
    % issparse (G) is always true for any GraphBLAS matrix G.
    %
    % See also ismatrix, isvector, isscalar, sparse, full, isa, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % ismatrix: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = ismatrix (G)
    %ISMATRIX always true for any GraphBLAS matrix.
    % ismatrix (G) is always true for any GraphBLAS matrix G.
    %
    % See also issparse, isvector, isscalar, sparse, full, isa, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isvector: determine if row or column vector
    %---------------------------------------------------------------------------

    function s = isvector (G)
    %ISVECTOR determine if the GraphBLAS matrix is a row or column vector.
    % isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.
    %
    % See also issparse, ismatrix, isscalar, sparse, full, isa, gb.
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) || (n == 1) ;
    end

    %---------------------------------------------------------------------------
    % isscalar: determine if scalar
    %---------------------------------------------------------------------------

    function s = isscalar (G)
    %ISSCALAR determine if the GraphBLAS matrix is a scalar.
    % isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
    %
    % See also issparse, ismatrix, isvector, sparse, full, isa, gb.
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) && (n == 1) ;
    end

    %---------------------------------------------------------------------------
    % isnumeric: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = isnumeric (G)
    %ISNUMERIC always true for any GraphBLAS matrix.
    % isnumeric (G) is always true for any GraphBLAS matrix G, including
    % logical matrices, since those matrices can be operated on in any
    % semiring, just like any other GraphBLAS matrix.
    %
    % See also isfloat, isreal, isinteger, islogical, gb.type, isa, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isfloat: determine if a GraphBLAS matrix has a floating-point type
    %---------------------------------------------------------------------------

    function s = isfloat (G)
    %ISFLOAT true for floating-point GraphBLAS matrices.
    %
    % See also isnumeric, isreal, isinteger, islogical, gb.type, isa, gb.
    t = gbtype (G.opaque) ;
    s = isequal (t, 'double') || isequal (t, 'single') || ...
        isequal (t, 'complex') ;
    end

    %---------------------------------------------------------------------------
    % isreal: determine if a GraphBLAS matrix is real (not complex)
    %---------------------------------------------------------------------------

    function s = isreal (G)
    %ISREAL true for real GraphBLAS matrices.
    %
    % See also isnumeric, isfloat, isinteger, islogical, gb.type, isa, gb.
    s = ~isequal (gbtype (G.opaque), 'complex') ;
    end

    %---------------------------------------------------------------------------
    % isinteger: determine if a GraphBLAS matrix has an integer type
    %---------------------------------------------------------------------------

    function s = isinteger (G)
    %ISINTEGER true for integer GraphBLAS matrices.
    %
    % See also isnumeric, isfloat, isreal, islogical, gb.type, isa, gb.
    t = gbtype (G.opaque) ;
    s = isequal (t (1:3), 'int') || isequal (t, (1:4), 'uint') ;
    end

    %---------------------------------------------------------------------------
    % islogical: determine if a GraphBLAS matrix has a logical type
    %---------------------------------------------------------------------------

    function s = islogical (G)
    %ISINTEGER true for logical GraphBLAS matrices.
    %
    % See also isnumeric, isfloat, isreal, isinteger, gb.type, isa, gb.
    t = gbtype (G.opaque) ;
    s = isequal (t, 'logical') ;
    end

    %---------------------------------------------------------------------------
    % isa: determine if a GraphBLAS matrix is of a particular class
    %---------------------------------------------------------------------------

    function s = isa (G, classname)
    %ISA Determine if a GraphBLAS matrix is of specific class.
    %
    % For any GraphBLAS matrix G, isa (G, 'gb'), isa (G, 'numeric'), and isa
    % (G, 'object') are always true.
    %
    % isa (G, 'float') is the same as isfloat (G), and is true if the gb matrix
    % G has type 'double', 'single', or 'complex'.
    %
    % isa (G, 'integer') is the same as isinteger (G), and is true if the gb
    % matrix G has type 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    % 'uint32', or 'uint64'.
    %
    % isa (G, classname) is true if the classname matches the type of G.
    %
    % See also gb.type, isnumeric, islogical, ischar, iscell, isstruct,
    % isfloat, isinteger, isobject, isjava, issparse, isreal, class.
    if (isequal (classname, 'gb') | isequal (classname, 'numeric'))
        % all GraphBLAS matrices are numeric, and have class name 'gb'
        s = true ;
    elseif (isequal (classname, 'float'))
        % GraphBLAS double, single, and complex matrices are 'float'
        s = isfloat (G) ;
    elseif (isequal (classname, 'integer'))
        % GraphBLAS int* and uint* matrices are 'integer'
        s = isinteger (G) ;
    elseif (isequal (gb.type (G), classname))
        % specific cases, such as isa (G, 'double')
        s = true ;
    else
        % catch-all for cases not handled above
        s = builtin ('isa', G, classname) ;
    end
    end

    %---------------------------------------------------------------------------
    % diag: diagonal matrices and diagonals of a matrix
    %---------------------------------------------------------------------------

    function C = diag (G,k)
    % DIAG Diagonal matrices and diagonals of a matrix.
    % C = diag (v,k) when v is a GraphBLAS vector with n components is a square
    % sparse GarphBLAS matrix of dimension n+abs(k), with the elements of v on
    % the kth diagonal. The main diagonal is k = 0; k > 0 is above the
    % diagonal, and k < 0 is below the main diagonal.  C = diag (v) is the
    % same as C = diag (v,0).
    %
    % c = diag (G,k) when G is a GraphBLAS matrix returns a GraphBLAS column
    % vector c formed the entries on the kth diagonal of G.  The main diagonal
    % is c = diag(G).
    %
    % The GraphBLAS diag function always constructs a GraphBLAS sparse matrix,
    % unlike the the MATLAB diag, which always constructs a MATLAB full matrix.
    %
    % Examples:
    %
    %   C1 = diag (gb (1:10, 'uint8'), 2)
    %   C2 = sparse (diag (1:10, 2))
    %   nothing = double (C1-C2)
    %
    %   A = magic (8)
    %   full (double ([diag(A,1) diag(gb(A),1)]))
    %
    %   m = 5 ;
    %   f = ones (2*m,1) ;
    %   A = diag(-m:m) + diag(f,1) + diag(f,-1)
    %   G = diag(gb(-m:m)) + diag(gb(f),1) + diag(gb(f),-1)
    %   nothing = double (A-G)
    %
    % See also diag, spdiags, tril, triu, gb.select.

    if (nargin < 2)
        k = 0 ;
    end
    [am, an] = size (G) ;
    isvec = (am == 1) || (an == 1) ;

    if (isvec)
        % C = diag (v,k) is an m-by-m matrix
        if (am == 1)
            % convert G to a column vector
            v = G.' ;
        else
            v = G ;
        end
        n = length (v) ;
        m = n + abs (k) ;
        if (k >= 0)
            [I, ~, X] = gb.extracttuples (v, struct ('kind', 'zero-based')) ;
            J = I + int64 (k) ;
        else
            [J, ~, X] = gb.extracttuples (v, struct ('kind', 'zero-based')) ;
            I = J - int64 (k) ;
        end
        C = gb.build (I, J, X, m, m) ;
    else
        % C = diag (G,k) is a column vector formed from the elements of the kth
        % diagonal of G
        C = gb.select ('diag', G, k) ;
        if (k >= 0)
            [I, ~, X] = gb.extracttuples (C, struct ('kind', 'zero-based')) ;
            m = min (an-k, am) ;
        else
            [~, I, X] = gb.extracttuples (C, struct ('kind', 'zero-based')) ;
            m = min (an, am+k) ;
        end
        J = zeros (length (X), 1, 'int64') ;
        C = gb.build (I, J, X, m, 1) ;
    end
    end

    %---------------------------------------------------------------------------
    % tril: lower triangular part
    %---------------------------------------------------------------------------

    function L = tril (G, k)
    %TRIL lower triangular part of a GraphBLAS matrix.
    % L = tril (G) returns the lower triangular part of G. L = tril (G,k)
    % returns the entries on and below the kth diagonal of G, where k=0 is
    % the main diagonal.
    %
    % See also triu.
    if (nargin < 2)
        k = 0 ;
    end
    L = gb.select ('tril', G.opaque, k) ;
    end

    %---------------------------------------------------------------------------
    % triu: upper triangular part
    %---------------------------------------------------------------------------

    function U = triu (G, k)
    %TRIU upper triangular part of a GraphBLAS matrix.
    % U = triu (G) returns the upper triangular part of G. U = triu (G,k)
    % returns the entries on and above the kth diagonal of X, where k=0 is
    % the main diagonal.
    %
    % See also tril.
    if (nargin < 2)
        k = 0 ;
    end
    U = gb.select ('triu', G.opaque, k) ;
    end

    %---------------------------------------------------------------------------
    % kron: Kronecker product
    %---------------------------------------------------------------------------

    function C = kron (A, B)
    %KRON sparse Kronecker product
    % C = kron (A,B) is the sparse Kronecker tensor product of A and B.
    C = gb.gbkron ('*', A, B) ;
    end

    %---------------------------------------------------------------------------
    % repmat: replicate and tile a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function C = repmat (G, m, n)
    %REPMAT Replicate and tile a GraphBLAS matrix.
    % C = repmat (G, m, n)  % constructs an m-by-n tiling of the gb matrix A
    % C = repmat (G, [m n]) % same as C = repmat (A, m, n)
    % C = repmat (G, n)     % constructs an n-by-n tiling of the gb matrix G
    %
    % See also kron, gb.gbkron.
    if (nargin == 3)
        R = ones (m, n, 'logical') ;
    else
        R = ones (m, 'logical') ;
    end
    C = gb.gbkron (['2nd.' gb.type(G)], R, G) ;
    end

    %---------------------------------------------------------------------------
    % reshape: reshape a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function C = reshape (G, arg1, arg2)
    %RESHAPE Reshape a GraphBLAS matrix.
    % C = reshape (G, m, n) or C = reshape (G, [m n]) returns the m-by-n
    % matrix whose elements are taken columnwise from G.  The matrix G must
    % have numel (G) == m*n.  That is numel (G) == numel (C) must be true.
    [mold nold] = size (G) ;
    mold = int64 (mold) ;
    nold = int64 (nold) ;
    if (nargin == 2)
        if (length (arg1) ~= 2)
            error ('reshape (G,s): s must have exactly two elements') ;
        end
        mnew = int64 (arg1 (1)) ;
        nnew = int64 (arg1 (2)) ;
    elseif (nargin == 3)
        if (~isscalar (arg1) | ~isscalar (arg2))
            error ('reshape (G,m,n): m and n must be scalars') ;
        end
        mnew = int64 (arg1) ;
        nnew = int64 (arg2) ;
    end
    if (mold * nold ~= mnew * nnew)
        error ('number of elements must not change') ;
    end
    if (isempty (G))
        C = gb (mnew, nnew, gb.type (G)) ;
    else
        [iold jold x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        % convert i and j from 2D (mold-by-nold) to 1D indices
        k = convert_index_2d_to_1d (iold, jold, mold) ;
        % convert k from 1D indices to 2D (mnew-by-nnew)
        [inew jnew] = convert_index_1d_to_2d (k, mnew) ;
        % rebuild the new matrix
        C = gb.build (inew, jnew, x, mnew, nnew) ;
    end
    end

    function k = convert_index_2d_to_1d (i, j, m)
    % the indices must be zero-based
    k = i + j * m ;
    end

    function [i j] = convert_index_1d_to_2d (k, m) ;
    % the indices must be zero-based
    i = rem (k, m) ;
    j = (k - i) / m ;
    end

    %---------------------------------------------------------------------------
    % abs: absolute value
    %---------------------------------------------------------------------------

    function C = abs (G)
    %ABS Absolute value.
    C = gb.apply ('abs', G) ;
    end

    %---------------------------------------------------------------------------
    % sign: signum function
    %---------------------------------------------------------------------------

    function C = sign (G)
    %SIGN Signum function.
    % For each element of a GraphBLAS matrix G, sign(G) returns 1 if the
    % element is greater than zero, 0 if it equals zero, and -1 if it is less
    % than zero.  The output C is a sparse GraphBLAS matrix, with no explicit
    % zeros; any entry not present is implicitly zero.
    C = spones (gb.select ('>0', G)) - spones (gb.select ('<0', G)) ;
    end

    %---------------------------------------------------------------------------
    % istril: check if lower triangular
    %---------------------------------------------------------------------------

    function result = istril (G)
    %ISTRIL  Determine if a matrix is lower triangular.
    % A GraphBLAS matrix G may have explicit zeros.  If these appear in the
    % upper triangular part of G, then istril (G) is false, but
    % istril (double (G)) can be true since double (G) drops those entries.

    % FUTURE: this will be much faster when written as a mexFunction.
    result = (gb.nvals (triu (G, 1)) == 0) ;
    end

    %---------------------------------------------------------------------------
    % istriu: check if upper triangular
    %---------------------------------------------------------------------------

    function result = istriu (G)
    %ISTRIU  Determine if a matrix is upper triangular.
    % A GraphBLAS matrix G may have explicit zeros.  If these appear in the
    % lower triangular part of G, then istriu (G) is false, but
    % istriu (double (G)) can be true since the double (G) drops those entries.

    % FUTURE: this will be much faster when written as a mexFunction.
    result = (gb.nvals (tril (G, -1)) == 0) ;
    end

    %---------------------------------------------------------------------------
    % isbanded: check if banded
    %---------------------------------------------------------------------------

    function result = isbanded (G, lo, hi)
    %ISBANDED True if G is a banded matrix.
    % isbanded (G, lo, hi) is true if the bandwidth of G is between lo and hi.

    % FUTURE: this will be much faster when 'bandwidth' is a mexFunction.
    [Glo, Ghi] = bandwidth (G) ;
    result = (Glo <= lo) & (Ghi <= hi) ;
    end

    %---------------------------------------------------------------------------
    % isdiag: check if diagonal
    %---------------------------------------------------------------------------

    function result = isdiag (G)
    %ISDIAG True if G is a diagonal matrix.

    % FUTURE: this will be much faster when 'bandwidth' is a mexFunction.
    result = isbanded (G, 0, 0) ;
    end

    %---------------------------------------------------------------------------
    % ishermitian: check if Hermitian
    %---------------------------------------------------------------------------

    function result = ishermitian (G, option)
    %ISHERMITIAN Determine if a GraphBLAS matrix is real symmetric or
    % complex Hermitian.
    %
    % See also issymetric.

    % FUTURE: this will be much faster.  See CHOLMOD/MATLAB/spsym.
    [m n] = size (G) ;
    if (m ~= n)
        result = false ;
    end
    if (nargin < 2)
        option = 'nonskew' ;
    end
    if (isequal (option, 'skew'))
        result = (norm (G + G', 1) == 0) ;
    else
        result = (norm (G - G', 1) == 0) ;
    end
    end

    %---------------------------------------------------------------------------
    % issymmetric: check if Hermitian
    %---------------------------------------------------------------------------

    function result = issymmetric (G, option)
    %ISHERMITIAN Determine if a GraphBLAS matrix is symmetric.
    %
    % See also ishermitian.

    % FUTURE: this will be much faster.  See CHOLMOD/MATLAB/spsym.
    [m n] = size (G) ;
    if (m ~= n)
        result = false ;
    end
    if (nargin < 2)
        option = 'nonskew' ;
    end
    if (isequal (option, 'skew'))
        result = (norm (G + G.', 1) == 0) ;
    else
        result = (norm (G - G.', 1) == 0) ;
    end
    end

    %---------------------------------------------------------------------------
    % bandwidth: determine the lower & upper bandwidth
    %---------------------------------------------------------------------------

    function [arg1,arg2] = bandwidth (G,uplo)
    %BANDWIDTH Determine the bandwidth of a GraphBLAS matrix.
    % [lo, hi] = bandwidth (G) returns the upper and lower bandwidth of G.
    % lo = bandwidth (G, 'lower') returns just the lower bandwidth.
    % hi = bandwidth (G, 'upper') returns just the upper bandwidth.
    %
    % See also isbanded, isdiag, istril, istriu.

    % FUTURE: this will be much faster when implemented in a mexFunction.
    if (gb.nvals (G) == 0)
        % matrix is empty
        hi = 0 ;
        lo = 0 ;
    else
        [i j] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        b = j - i ;
        hi = max (0,  double (max (b))) ;
        lo = max (0, -double (min (b))) ;
    end
    if (nargin == 1)
       arg1 = lo ;
       arg2 = hi ;
    else
        if (nargout > 1)
            error ('too many output arguments') ;
        elseif isequal (uplo, 'lower')
            arg1 = lo ;
        elseif isequal (uplo, 'upper')
            arg1 = hi ;
        else
            error ('unrecognized input parameter') ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % sqrt: element-wise square root
    %---------------------------------------------------------------------------

    function C = sqrt (G)
    %SQRT Square root.
    % SQRT (G) is the square root of the elements of the GraphBLAS matrix G.
    % Complex matrices are not yet supported.
    C = G.^(.5) ;
    end

    %---------------------------------------------------------------------------
    % sum: reduce a matrix to a vector or scalar, via the '+' operator
    %---------------------------------------------------------------------------

    function C = sum (G, option)
    %SUM Sum of elements.
    % C = sum (G), where G is an m-by-n GraphBLAS matrix, computes a 1-by-n row
    % vector C where C(j) is the sum of all entries in G(:,j).  If G is a row
    % or column vector, then sum (G) is a scalar sum of all the entries in the
    % vector.
    %
    % C = sum (G,'all') sums all elements of G to a single scalar.
    %
    % C = sum (G,1) is the default when G is a matrix, which is to sum each
    % column to a scalar, giving a 1-by-n row vector.  If G is already a row
    % vector, then C = G.
    %
    % C = sum (G,2) sums each row to a scalar, resulting in an m-by-1 column
    % vector C where C(i) is the sum of all entries in G(i,:).
    %
    % The MATLAB sum (A, ... type, nanflag) allows for different types of sums
    % to be computed, and the NaN behavior can be specified.  The GraphBLAS
    % sum (G,...) uses only a type of 'native', and a nanflag of 'includenan'.
    % See 'help sum' for more details.

    if (isequal (gb.type (G), 'logical'))
        op = '+.int64' ;
    else
        op = '+' ;
    end
    if (nargin == 1)
        % C = sum (G); check if G is a row vector
        if (isvector (G))
            % C = sum (G) for a vector G results in a scalar C
            C = gb.reduce (op, G) ;
        else
            % C = sum (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce (op, G, struct ('in0', 'transpose'))' ;
        end
    elseif (isequal (option, 'all'))
        % C = sum (G, 'all'), reducing all entries to a scalar
        C = gb.reduce (op, G) ;
    elseif (isequal (option, 1))
        % C = sum (G,1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb.vreduce (op, G, struct ('in0', 'transpose'))' ;
    elseif (isequal (option, 2))
        % C = sum (G,2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb.vreduce (op, G) ;
    else
        error ('unknown option') ;
    end
    end

    %---------------------------------------------------------------------------
    % prod: reduce a matrix to a vector or scalar, via the '*' operator
    %---------------------------------------------------------------------------

    function C = prod (G, option)
    %PROD Product of elements.
    % C = prod (G), where G is an m-by-n GraphBLAS matrix, computes a 1-by-n
    % row vector C where C(j) is the product of all entries in G(:,j).  If G is
    % a row or column vector, then prod (G) is a scalar product of all the
    % entries in the vector.
    %
    % C = prod (G,'all') takes the product of all elements of G to a single
    % scalar.
    %
    % C = prod (G,1) is the default when G is a matrix, which is to take the
    % product of each column, giving a 1-by-n row vector.  If G is already a
    % row vector, then C = G.
    %
    % C = prod (G,2) takes the product of each row, resulting in an m-by-1
    % column vector C where C(i) is the product of all entries in G(i,:).
    %
    % The MATLAB prod (A, ... type, nanflag) allows for different types of
    % products to be computed, and the NaN behavior can be specified.  The
    % GraphBLAS prod (G,...) uses only a type of 'native', and a nanflag of
    % 'includenan'.  See 'help prod' for more details.

    [m n] = size (G) ;
    d = struct ('in0', 'transpose') ;
    if (isequal (gb.type (G), 'logical'))
        op = '&.logical' ;
    else
        op = '*' ;
    end
    if (nargin == 1)
        % C = prod (G); check if G is a row vector
        if (isvector (G))
            % C = prod (G) for a vector G results in a scalar C
            if (gb.nvals (G) < m*n)
                C = gb (0, gb.type (G)) ;
            else
                C = gb.reduce (op, G) ;
            end
        else
            % C = prod (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = (gb.vreduce (op, G, d) .* (col_degree (G) == m))' ;
        end
    elseif (isequal (option, 'all'))
        % C = prod (G, 'all'), reducing all entries to a scalar
        if (gb.nvals (G) < m*n)
            C = gb (0, gb.type (G)) ;
        else
            C = gb.reduce (op, G) ;
        end
    elseif (isequal (option, 1))
        % C = prod (G,1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = (gb.vreduce (op, G, d) .* (col_degree (G) == m))' ;
    elseif (isequal (option, 2))
        % C = prod (G,2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb.vreduce (op, G) .* (row_degree (G) == n) ;
    else
        error ('unknown option') ;
    end
    end

    %---------------------------------------------------------------------------
    % norm: compute the 1-norm or inf-norm
    %---------------------------------------------------------------------------

    function s = norm (G,kind)
    %NORM norm of a GraphBLAS sparse matrix.
    %
    % If G is a matrix:
    %
    %   norm (G,1) is the maximum sum of the columns of abs (G).
    %   norm (G,inf) is the maximum sum of the rows of abs (G).
    %   norm (G,'fro') is the Frobenius norm of G: the sqrt of the sum of the
    %       squares of the entries in G.
    %   The 2-norm is not available for either MATLAB or GraphBLAS sparse
    %       matrices.
    %
    % If G is a row or column vector:
    %
    %   norm (G,1) is the sum of abs (G)
    %   norm (G,2) is the sqrt of the sum of G.^2
    %   norm (G,inf) is the maximum of abs (G)
    %   norm (G,-inf) is the minimum of abs (G)
    %
    % See also gb.reduce.

    if (nargin == 1)
        kind = 2 ;
    end
    if (kind == 0)
        error ('unknown norm') ;
    end
    if (ischar (kind))
        if (isequal (kind, 'fro'))
            kind = 0 ;
        else
            error ('unknown norm') ;
        end
    end

    if (isvector (G))
        if (kind == 1)
            s = sum (abs (G)) ;
        elseif (kind == 2 | kind == 0)
            s = sqrt (sum (G.^2)) ;
        elseif (kind == inf)
            s = max (abs (G)) ;
        elseif (kind == -inf)
            s = min (abs (G)) ;
        else
            error ('unknown norm') ;
        end
    else
        if (kind == 1)
            s = max (sum (abs (G))) ;
        elseif (kind == 2)
            error ('Sparse norm (G,2) is not available.') ;
        elseif (kind == 0)
            s = sqrt (sum (G.^2, 'all')) ;
        elseif (kind == inf)
            s = max (sum (abs (G), 2)) ;
        elseif (kind == -inf)
            error ('Sparse norm(G,-inf) is not available.') ;
        else
            error ('unknown norm') ;
        end
    end
    s = full (double (s)) ;
    end

    %---------------------------------------------------------------------------
    % max: reduce a matrix to a vector or scalar, via the 'max' operator
    %---------------------------------------------------------------------------

    function C = max (varargin)
    %MAX Maximum elements of a GraphBLAS or MATLAB matrix.
    %
    % C = max (G) is the largest entry in the vector G.  If G is a matrix,
    % C is a row vector with C(j) = max (G (:,j)).
    %
    % C = max (A,B) is an array of the element-wise maximum of two matrices
    % A and B, which either have the same size, or one can be a scalar.
    % Either A and/or B can be GraphBLAS or MATLAB matrices.
    %
    % C = max (G, [ ], 'all') is a scalar, with the largest entry in G.
    % C = max (G, [ ], 1) is a row vector with C(j) = max (G (:,j))
    % C = max (G, [ ], 2) is a column vector with C(i) = max (G (i,:))
    %
    % The indices of the maximum entry, or [C,I] = max (...) in the MATLAB
    % built-in max function, are not computed.  The max (..., nanflag) option
    % is not available; only the 'includenan' behavior is supported.

    G = varargin {1} ;
    [m n] = size (G) ;
    if (isequal (gb.type (G), 'logical'))
        op = '|.logical' ;
    else
        op = 'max' ;
    end

    if (nargin == 1)

        % C = max (G)
        if (isvector (G))
            % C = max (G) for a vector G results in a scalar C
            C = gb.reduce (op, G) ;
            if (gb.nvals (G) < m*n) ;
                C = max (C, 0) ;
            end
        else
            % C = max (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
            % if C(j) < 0, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, (C < 0) & (col_degree (G) < m), 0)' ;
        end

    elseif (nargin == 2)

        % C = max (A,B)
        A = varargin {1} ;
        B = varargin {2} ;
        if (isscalar (A))
            if (isscalar (B))
                % both A and B are scalars.  Result is also a scalar.
                C = sparse_comparator (op, A, B) ;
            else
                % A is a scalar, B is a matrix
                if (get_scalar (A) > 0)
                    % since A > 0, the result is full
                    C = gb.eadd (op, gb.expand (A, true (size (B))), B) ;
                else
                    % since A <= 0, the result is sparse.  Expand the scalar A
                    % to the pattern of B.
                    C = gb.eadd (op, gb.expand (A, B), B) ;
                end
            end
        else
            if (isscalar (B))
                % A is a matrix, B is a scalar
                if (get_scalar (B) > 0)
                    % since B > 0, the result is full
                    C = gb.eadd (op, A, gb.expand (B, true (size (A)))) ;
                else
                    % since B <= 0, the result is sparse.  Expand the scalar B
                    % to the pattern of A.
                    C = gb.eadd (op, A, gb.expand (B, A)) ;
                end
            else
                % both A and B are matrices.  Result is sparse.
                C = sparse_comparator (op, A, B) ;
            end
        end

    elseif (nargin == 3)

        % C = max (G, [ ], option)
        option = varargin {3} ;
        if (isequal (option, 'all'))
            % C = max (G, [ ] 'all'), reducing all entries to a scalar
            C = gb.reduce (op, G) ;
            if (gb.nvals (G) < m*n) ;
                C = max (C, 0) ;
            end
        elseif (isequal (option, 1))
            % C = max (G, [ ], 1) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
            % if C(j) < 0, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, (C < 0) & (col_degree (G) < m), 0)' ;
        elseif (isequal (option, 2))
            % C = max (G, [ ], 2) reduces each row to a scalar,
            % giving an m-by-1 column vector.
            C = gb.vreduce (op, G) ;
            % if C(i) < 0, but the row is sparse, then assign C(i) = 0.
            C = gb.subassign (C, (C < 0) & (row_degree (G) < n), 0) ;
        else
            error ('unknown option') ;
        end

    else
        error ('invalid usage') ;
    end
    end

    %---------------------------------------------------------------------------
    % min: reduce a matrix to a vector or scalar, via the 'min' operator
    %---------------------------------------------------------------------------

    function C = min (varargin)
    %MIN Minimum elements of a GraphBLAS or MATLAB matrix.
    %
    % C = min (G) is the smallest entry in the vector G.  If G is a matrix,
    % C is a row vector with C(j) = min (G (:,j)).
    %
    % C = min (A,B) is an array of the element-wise minimum of two matrices
    % A and B, which either have the same size, or one can be a scalar.
    % Either A and/or B can be GraphBLAS or MATLAB matrices.
    %
    % C = min (G, [ ], 'all') is a scalar, with the smallest entry in G.
    % C = min (G, [ ], 1) is a row vector with C(j) = min (G (:,j))
    % C = min (G, [ ], 2) is a column vector with C(i) = min (G (i,:))
    %
    % The indices of the minimum entry, or [C,I] = min (...) in the MATLAB
    % built-in min function, are not computed.  The min (..., nanflag) option
    % is not available; only the 'includenan' behavior is supported.

    G = varargin {1} ;
    [m n] = size (G) ;

    if (isequal (gb.type (G), 'logical'))
        op = '&.logical' ;
    else
        op = 'min' ;
    end

    if (nargin == 1)

        % C = min (G)
        if (isvector (G))
            % C = min (G) for a vector G results in a scalar C
            C = gb.reduce (op, G) ;
            if (gb.nvals (G) < m*n) ;
                C = min (C, 0) ;
            end
        else
            % C = min (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
            % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, (C > 0) & (col_degree (G) < m), 0)' ;
        end

    elseif (nargin == 2)

        % C = min (A,B)
        A = varargin {1} ;
        B = varargin {2} ;
        if (isscalar (A))
            if (isscalar (B))
                % both A and B are scalars.  Result is also a scalar.
                C = sparse_comparator (op, A, B) ;
            else
                % A is a scalar, B is a matrix
                if (get_scalar (A) < 0)
                    % since A < 0, the result is full
                    C = gb.eadd (op, gb.expand (A, true (size (B))), B) ;
                else
                    % since A >= 0, the result is sparse.  Expand the scalar A
                    % to the pattern of B.
                    C = gb.eadd (op, gb.expand (A, B), B) ;
                end
            end
        else
            if (isscalar (B))
                % A is a matrix, B is a scalar
                if (get_scalar (B) < 0)
                    % since B < 0, the result is full
                    C = gb.eadd (op, A, gb.expand (B, true (size (A)))) ;
                else
                    % since B >= 0, the result is sparse.  Expand the scalar B
                    % to the pattern of A.
                    C = gb.eadd (op, A, gb.expand (B, A)) ;
                end
            else
                % both A and B are matrices.  Result is sparse.
                C = sparse_comparator (op, A, B) ;
            end
        end

    elseif (nargin == 3)

        % C = min (G, [ ], option)
        option = varargin {3} ;
        if (isequal (option, 'all'))
            % C = min (G, [ ] 'all'), reducing all entries to a scalar
            C = gb.reduce (op, G) ;
            if (gb.nvals (G) < m*n) ;
                C = min (C, 0) ;
            end
        elseif (isequal (option, 1))
            % C = min (G, [ ], 1) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
            % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, (C > 0) & (col_degree (G) < m), 0)' ;
        elseif (isequal (option, 2))
            % C = min (G, [ ], 2) reduces each row to a scalar,
            % giving an m-by-1 column vector.
            C = gb.vreduce (op, G) ;
            % if C(i) > 0, but the row is sparse, then assign C(i) = 0.
            C = gb.subassign (C, (C > 0) & (row_degree (G) < n), 0) ;
        else
            error ('unknown option') ;
        end

    else
        error ('invalid usage') ;
    end
    end

    %---------------------------------------------------------------------------
    % any: reduce a matrix to a vector or scalar, via the '|' operator
    %---------------------------------------------------------------------------

    function C = any (G, option)
    %ANY True if any element is nonzero or true.
    %
    % C = any (G) is true if any entry in G is nonzero or true.  If G is a
    % matrix, C is a row vector with C(j) = any (G (:,j)).
    %
    % C = any (G, 'all') is a scalar, true if any entry in G is nonzero or true.
    % C = any (G, 1) is a row vector with C(j) = any (G (:,j))
    % C = any (G, 2) is a column vector with C(i) = any (G (i,:))
    %
    % See also all, nnz, gb.nvals.

    [m n] = size (G) ;

    if (nargin == 1)

        % C = any (G)
        if (isvector (G))
            % C = any (G) for a vector G results in a scalar C
            C = gb.reduce ('|.logical', G) ;
        else
            % C = any (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce ('|.logical', G, struct ('in0', 'transpose'))' ;
        end

    elseif (nargin == 2)

        % C = any (G, option)
        if (isequal (option, 'all'))
            % C = any (G, 'all'), reducing all entries to a scalar
            C = gb.reduce ('|.logical', G) ;
        elseif (isequal (option, 1))
            % C = any (G, 1) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce ('|.logical', G, struct ('in0', 'transpose'))' ;
        elseif (isequal (option, 2))
            % C = any (G, 2) reduces each row to a scalar,
            % giving an m-by-1 column vector.
            C = gb.vreduce ('|.logical', G) ;
        else
            error ('unknown option') ;
        end

    else
        error ('invalid usage') ;
    end
    end

    %---------------------------------------------------------------------------
    % all: reduce a matrix to a vector or scalar, via the '&' operator
    %---------------------------------------------------------------------------

    function C = all (G, option)
    %ALL True if all elements are nonzero or true.
    %
    % C = all (G) is true if all entries G are nonzero or true.  If G is a
    % matrix, C is a row vector with C(j) = all (G (:,j)).
    %
    % C = all (G, 'all') is a scalar, true if all entries G are nonzero or true.
    % C = all (G, 1) is a row vector with C(j) = all (G (:,j))
    % C = all (G, 2) is a column vector with C(i) = all (G (i,:))
    %
    % See also any, nnz, gb.nvals.

    [m n] = size (G) ;
    nvals = gb.nvals (G) ;

    if (nargin == 1)

        % C = all (G)
        if (isvector (G))
            % C = all (G) for a vector G results in a scalar C
            if (nvals < m*n) ;
                C = gb (false, 'logical') ;
            else
                C = gb.reduce ('&.logical', G) ;
            end
        else
            % C = all (G) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce ('&.logical', G, struct ('in0', 'transpose')) ;
            % if C(j) is true, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, C & (col_degree (G) < m), 0)' ;
        end

    elseif (nargin == 2)

        % C = all (G, option)
        if (isequal (option, 'all'))
            % C = all (G, 'all'), reducing all entries to a scalar
            if (nvals < m*n) ;
                C = gb (false, 'logical') ;
            else
                C = gb.reduce ('&.logical', G) ;
            end
        elseif (isequal (option, 1))
            % C = all (G, 1) reduces each column to a scalar,
            % giving a 1-by-n row vector.
            C = gb.vreduce ('&.logical', G, struct ('in0', 'transpose')) ;
            % if C(j) is true, but the column is sparse, then assign C(j) = 0.
            C = gb.subassign (C, C & (col_degree (G) < m), 0)' ;
        elseif (isequal (option, 2))
            % C = all (G, 2) reduces each row to a scalar,
            % giving an m-by-1 column vector.
            C = gb.vreduce ('&.logical', G) ;
            % if C(i) is true, but the row is sparse, then assign C(i) = 0.
            C = gb.subassign (C, C & (row_degree (G) < n), 0) ;
        else
            error ('unknown option') ;
        end

    else
        error ('invalid usage') ;
    end
    end

    %---------------------------------------------------------------------------
    % eps: spacing of floating-point numbers
    %---------------------------------------------------------------------------

    function C = eps (G)
    %EPS Spacing of floating-point numbers

    % FUTURE: this could be much faster as a mexFunction.
    if (~isfloat (G))
        error ('Type must be ''single'', ''double'', or ''complex''') ;
    end
    [m n] = size (G) ;
    [i j x] = gb.extracttuples (full (G), struct ('kind', 'zero-based')) ;
    C = gb.build (i, j, eps (x), m, n) ;
    end

    %---------------------------------------------------------------------------
    % ceil: element-wise ceiling operator
    %---------------------------------------------------------------------------

    function C = ceil (G)
    %CEIL round entries to the nearest integers towards infinity
    % See also floor, round, fix.

    % FUTURE: this could be much faster as a mexFunction.
    if (isfloat (G) && gb.nvals (G) > 0)
        [m n] = size (G) ;
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, ceil (x), m, n) ;
    else
        C = G ;
    end
    end

    %---------------------------------------------------------------------------
    % floor: element-wise floor operator
    %---------------------------------------------------------------------------

    function C = floor (G)
    %FLOOR round entries to the nearest integers towards -infinity
    % See also ceil, round, fix.

    % FUTURE: this could be much faster as a mexFunction.
    if (isfloat (G) && gb.nvals (G) > 0)
        [m n] = size (G) ;
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, floor (x), m, n) ;
    else
        C = G ;
    end
    end

    %---------------------------------------------------------------------------
    % round: element-wise round operator
    %---------------------------------------------------------------------------

    function C = round (G)
    %ROUND round entries to the nearest integers
    % See also ceil, floor, fix.

    % FUTURE: this could be much faster as a mexFunction.
    if (isfloat (G) && gb.nvals (G) > 0)
        [m n] = size (G) ;
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, round (x), m, n) ;
    else
        C = G ;
    end
    end

    %---------------------------------------------------------------------------
    % fix: element-wise fix operator
    %---------------------------------------------------------------------------

    function C = fix (G)
    %FIX Round towards zero.
    % See also ceil, floor, round.

    % FUTURE: this could be much faster as a mexFunction.
    if (isfloat (G) && gb.nvals (G) > 0)
        [m n] = size (G) ;
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, fix (x), m, n) ;
    else
        C = G ;
    end
    end

    %---------------------------------------------------------------------------
    % isfinite: element-wise isfinite operator
    %---------------------------------------------------------------------------

    function C = isfinite (G)
    %ISFINITE True for finite elements.
    % See also isnan, isinf.

    % FUTURE: this could be much faster as a mexFunction.
    [m n] = size (G) ;
    if (isfloat (G) && m > 0 && n > 0)
        [i j x] = gb.extracttuples (full (G), struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, isfinite (x), m, n) ;
    else
        % C is all true
        C = gb (true (m, n)) ;
    end
    end

    %---------------------------------------------------------------------------
    % isinf: element-wise isinf operator
    %---------------------------------------------------------------------------

    function C = isinf (G)
    %ISINF True for infinite elements.
    % See also isnan, isfinite.

    % FUTURE: this could be much faster as a mexFunction.
    [m n] = size (G) ;
    if (isfloat (G) && gb.nvals (G) > 0)
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, isinf (x), m, n) ;
    else
        % C is all false
        C = gb (m, n, 'logical') ;
    end
    end

    %---------------------------------------------------------------------------
    % isnan: element-wise isnan operator
    %---------------------------------------------------------------------------

    function C = isnan (G)
    %ISNAN True for NaN elements.
    % See also isinf, isfinite.

    % FUTURE: this could be much faster as a mexFunction.
    [m n] = size (G) ;
    if (isfloat (G) && gb.nvals (G) > 0)
        [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
        C = gb.build (i, j, isnan (x), m, n) ;
    else
        % C is all false
        C = gb (m, n, 'logical') ;
    end
    end

    %---------------------------------------------------------------------------
    % spfun: apply a MATLAB function to the elmenents of a matrix
    %---------------------------------------------------------------------------

    function C = spfun (fun, G)
    %SPFUN Apply function to the entries of a GraphBLAS matrix.
    % C = spfun (fun, G) evaluates the function fun on the entries of G.

    % FUTURE: this would be much faster as a mexFunction, but calling feval
    % from inside a mexFunction would not be trivial.
    [m n] = size (G) ;
    [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
    x = feval (fun, x) ;
    C = gb.build (i, j, x, m, n, '1st', gb.type (x)) ;
    end

    %---------------------------------------------------------------------------
    % amd: approximate minimum degree ordering
    %---------------------------------------------------------------------------

    function p = amd (G, varargin)
    %AMD approximate minimum degree ordering.
    % See 'help amd' for details.
    p = builtin ('amd', logical (G), varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % colamd: column approximate minimum degree ordering
    %---------------------------------------------------------------------------

    function [p, varargout] = colamd (G, varargin)
    %COLAMD column approximate minimum degree ordering.
    % See 'help colamd' for details.
    [p, varargout{1:nargout-1}] = colamd (double (G), varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % symamd: approximate minimum degree ordering
    %---------------------------------------------------------------------------

    function [p, varargout] = symamd (G, varargin)
    %SYMAMD approximate minimum degree ordering.
    % See 'help symamd' for details.
    [p, varargout{1:nargout-1}] = symamd (double (G), varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % symrcm: reverse Cuthill-McKee ordering
    %---------------------------------------------------------------------------

    function p = symrcm (G)
    %SYMRCM Reverse Cuthill-McKee ordering.
    % See 'help symrcm' for details.
    p = builtin ('symrcm', logical (G)) ;
    end

    %---------------------------------------------------------------------------
    % dmperm: Dulmage-Mendelsohn permutation
    %---------------------------------------------------------------------------

    function [p, varargout] = dmperm (G)
    %DMPERM Dulmage-Mendelsohn permutation.
    % See 'help dmperm' for details.
    [p, varargout{1:nargout-1}] = builtin ('dmperm', logical (G)) ;
    end

    %---------------------------------------------------------------------------
    % conj: complex conjugate
    %---------------------------------------------------------------------------

    function C = conj (G)
    %CONJ complex conjugate.
    % Since all GraphBLAS matrices are currently real, conj (G) is just G.
    % Complex support will be added in the future.
    C = G ;
    end

    %---------------------------------------------------------------------------
    % real: real part of a complex matrix
    %---------------------------------------------------------------------------

    function C = real (G)
    %REAL complex real part.
    % Since all GraphBLAS matrices are currently real, real (G) is just G.
    % Complex support will be added in the future.
    C = G ;
    end

    %---------------------------------------------------------------------------
    % eig: eigenvalues and eigenvectors
    %---------------------------------------------------------------------------

    function [V, varargout] = eig (G, varargin)
    %EIG Eigenvalues and eigenvectors of a GraphBLAS matrix.
    % See 'help eig' for details.
    if (isreal (G) & issymmetric (G))
        % G can be sparse if G is real and symmetric
        G = double (G) ;
    else
        % otherwise, G must be full.
        G = full (double (G)) ;
    end
    if (nargin == 1)
        [V, varargout{1:nargout-1}] = builtin ('eig', G) ;
    else
        args = varargin ;
        for k = 1:length (args)
            if (isa (args {k}, 'gb'))
                args {k} = full (double (args {k})) ;
            end
        end
        [V, varargout{1:nargout-1}] = builtin ('eig', G, args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % FUTURE: these could also be overloaded (not static) methods:
    %---------------------------------------------------------------------------

    % spdiags, blkdiag, bsxfun, cummin, cummax, cumprod, diff, inv, issorted,
    % issortedrows, reshape, sort, rem, mod, lu, chol, qr, ...  See 'methods
    % double' for more options.

%===============================================================================
% operator overloading =========================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % plus: C = A + B
    %---------------------------------------------------------------------------

    function C = plus (A, B)
    %PLUS sparse matrix addition, C = A+B.
    % A and B can be GraphBLAS matrices or MATLAB sparse or full matrices, in
    % any combination.  If A and B are matrices, the pattern of C is the set
    % union of A and B.  If one of A or B is a scalar, the scalar is expanded
    % into a dense matrix the size of the other matrix, and the result is a
    % dense matrix.  If the type of A and B differ, the type of A is used, as:
    % C = A + gb (B, gb.type (A)).
    %
    % See also gb.eadd, minus, uminus.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars.  Result is also a scalar.
            C = gb.eadd ('+', A, B) ;
        else
            % A is a scalar, B is a matrix.  Result is full.
            C = gb.eadd ('+', gb.expand (A, true (size (B))), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar.  Result is full.
            C = gb.eadd ('+', A, gb.expand (B, true (size (A)))) ;
        else
            % both A and B are matrices.  Result is sparse.
            C = gb.eadd ('+', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % minus: C = A - B
    %---------------------------------------------------------------------------

    function C = minus (A, B)
    %MINUS sparse matrix subtraction, C = A-B.
    % A and B can be GraphBLAS matrices or MATLAB sparse or full matrices, in
    % any combination.  If A and B are matrices, the pattern of C is the set
    % union of A and B.  If one of A or B is a scalar, the scalar is expanded
    % into a dense matrix the size of the other matrix, and the result is a
    % dense matrix.  If the type of A and B differ, the type of A is used, as:
    % C = A + gb (B, gb.type (A)).
    %
    % See also gb.eadd, plus, uminus.
    C = A + (-B) ;
    end

    %---------------------------------------------------------------------------
    % uminus: C = -A
    %---------------------------------------------------------------------------

    function C = uminus (A)
    %UMINUS negate a GraphBLAS sparse matrix.
    C = gb.apply ('-', A) ;
    end

    %---------------------------------------------------------------------------
    % uplus: C = +A
    %---------------------------------------------------------------------------

    function C = uplus (A)
    %UPLUS C = +A
    C = A ;
    end

    %---------------------------------------------------------------------------
    % times: C = A .* B
    %---------------------------------------------------------------------------

    function C = times (A, B)
    %TIMES C = A.*B, sparse matrix element-wise multiplication
    % If both A and B are matrices, the pattern of C is the intersection of A
    % and B.  If one is a scalar, the pattern of C is the same as the pattern
    % of the one matrix.
    %
    % See also gb.emult.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.emult ('*', A, B) ;
        else
            % A is a scalar, B is a matrix
            C = gb.emult ('*', gb.expand (A, B), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            C = gb.emult ('*', A, gb.expand (B, A)) ;
        else
            % both A and B are matrices
            C = gb.emult ('*', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % mtimes: C = A * B
    %---------------------------------------------------------------------------

    function C = mtimes (A, B)
    %MTIMES sparse matrix-matrix multiplication over the standard semiring.
    % C=A*B multiples two matrices using the standard '+.*' semiring, If the
    % type of A and B differ, the type of A is used.  That is, C=A*B is the
    % same as C = gb.mxm (['+.*' gb.type(A)], A, B).  A and B can be GraphBLAS
    % matrices or MATLAB sparse or full matrices, in any combination.
    % If either A or B are scalars, C=A*B is the same as C=A.*B.
    %
    % See also gb.mxm, gb.emult, times.
    if (isscalar (A) | isscalar (B))
        C = A .* B ;
    else
        C = gb.mxm ('+.*', A, B) ;
    end
    end

    %---------------------------------------------------------------------------
    % rdivide: C = A ./ B
    %---------------------------------------------------------------------------

    function C = rdivide (A, B)
    %TIMES C = A./B, sparse matrix element-wise division
    % C = A./B when B is a matrix results in a dense matrix C, with all entries
    % present.  If A is a matrix and B is a scalar, then C has the pattern of
    % A, except if B is zero and A is double, single, or complex.  In that
    % case, since 0/0 is NaN, C is a dense matrix.  If the types of A and B
    % differ, C has the type of A, and B is typecasted into the type of A
    % before computing C=A./B.  A and B can be GraphBLAS matrices or MATLAB
    % sparse or full matrices, in any combination.
    %
    % See also rdivide, gb.emult, gb.eadd.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.emult ('/', A, B) ;
        else
            % A is a scalar, B is a matrix.  A is expanded to full.
            % The result is a dense gb matrix.
            C = gb.eadd ('/', gb.expand (A, true (size (B))), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) == 0 & isfloat (A))
                % 0/0 is Nan, and thus must be computed computed if A is
                % floating-point.  The result is a dense matrix.
                C = gb.eadd ('/', A, gb.expand (B, true (size (A)))) ;
            else
                % The scalar B is nonzero so just compute A/B in the pattern of
                % A.  The result is sparse (the pattern of A).
                C = gb.emult ('/', A, gb.expand (B, A)) ;
            end
        else
            % both A and B are matrices.  The result is a dense matrix.
            C = gb.eadd ('/', full (A), full (B)) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % ldivide: C = A .\ B
    %---------------------------------------------------------------------------

    function C = ldivide (A, B)
    %TIMES C = A.\B, sparse matrix element-wise division
    % C = A.\B is the same as C = B./A.  See rdivide for more details.
    C = rdivide (B, A) ;
    end

    %---------------------------------------------------------------------------
    % mrdivide: C = A / B
    %---------------------------------------------------------------------------

    function C = mrdivide (A, B)
    % C = A/B, matrix right division
    %
    % If B is a scalar, then C = A./B is computed; see 'help rdivide'.
    %
    % Otherwise, C is computed by first converting A and B to MATLAB sparse
    % matrices, and the result is converted back to a GraphBLAS double or
    % complex matrix.
    if (isscalar (B))
        C = rdivide (A, B) ;
    else
        C = gb (builtin ('mrdivide', double (A), double (B))) ;
    end
    end

    %---------------------------------------------------------------------------
    % mldivide: C = A \ B
    %---------------------------------------------------------------------------

    function C = mldivide (A, B)
    % C = A\B, matrix left division
    %
    % If A is a scalar, then C = A.\B is computed; see 'help ldivide'.
    %
    % Otherwise, C is computed by first converting A and B to MATLAB sparse
    % matrices, and the result is converted back to a GraphBLAS double or
    % complex matrix.
    if (isscalar (A))
        C = rdivide (B, A) ;
    else
        C = gb (builtin ('mldivide', double (A), double (B))) ;
    end
    end

    %---------------------------------------------------------------------------
    % power: C = A .^ B
    %---------------------------------------------------------------------------

    function C = power (A, B)
    %.^ Array power.
    % C = A.^B computes element-wise powers.  One or both of A and B may be
    % scalars.  Otherwise, A and B must have the same size.  The computation
    % takes O(m*n) time if the matrices are m-by-n, except in the case that
    % B is a positive scalar (greater than zero).  In that case, the pattern
    % of C is a subset of the pattern of A.
    %
    % Note that complex matrices are not yet supported.

    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            A = full (A) ;
            B = full (B) ;
        else
            % A is a scalar, B is a matrix; expand A to the size of B
            A = gb.expand (A, true (size (B))) ;
            B = full (B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) <= 0)
                % so the result is full
                A = full (A) ;
                B = gb.expand (B, true (size (A))) ;
            else
                % The scalar b is > 0, and thus 0.^b is zero.  The result is
                % sparse.  B is expanded to a matrix wit the same pattern as A.
                B = gb.expand (B, A) ;
            end
        else
            % both A and B are matrices.
            A = full (A) ;
            B = full (B) ;
        end
    end

    % GraphBLAS does not have a binary operator f(x,y)=x^y.  It could be
    % constructed as a user-defined operator, but this is reasonably fast.
    % FUTURE: create a binary operator f(x,y) = x^y.
    [m, n] = size (A) ;
    [I, J, Ax] = gb.extracttuples (A) ;
    [I, J, Bx] = gb.extracttuples (B) ;
    C = gb.select ('nonzero', gb.build (I, J, (Ax .^ Bx), m, n)) ;
    end

    %---------------------------------------------------------------------------
    % mpower: C = A ^ B
    %---------------------------------------------------------------------------

    function C = mpower (A, B)
    %A^B
    % A must be a square matrix.  B must an integer >= 0.
    [m n] = size (A) ;
    if (m ~= n)
        error ('For C=A^B, A must be square') ;
    end
    if (~isscalar (B))
        error ('For C=A^B, B must be a non-negative integer scalar') ;
    end
    b = get_scalar (B) ;
    if (isreal (b) && isfinite (b) && round (b) == b && b >= 0)
        if (b == 0)
            % C is identity, of the same type as A
            % FUTURE: ones (...) needs to be 'double' if A is complex.
            C = gb.build (1:n, 1:n, ones (1, n, gb.type (A)), n, n) ;
        else
            % C = A^b where b > 0 is an integer
            C = compute_mpower (A, b) ;
        end
    else
        error ('For C=A^B, B must be a non-negative integer scalar') ;
    end
    end

    %---------------------------------------------------------------------------
    % lt: C = (A < B)
    %---------------------------------------------------------------------------

    function C = lt (A, B)
    %A < B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    % The pattern of C depends on the type of inputs:
    % A scalar, B scalar:  C is scalar.
    % A scalar, B matrix:  C is full if A<0, otherwise C is a subset of B.
    % B scalar, A matrix:  C is full if B>0, otherwise C is a subset of A.
    % A matrix, B matrix:  C has the pattern of the set union, A+B.
    % Zeroes are then dropped from C after it is computed.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.select ('nonzero', gb.emult ('<', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) < 0)
                % since a < 0, entries not present in B result in a true value,
                % so the result is dense.  Expand A to a dense matrix.
                A = gb.expand (A, true (size (B))) ;
                C = gb.select ('nonzero', gb.emult ('<', A, full (B))) ;
            else
                % since a >= 0, entries not present in B result in a false
                % value, so the result is a sparse subset of B.  select all
                % entries in B > a, then convert to true.
                C = gb.apply ('1.logical', gb.select ('>thunk', B, A)) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) > 0)
                % since b > 0, entries not present in A result in a true value,
                % so the result is dense.  Expand B to a dense matrix.
                B = gb.expand (B, true (size (A))) ;
                C = gb.select ('nonzero', gb.emult ('<', full (A), B)) ;
            else
                % since b <= 0, entries not present in A result in a false
                % value, so the result is a sparse subset of A.  Select all
                % entries in A < b, then convert to true.
                C = gb.apply ('1.logical', gb.select ('<thunk', A, B)) ;
            end
        else
            % both A and B are matrices.  C is the set union of A and B.
            C = sparse_comparator ('<', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % gt: C = (A > B)
    %---------------------------------------------------------------------------

    function C = gt (A, B)
    %A > B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.
    C = lt (B, A) ;
    end

    %---------------------------------------------------------------------------
    % le: C = (A <= B)
    %---------------------------------------------------------------------------

    function C = le (A, B)
    %A <= B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    % The pattern of C depends on the type of inputs:
    % A scalar, B scalar:  C is scalar.
    % A scalar, B matrix:  C is full if A<=0, otherwise C is a subset of B.
    % B scalar, A matrix:  C is full if B>=0, otherwise C is a subset of A.
    % A matrix, B matrix:  C is full.
    % Zeroes are then dropped from C after it is computed.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.select ('nonzero', gb.emult ('<=', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) <= 0)
                % since a <= 0, entries not present in B result in a true value,
                % so the result is dense.  Expand A to a dense matrix.
                A = gb.expand (A, true (size (B))) ;
                C = gb.select ('nonzero', gb.emult ('<=', A, full (B))) ;
            else
                % since a > 0, entries not present in B result in a false
                % value, so the result is a sparse subset of B.  select all
                % entries in B >= a, then convert to true.
                C = gb.apply ('1.logical', gb.select ('>=thunk', B, A)) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) >= 0)
                % since b >= 0, entries not present in A result in a true value,
                % so the result is dense.  Expand B to a dense matrix.
                B = gb.expand (B, true (size (A))) ;
                C = gb.select ('nonzero', gb.emult ('<=', full (A), B)) ;
            else
                % since b < 0, entries not present in A result in a false
                % value, so the result is a sparse subset of A.  select all
                % entries in A <= b, then convert to true.
                C = gb.apply ('1.logical', gb.select ('<=thunk', A, B)) ;
            end
        else
            % both A and B are matrices.  C is full.
            C = dense_comparator ('<=', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % ge: C = (A >= B)
    %---------------------------------------------------------------------------

    function C = ge (A, B)
    %A >= B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.
    C = le (B, A) ;
    end

    %---------------------------------------------------------------------------
    % ne: C = (A ~= B)
    %---------------------------------------------------------------------------

    function C = ne (A, B)
    %A ~= B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    % The pattern of C depends on the type of inputs:
    % A scalar, B scalar:  C is scalar.
    % A scalar, B matrix:  C is full if A~=0, otherwise C is a subset of B.
    % B scalar, A matrix:  C is full if B~=0, otherwise C is a subset of A.
    % A matrix, B matrix:  C is full.
    % Zeroes are then dropped from C after it is computed.

    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.select ('nonzero', gb.emult ('==', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) ~= 0)
                % since a ~= 0, entries not present in B result in a true value,
                % so the result is dense.  Expand A to a dense matrix.
                A = gb.expand (A, true (size (B))) ;
                C = gb.select ('nonzero', gb.emult ('~=', A, full (B))) ;
            else
                % since a == 0, entries not present in B result in a false
                % value, so the result is a sparse subset of B.  select all
                % entries in B ~= 0, then convert to true.
                C = gb.apply ('1.logical', gb.select ('nonzero', B)) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) ~= 0)
                % since b ~= 0, entries not present in A result in a true value,
                % so the result is dense.  Expand B to a dense matrix.
                B = gb.expand (B, true (size (A))) ;
                C = gb.select ('nonzero', gb.emult ('~=', full (A), B)) ;
            else
                % since b == 0, entries not present in A result in a false
                % value, so the result is a sparse subset of A.  select all
                % entries in A ~= 0, then convert to true.
                C = gb.apply ('1.logical', gb.select ('nonzero', A)) ;
            end
        else
            % both A and B are matrices.  C is full.
            C = dense_comparator ('~=', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % eq: C = (A == B)
    %---------------------------------------------------------------------------

    function C = eq (A, B)
    %A == B
    % Element-by-element comparison of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    % The pattern of C depends on the type of inputs:
    % A scalar, B scalar:  C is scalar.
    % A scalar, B matrix:  C is full if A==0, otherwise C is a subset of B.
    % B scalar, A matrix:  C is full if B==0, otherwise C is a subset of A.
    % A matrix, B matrix:  C is full.
    % Zeroes are then dropped from C after it is computed.

    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars
            C = gb.select ('nonzero', gb.emult ('==', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) == 0)
                % since a == 0, entries not present in B result in a true value,
                % so the result is dense.  Expand A to a dense matrix.
                A = gb.expand (A, true (size (B))) ;
                C = gb.select ('nonzero', gb.emult ('==', A, full (B))) ;
            else
                % since a ~= 0, entries not present in B result in a false
                % value, so the result is a sparse subset of B.  select all
                % entries in B == a, then convert to true.
                C = gb.apply ('1.logical', gb.select ('==thunk', B, A)) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) == 0)
                % since b == 0, entries not present in A result in a true value,
                % so the result is dense.  Expand B to a dense matrix.
                B = gb.expand (B, true (size (A))) ;
                C = gb.select ('nonzero', gb.emult ('==', full (A), B)) ;
            else
                % since b ~= 0, entries not present in A result in a false
                % value, so the result is a sparse subset of A.  select all
                % entries in A == b, then convert to true.
                C = gb.apply ('1.logical', gb.select ('==thunk', A, B)) ;
            end
        else
            % both A and B are matrices.  C is full.
            C = dense_comparator ('==', A, B) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % and: C = (A & B)
    %---------------------------------------------------------------------------

    function C = and (A, B)
    %& logical AND.
    % Element-by-element logical AND of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    if (isscalar (A))
        if (isscalar (B))
            % A and B are scalars
            C = gb.select ('nonzero', gb.emult ('&.logical', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) == 0)
                % A is false, so C is empty, the same size as B
                [m n] = size (B) ;
                C = gb (m, n, 'logical') ;
            else
                % A is true, so C is B typecasted to logical
                C = gb (gb.select ('nonzero', B), 'logical') ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) == 0)
                % B is false, so C is empty, the same size as A
                [m n] = size (A) ;
                C = gb (m, n, 'logical') ;
            else
                % B is true, so C is A typecasted to logical
                C = gb (gb.select ('nonzero', A), 'logical') ;
            end
        else
            % both A and B are matrices.  C is the set intersection of A and B
            C = gb.select ('nonzero', gb.emult ('&.logical', A, B)) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % or: C = (A | B)
    %---------------------------------------------------------------------------

    function C = or (A, B)
    %| logical OR.
    % Element-by-element logical OR of A and B.  One or both may be scalars.
    % Otherwise, A and B must have the same size.

    if (isscalar (A))
        if (isscalar (B))
            % A and B are scalars
            C = gb.select ('nonzero', gb.emult ('|.logical', A, B)) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) == 0)
                % A is false, so C is B typecasted to logical
                C = gb (gb.select ('nonzero', B), 'logical') ;
            else
                % A is true, so C is a full matrix the same size as B
                C = gb (true (size (B))) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) == 0)
                % B is false, so C is A typecasted to logical
                C = gb (gb.select ('nonzero', A), 'logical') ;
            else
                % B is true, so C is a full matrix the same size as A
                C = gb (true (size (A))) ;
            end
        else
            % both A and B are matrices.  C is the set union of A and B
            C = gb.select ('nonzero', gb.eadd ('|.logical', A, B)) ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % not: C = (~A)
    %---------------------------------------------------------------------------

    function C = not (A)
    %~ logical negation of a GraphBLAS matrix
    % C = ~A computes the logical negation of a GraphBLAS matrix A.  The result
    % C is dense, and the computation takes O(m*n) time and space, so sparsity
    % is not exploited.  To negate just the entries in the pattern of A, use
    % C = gb.apply ('~.logical', A), which has the same pattern as A.
    %
    % See also gb.apply.
    C = gb.select ('nonzero', gb.apply ('~.logical', full (A))) ;
    end

    %---------------------------------------------------------------------------
    % ctranspose: C = A'
    %---------------------------------------------------------------------------

    function C = ctranspose (A)
    %CTRANSPOSE C = A', matrix transpose a GraphBLAS matrix.
    % Note that complex matrices are not yet supported.  When they are, this
    % will compute the complex conjugate transpose C=A' when A is complex.
    %
    % See also gb.gbtranspose, transpose.
    C = gb.gbtranspose (A) ;
    end

    %---------------------------------------------------------------------------
    % transpose: C = A'
    %---------------------------------------------------------------------------

    function C = transpose (A)
    %TRANSPOSE C = A.', array transpose of a GraphBLAS matrix.
    %
    % See also gb.gbtranspose, ctranspose.
    C = gb.gbtranspose (A) ;
    end

    %---------------------------------------------------------------------------
    % horzcat: C = [A1, A2, ..., An]
    %---------------------------------------------------------------------------

    function C = horzcat (varargin)
    %HORZCAT Horizontal concatenation.
    % [A B] or [A,B] is the horizontal concatenation of A and B.
    % A and B may be GraphBLAS or MATLAB matrices, in any combination.
    % Multiple matrices may be concatenated, as [A, B, C, ...].

    % FUTURE: this will be much faster when it is a mexFunction.

    % determine the size of each matrix and the size of the result
    nmatrices = length (varargin) ;
    nvals = zeros (1, nmatrices) ;
    ncols = zeros (1, nmatrices) ;
    A = varargin {1} ;
    [m n] = size (A) ;
    nvals (1) = gb.nvals (A) ;
    ncols (1) = n ;
    type = gb.type (A) ;
    clear A
    for k = 2:nmatrices
        B = varargin {k} ;
        [m2 n] = size (B) ;
        if (m ~= m2)
            error('Dimensions of arrays being concatenated are not consistent');
        end
        nvals (k) = gb.nvals (B) ;
        ncols (k) = n ;
        clear B ;
    end
    ncols = [0 cumsum(ncols)] ;
    nvals = [0 cumsum(nvals)] ;
    cnvals = nvals (end) ;
    n = ncols (end) ;

    % allocate the I,J,X arrays
    I = zeros (cnvals, 1, 'int64') ;
    J = zeros (cnvals, 1, 'int64') ;
    X = zeros (cnvals, 1, type) ;

    % fill the I,J,X arrays
    d.kind = 'zero-based' ;
    for k = 1:nmatrices
        [i, j, x] = gb.extracttuples (varargin {k}, d) ;
        noffset = int64 (ncols (k)) ;
        koffset = nvals (k) ;
        kvals = gb.nvals (varargin {k}) ;
        I ((koffset+1):(koffset+kvals)) = i ;
        J ((koffset+1):(koffset+kvals)) = j + noffset ;
        X ((koffset+1):(koffset+kvals)) = x ;
    end

    % build the output matrix
    C = gb.build (I, J, X, m, n) ;
    end

    %---------------------------------------------------------------------------
    % vertcat: C = [A1 ; A2 ; ... ; An]
    %---------------------------------------------------------------------------

    function C = vertcat (varargin)
    %VERTCAT Vertical concatenation.
    % [A ; B] is the vertical concatenation of A and B.
    % A and B may be GraphBLAS or MATLAB matrices, in any combination.
    % Multiple matrices may be concatenated, as [A ; B ; C ; ...].

    % FUTURE: this will be much faster when it is a mexFunction.
    % The version below requires a sort in gb.build.

    % determine the size of each matrix and the size of the result
    nmatrices = length (varargin) ;
    nvals = zeros (1, nmatrices) ;
    nrows = zeros (1, nmatrices) ;
    A = varargin {1} ;
    [m n] = size (A) ;
    nvals (1) = gb.nvals (A) ;
    nrows (1) = m ;
    type = gb.type (A) ;
    clear A
    for k = 2:nmatrices
        B = varargin {k} ;
        [m n2] = size (B) ;
        if (n ~= n2)
            error('Dimensions of arrays being concatenated are not consistent');
        end
        nvals (k) = gb.nvals (B) ;
        nrows (k) = m ;
        clear B ;
    end
    nrows = [0 cumsum(nrows)] ;
    nvals = [0 cumsum(nvals)] ;
    cnvals = nvals (end) ;
    m = nrows (end) ;

    % allocate the I,J,X arrays
    I = zeros (cnvals, 1, 'int64') ;
    J = zeros (cnvals, 1, 'int64') ;
    X = zeros (cnvals, 1, type) ;

    % fill the I,J,X arrays
    d.kind = 'zero-based' ;
    for k = 1:nmatrices
        [i, j, x] = gb.extracttuples (varargin {k}, d) ;
        moffset = int64 (nrows (k)) ;
        koffset = nvals (k) ;
        kvals = gb.nvals (varargin {k}) ;
        I ((koffset+1):(koffset+kvals)) = i + moffset ;
        J ((koffset+1):(koffset+kvals)) = j ;
        X ((koffset+1):(koffset+kvals)) = x ;
    end

    % build the output matrix
    C = gb.build (I, J, X, m, n) ;
    end

    %---------------------------------------------------------------------------
    % subsref: C = A (I,J)
    %---------------------------------------------------------------------------

    function C = subsref (A, S)
    %SUBSREF C = A(I,J) or C = A(I); extract submatrix of a GraphBLAS matrix.
    % C = A(I,J) extracts the A(I,J) submatrix of the GraphBLAS matrix A.  With
    % a single index, C = A(I) extracts a subvector C of a vector A.  Linear
    % indexing of a matrix is not yet supported.
    %
    % x = A (M) for a logical matrix M constructs an nnz(M)-by-1 vector x, for
    % MATLAB-style logical indexing.  A or M may be MATLAB sparse or full
    % matrices, or GraphBLAS matrices, in any combination.  M must be either a
    % MATLAB logical matrix (sparse or dense), or a GraphBLAS logical matrix;
    % that is, gb.type (M) must be 'logical'.
    %
    % NOTE: GraphBLAS can construct huge sparse matrices, but they cannot
    % always be indexed with A(I,J), because of a limitation of the colon
    % notation in the MATLAB subsref method.  A colon expression is expanded
    % into an explicit vector, but can be too big.  Use gb.extract in this
    % case, which can be passed the three integers start:inc:fini.
    %
    % Example:
    %
    %   n = 1e14 ;
    %   H = gb (n, n)               % a huge empty matrix
    %   I = [1 1e9 1e12 1e14] ;
    %   M = magic (4)
    %   H (I,I) = M
    %   J = {1, 1e13} ;             % represents 1:1e13 colon notation
    %   C = gb.extract (H, J, J)    % this is very fast
    %   E = H (1:1e13, 1:1e13)      % but this is not possible 
    %
    % See also subsasgn, gb.subassign, gb.assign, gb.extract.

    if (~isequal (S.type, '()'))
        error ('index type %s not supported', S.type) ;
    end
    ndims = length (S.subs) ;
    if (ndims == 1)
        if (isequal (gb.type (S.subs {1}), 'logical'))
            % C = A (M) for a logical indexing
            M = S.subs {1} ;
            if (isa (M, 'gb'))
                M = M.opaque ;
            end
            if (isa (A, 'gb'))
                A = A.opaque ;
            end
            C = gb (gblogextract (A, M)) ;
        else
            % C = A (I) for a vector A
            if (~isvector (A))
                error ('Linear indexing of a gb matrix is not yet supported') ;
            end
            [I, whole_vector] = get_index (S.subs (1)) ;
            if (size (A, 1) > 1)
                C = gb.extract (A, I, { }) ;
            else
                C = gb.extract (A, { }, I) ;
            end
            if (whole_vector & size (C,1) == 1)
                C = C.' ;
            end
        end
    elseif (ndims == 2)
        % C = A (I,J)
        I = get_index (S.subs (1)) ;
        J = get_index (S.subs (2)) ;
        C = gb.extract (A, I, J) ;
    else
        error ('%dD indexing not supported', ndims) ;
    end
    end

    %---------------------------------------------------------------------------
    % subsasgn: C (I,J) = A
    %---------------------------------------------------------------------------

    function Cout = subsasgn (Cin, S, A)
    %SUBSASGN C(I,J) = A or C(I) = A; assign submatrix into a GraphBLAS matrix.
    % C(I,J) = A assigns A into the C(I,J) submatrix of the GraphBLAS matrix C.
    % A must be either a matrix of size length(I)-by-length(J), or a scalar.
    % Note that C(I,J) = 0 differs from C(I,J) = sparse (0).  The former places
    % an explicit entry with value zero in all positions of C(I,J).  The latter
    % deletes all entries in C(I,J).  With a MATLAB sparse matrix C, both
    % statements delete all entries in C(I,J) since MATLAB never stores
    % explicit zeros in its sparse matrices.
    %
    % With a single index, C(I) = A, both C and A must be vectors; linear
    % indexing is not yet supported.  In this case A must either be a vector
    % of length the same as I, or a scalar.
    %
    % If M is a logical matrix, C (M) = x is an assignment via logical indexing,
    % where C and M have the same size, and x(:) is either a vector of length
    % nnz (M), or a scalar.
    %
    % Note that C (M) = A (M), where the same logical matrix M is used on both
    % the sides of the assignment, is identical to C = gb.subassign (C, M, A).
    % If C and A (or M) are GraphBLAS matrices, C (M) = A (M) uses GraphBLAS
    % via operator overloading.  The statement C (M) = A (M) takes about twice
    % the time as C = gb.subassign (C, M, A), so the latter is preferred for
    % best performance.  However, both methods in GraphBLAS are many thousands
    % of times faster than C (M) = A (M) using purely MATLAB sparse matrices C,
    % M, and A, when the matrices are large.  So either method works fine,
    % relatively speaking.
    %
    % If I or J are very large colon notation expressions, then C(I,J)=A is not
    % possible, because MATLAB creates I and J as explicit lists first.  See
    % gb.subassign instead.  See also the example with 'help gb.extract'.
    %
    % See also subsref, gb.assign, gb.subassign.

    if (~isequal (S.type, '()'))
        error ('index type %s not supported', S.type) ;
    end
    ndims = length (S.subs) ;
    if (ndims == 1)
        if (isequal (gb.type (S.subs {1}), 'logical'))
            % C (M) = A for logical assignment
            M = S.subs {1} ;
            if (isscalar (A))
                % C (M) = scalar
                Cout = gb.subassign (Cin, M, A) ;
            else
                % C (M) = A where A is a vector
                if (isa (M, 'gb'))
                    M = M.opaque ;
                end
                if (size (A, 2) ~= 1)
                    % make sure A is a column vector of size mnz-by-1
                    A = A (:) ;
                end
                if (isa (A, 'gb'))
                    A = A.opaque ;
                end
                if (isa (Cin, 'gb'))
                    Cin = Cin.opaque ;
                end
                Cout = gb (gblogassign (Cin, M, A)) ;
            end
        else
            % C (I) = A where C and A are vectors
            I = get_index (S.subs (1)) ;
            Cout = gb.subassign (Cin, A, I) ;
        end
    elseif (ndims == 2)
        I = get_index (S.subs (1)) ;
        J = get_index (S.subs (2)) ;
        Cout = gb.subassign (Cin, A, I, J) ;
    else
        error ('%dD indexing not supported', ndims) ;
    end
    end

    %---------------------------------------------------------------------------
    % end: object indexing
    %---------------------------------------------------------------------------

    function index = end (G, k, ndims)
    %END Last index in an indexing expression for a GraphBLAS matrix.
    if (ndims == 1)
        if (~isvector (G))
            error ('Linear indexing not supported') ;
        end
        index = length (G) ;
    elseif (ndims == 2)
        s = size (G) ;
        index = s (k) ;
    else
        error ('%dD indexing not supported', ndims) ;
    end
    end

end

%===============================================================================
methods (Static) %==============================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % gb.clear: clear all internal GraphBLAS workspace and settings
    %---------------------------------------------------------------------------

    function clear
    %GB.CLEAR free all internal workspace in SuiteSparse:GraphBLAS
    %
    % Usage:
    %
    %   gb.clear
    %
    % GraphBLAS keeps an internal workspace to speedup its operations.  It also
    % uses several global settings.  These can both be cleared with gb.clear.
    %
    % This method is optional.  Simply terminating the MATLAB session, or
    % typing 'clear all' will do the same thing.  However, if you are finished
    % with GraphBLAS and wish to free its internal workspace, but do not wish
    % to free everything else freed by 'clear all', then use this method.
    % gb.clear also clears any non-default setting of gb.threads, gb.chunk, and
    % gb.format.
    %
    % See also: clear, gb.threads, gb.chunk, gb.format

    gbclear ;
    end

    %---------------------------------------------------------------------------
    % gb.descriptorinfo: list the contents of a GraphBLAS descriptor
    %---------------------------------------------------------------------------

    function descriptorinfo (d)
    %GB.DESCRIPTOR list the contents of a SuiteSparse:GraphBLAS descriptor
    %
    % Usage:
    %
    %   gb.descriptorinfo
    %   gb.descriptorinfo (d)
    %
    % The GraphBLAS descriptor is a MATLAB struct that can be used to modify
    % the behavior of GraphBLAS operations.  It contains the following
    % components, each of which are a string or a number.  Any component of
    % struct that is not present is set to the default value.  If the
    % descriptor d is empty, or not present, in a GraphBLAS function, all
    % default settings are used.
    %
    % The following descriptor values are strings:
    %
    %   d.out   'default' or 'replace'      determines if C is cleared before
    %                                         the accum/mask step
    %   d.mask  'default' or 'complement'   determines if M or !M is used
    %   d.in0   'default' or 'transpose'    determines A or A' is used
    %   d.in1   'default' or 'transpose'    determines B or B' is used
    %   d.axb   'default', 'Gustavson', 'heap', or 'dot'
    %            determines the method used in gb.mxm.  The default is to let
    %            GraphBLAS determine the method automatically, via a
    %            heuristic.
    %   d.kind   For most gb.methods, this is a string equal to 'default',
    %            'gb', 'sparse', or 'full'.  The default is d.kind = 'gb',
    %            where the GraphBLAS operation returns an object, which is
    %            preferred since GraphBLAS sparse matrices are faster and can
    %            represent many more data types.  However, if you want a
    %            standard MATLAB sparse matrix, use d.kind='sparse'.  Use
    %            d.kind='full' for a MATLAB dense matrix.  For any gb.method
    %            that takes a descriptor, the following uses are the same, but
    %            the first method is faster and takes less temporary workspace:
    %
    %               d.kind = 'sparse' ;
    %               S = gb.method (..., d) ;
    %
    %               % with no d, or d.kind = 'default'
    %               S = double (gb.method (...)) :
    %
    %           [I, J, X] = gb.extracttuples (G,d) uses d.kind = 'one-based' or
    %           'zero-based' to determine the type of I and J.
    %
    % These descriptor values are scalars:
    %
    %   d.nthreads  max # of threads to use; default is omp_get_max_threads.
    %   d.chunk     controls # of threads to use for small problems.
    %
    % gb.descriptorinfo (d) lists the contents of a GraphBLAS descriptor and
    % checks if its contents are valid.  Also refer to the
    % SuiteSparse:GraphBLAS User Guide for more details.
    %
    % See also gb, gb.unopinfo, gb.binopinfo, gb.monoidinfo, gb.semiringinfo.

    if (nargin == 0)
        help gb.descriptorinfo
    else
        gbdescriptorinfo (d) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.unopinfo: list the details of a GraphBLAS unary operator
    %---------------------------------------------------------------------------

    function unopinfo (op, type)
    %GB.UNOPINFO list the details of a GraphBLAS unary operator
    %
    % Usage
    %
    %   gb.unopinfo
    %   gb.unopinfo (op)
    %   gb.unopinfo (op, type)
    %
    % For gb.unopinfo(op), the op must be a string of the form 'op.type',
    % where 'op' is listed below.  The second usage allows the type to be
    % omitted from the first argument, as just 'op'.  This is valid for all
    % GraphBLAS operations, since the type defaults to the type of the input
    % matrix.  However, gb.unopinfo does not have a default type and thus
    % one must be provided, either in the op as gb.unopinfo ('abs.double'), or
    % in the second argument, gb.unopinfo ('abs', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 6 different unary
    % operators, each of which may be used with any of the 11 types, for a
    % total of 6*11 = 66 valid unary operators.  Unary operators are defined by
    % a string of the form 'op.type', or just 'op'.  In the latter case, the
    % type defaults to the type of the matrix inputs to the GraphBLAS
    % operation.
    %
    % The following unary operators are available.
    %
    %   operator name(s)    f(x,y)      |  operator names(s) f(x,y)
    %   ----------------    ------      |  ----------------- ------
    %   identity            x           |  lnot not ~        ~x
    %   ainv - negate       -x          |  one 1             1
    %   minv                1/x         |  abs               abs(x)
    %
    % The logical operator, lnot, also comes in 11 types.  z = lnot.double (x)
    % tests the condition (x ~= 0), and returns the double value 1.0 if true,
    % or 0.0 if false.
    %
    % Example:
    %
    %   % valid unary operators
    %   gb.unopinfo ('abs.double') ;
    %   gb.unopinfo ('not.int32') ;
    %
    %   % invalid unary operator (generates an error; this is a binary op):
    %   gb.unopinfo ('+.double') ;
    %
    % gb.unopinfo generates an error for an invalid op, so user code can test
    % the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.binopinfo, gb.monoidinfo, gb.semiringinfo,
    % gb.descriptorinfo.

    % FUTURE: add complex unary operators

    if (nargin == 0)
        help gb.unopinfo
    elseif (nargin == 1)
        gbunopinfo (op) ;
    else
        gbunopinfo (op, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.binopinfo: list the details of a GraphBLAS binary operator
    %---------------------------------------------------------------------------

    function binopinfo (op, type)
    %GB.BINOPINFO list the details of a GraphBLAS binary operator
    %
    % Usage
    %
    %   gb.binopinfo
    %   gb.binopinfo (op)
    %   gb.binopinfo (op, type)
    %
    % For gb.binopinfo(op), the op must be a string of the form
    % 'op.type', where 'op' is listed below.  The second usage allows the
    % type to be omitted from the first argument, as just 'op'.  This is
    % valid for all GraphBLAS operations, since the type defaults to the
    % type of the input matrices.  However, gb.binopinfo does not have a
    % default type and thus one must be provided, either in the op as
    % gb.binopinfo ('+.double'), or in the second argument, gb.binopinfo
    % ('+', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 25 different binary
    % operators, each of which may be used with any of the 11 types, for
    % a total of 25*11 = 275 valid binary operators.  Binary operators
    % are defined by a string of the form 'op.type', or just 'op'.  In
    % the latter case, the type defaults to the type of the matrix inputs
    % to the GraphBLAS operation.
    %
    % The 6 comparator operators come in two flavors.  For the is*
    % operators, the result has the same type as the inputs, x and y,
    % with 1 for true and 0 for false.  For example isgt.double (pi, 3.0)
    % is the double value 1.0.  For the second set of 6 operators (eq,
    % ne, gt, lt, ge, le), the result is always logical (true or false).
    % In a semiring, the type of the add monoid must exactly match the
    % type of the output of the multiply operator, and thus
    % 'plus.iseq.double' is valid (counting how many terms are equal).
    % The 'plus.eq.double' semiring is valid, but not the same semiring
    % since the 'plus' of 'plus.eq.double' has a logical type and is thus
    % equivalent to 'or.eq.double'.   The 'or.eq' is true if any terms
    % are equal and false otherwise (it does not count the number of
    % terms that are equal).
    %
    % The following binary operators are available.  Many have equivalent
    % synonyms, so that '1st' and 'first' both define the first(x,y) = x
    % operator.
    %
    %   operator name(s) f(x,y)         |   operator names(s) f(x,y)
    %   ---------------- ------         |   ----------------- ------
    %   1st first        x              |   iseq             x == y
    %   2nd second       y              |   isne             x ~= y
    %   min              min(x,y)       |   isgt             x > y
    %   max              max(x,y)       |   islt             x < y
    %   +   plus         x+y            |   isge             x >= y
    %   -   minus        x-y            |   isle             x <= y
    %   rminus           y-x            |   ==  eq           x == y
    %   *   times        x*y            |   ~=  ne           x ~= y
    %   /   div          x/y            |   >   gt           x > y
    %   \   rdiv         y/x            |   <   lt           x < y
    %   |   || or  lor   x | y          |   >=  ge           x >= y
    %   &   && and land  x & y          |   <=  le           x <= y
    %   xor lxor         xor(x,y)       |
    %
    % The three logical operators, lor, land, and lxor, also come in 11
    % types.  z = lor.double (x,y) tests the condition (x~=0) || (y~=0),
    % and returns the double value 1.0 if true, or 0.0 if false.
    %
    % Example:
    %
    %   % valid binary operators
    %   gb.binopinfo ('+.double') ;
    %   gb.binopinfo ('1st.int32') ;
    %
    %   % invalid binary operator (an error; this is a unary op):
    %   gb.binopinfo ('abs.double') ;
    %
    % gb.binopinfo generates an error for an invalid op, so user code can
    % test the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.unopinfo, gb.semiringinfo, gb.descriptorinfo.

    % FUTURE: add complex binary operators

    if (nargin == 0)
        help gb.binopinfo
    elseif (nargin == 1)
        gbbinopinfo (op) ;
    else
        gbbinopinfo (op, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.monoidinfo: list the details of a GraphBLAS monoid
    %---------------------------------------------------------------------------

    function monoidinfo (monoid, type)
    %GB.MONOIDINFO list the details of a GraphBLAS monoid
    %
    % Usage
    %
    %   gb.monoidinfo
    %   gb.monoidinfo (monoid)
    %   gb.monoidinfo (monoid, type)
    %
    % For gb.monoidinfo(op), the op must be a string of the form
    % 'op.type', where 'op' is listed below.  The second usage allows the
    % type to be omitted from the first argument, as just 'op'.  This is
    % valid for all GraphBLAS operations, since the type defaults to the
    % type of the input matrices.  However, gb.monoidinfo does not have a
    % default type and thus one must be provided, either in the op as
    % gb.monoidinfo ('+.double'), or in the second argument,
    % gb.monoidinfo ('+', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 44 different
    % monoids.  The valid monoids are: '+', '*', 'max', and 'min' for all
    % but the 'logical' type, and '|', '&', 'xor', and 'ne' for the
    % 'logical' type.
    %
    % Example:
    %
    %   % valid monoids
    %   gb.monoidinfo ('+.double') ;
    %   gb.monoidinfo ('*.int32') ;
    %
    %   % invalid monoids
    %   gb.monoidinfo ('1st.int32') ;
    %   gb.monoidinfo ('abs.double') ;
    %
    % gb.monoidinfo generates an error for an invalid monoid, so user
    % code can test the validity of an op with the MATLAB try/catch
    % mechanism.
    %
    % See also gb.unopinfo, gb.binopinfo, gb.semiringinfo,
    % gb.descriptorinfo.

    % FUTURE: add complex monoids

    if (nargin == 0)
        help gb.monoidinfo
    elseif (nargin == 1)
        gbmonoidinfo (monoid) ;
    else
        gbmonoidinfo (monoid, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.semiringinfo: list the details of a GraphBLAS semiring
    %---------------------------------------------------------------------------

    function semiringinfo (s, type)
    %GB.SEMIRINGINFO list the details of a GraphBLAS semiring
    %
    % Usage
    %
    %   gb.semiringinfo
    %   gb.semiringinfo (semiring)
    %   gb.semiringinfo (semiring, type)
    %
    % For gb.semiring(semiring), the semiring must be a string of the form
    % 'add.mult.type', where 'add' and 'mult' are binary operators.  The second
    % usage allows the type to be omitted from the first argument, as just
    % 'add.mult'.  This is valid for all GraphBLAS operations, since the type
    % defaults to the type of the input matrices.  However, gb.semiringinfo
    % does not have a default type and thus one must be provided, either in the
    % semiring as gb.semiringinfo ('+.*.double'), or in the second argument,
    % gb.semiringinfo ('+.*', 'double').
    %
    % The add operator must be a valid monoid: plus, times, min, max, and the
    % boolean operators or.logical, and.logical, ne.logical, and xor.logical.
    % The binary operator z=f(x,y) of a monoid must be associative and
    % commutative, with an identity value id such that f(x,id) = f(id,x) = x.
    % Furthermore, the types of x, y, and z for the monoid operator f must all
    % be the same.  Thus, the '<.double' is not a valid monoid operator, since
    % its 'logical' output type does not match its 'double' inputs, and since
    % it is neither associative nor commutative.  Thus, <.*.double is not a
    % valid semiring.
    %
    % Example:
    %
    %   % valid semirings
    %   gb.semiringinfo ('+.*.double') ;
    %   gb.semiringinfo ('min.1st.int32') ;
    %
    %   % invalid semiring (generates an error; since '<' is not a monoid)
    %   gb.semiringinfo ('<.*.double') ;
    %
    % gb.semiringinfo generates an error for an invalid semiring, so user code
    % can test the validity of a semiring with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.unopinfo, gb.binopinfo, gb.descriptorinfo.

    % FUTURE: add complex semirings

    if (nargin == 0)
        help gb.semiringinfo
    elseif (nargin == 1)
        gbsemiringinfo (s) ;
    else
        gbsemiringinfo (s, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.threads: get/set the # of threads to use in GraphBLAS
    %---------------------------------------------------------------------------

    function nthreads = threads (varargin)
    %GB.THREADS get/set the max number of threads to use in GraphBLAS
    %
    % Usage:
    %   nthreads = gb.threads ;      % get the current maximum # of threads
    %   gb.threads (nthreads) ;      % set the maximum # of threads
    %
    % gb.threads gets and/or sets the maximum number of threads to use in
    % GraphBLAS.  By default, if GraphBLAS has been compiled with OpenMP, it
    % uses the number of threads returned by omp_get_max_threads.  Otherwise,
    % it can only use a single thread.
    %
    % Changing the number of threads with gb.threads(nthreads) causes all
    % subsequent GraphBLAS operations to use at most the given number of
    % threads.  GraphBLAS may use fewer threads, if the problem is small (see
    % gb.chunk).  The setting is kept for the remainder of the current MATLAB
    % session, or until 'clear all' or gb.clear is used, at which point the
    % setting reverts to the default number of threads.
    %
    % MATLAB can detect the number of physical and logical cores via an
    % undocumented builtin function: ncores = feature('numcores'), or via
    % maxNumCompThreads.
    %
    % Example:
    %
    %   feature ('numcores') ;          % print info about cores
    %   ncores = feature ('numcores') ; % get # of logical cores MATLAB uses
    %   ncores = maxNumCompThreads ;    % same as feature ('numcores')
    %   gb.threads (2*ncores) ;         % GraphBLAS will use <= 2*ncores threads
    %
    % See also feature, maxNumCompThreads, gb.chunk.

    nthreads = gbthreads (varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % gb.chunk: get/set the chunk size to use in GraphBLAS
    %---------------------------------------------------------------------------

    function c = chunk (varargin)
    %GB.CHUNK get/set the chunk size to use in GraphBLAS
    %
    % Usage:
    %   c = gb.chunk ;      % get the current chunk c
    %   gb.chunk (c) ;      % set the chunk c
    %
    % gb.chunk gets and/or sets the chunk size to use in GraphBLAS, which
    % controls how many threads GraphBLAS uses for small problems.  The default
    % is 4096.  If w is a measure of the work required (w = nvals(A) + nvals(B)
    % for C=A+B, for example), then the number of threads GraphBLAS uses is
    % min (max (1, floor (w/c)), gb.nthreads).
    %
    % Changing the chunk via gb.chunk(c) causes all subsequent GraphBLAS
    % operations to use that chunk size c.  The setting persists for the
    % current MATLAB session, or until 'clear all' or gb.clear is used, at
    % which point the setting reverts to the default.
    %
    % See also gb.threads.

    c = gbchunk (varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % gb.nvals: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function e = nvals (A)
    %GB.NVALS the number of entries in a matrix.
    % gb.nvals (A) is the number of explicit entries in a GraphBLAS matrix.
    % Note that the entries can have any value, including zero.  MATLAB drops
    % zero-valued entries from its sparse matrices.  This cannot be done in a
    % GraphBLAS matrix because of the different semirings that may be used.  In
    % a shortest-path problem, for example, and edge with weight zero is very
    % different from no edge at all.
    %
    % For a MATLAB sparse matrix S, gb.nvals (S) and nnz (S) are the same.
    % For a MATLAB full matrix F, gb.nvals (F) and numel (F) are the same.
    %
    % See also nnz, numel.

    if (isa (A, 'gb'))
        e = gbnvals (A.opaque) ;
    else
        e = gbnvals (A) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.empty: construct an empty GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function G = empty (arg1, arg2)
    %GB.EMPTY construct an empty GraphBLAS sparse matrix
    % C = gb.empty is a 0-by-0 empty matrix.
    % C = gb.empty (m) is an m-by-0 empty matrix.
    % C = gb.empty ([m n]) or gb.empty (m,n) is an m-by-n empty matrix, where
    % one of m or n must be zero.
    %
    % All matrices are constructed with the 'double' type.  Use gb (m,n,type)
    % to construct empty single, int*, uint*, and logical m-by-n matrices.
    %
    % See also gb.
    m = 0 ;
    n = 0 ;
    if (nargin == 1)
        if (length (arg1) == 1)
            m = arg1 (1) ;
        elseif (length (arg1) == 2)
            m = arg1 (1) ;
            n = arg1 (2) ;
        else
            error ('invalid dimensions') ;
        end
    elseif (nargin == 2)
        m = arg1 ;
        n = arg2 ;
    end
    if (~ ((m == 0) || (n == 0)))
        error ('At least one dimension must be zero') ;
    end
    G = gb (m, n) ;
    end

    %---------------------------------------------------------------------------
    % gb.type: get the type of a MATLAB or GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function s = type (X)
    %GB.TYPE get the type of a MATLAB or GraphBLAS matrix.
    % s = gb.type (X) returns the type of a GraphBLAS matrix X as a string:
    % 'double', 'single', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    % 'uint32', 'uint64', 'logical', or (in the future) 'complex'.  Note that
    % 'complex' is treated as a type, not an attribute, which differs from the
    % MATLAB convention.  Note that complex matrices are not yet supported.
    %
    % If X is not a GraphBLAS matrix, gb.type (X) is the same as class (X),
    % except when X is a MATLAB double complex matrix, which case gb.type (X)
    % will be 'complex' (in the future).
    %
    % See also class, gb.
    if (isa (X, 'gb'))
        % extract the GraphBLAS opaque matrix struct and then get its type
        s = gbtype (X.opaque) ;
    elseif (isobject (X))
        % the gbtype mexFunction cannot handle object inputs, so use class (X)
        s = class (X) ;
    else
        % get the type of a MATLAB matrix, cell, char, function_handle, ...
        s = gbtype (X) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.format: get/set the default GraphBLAS matrix format
    %---------------------------------------------------------------------------

    function f = format (arg)
    %GB.FORMAT get/set the default GraphBLAS matrix format.
    %
    % In its ANSI C interface, SuiteSparse:GraphBLAS stores its matrices by
    % row, by default, since that format tends to be fastest for graph
    % algorithms, but it can also store its matrices by column.  MATLAB sparse
    % and dense sparse matrices are always stored by column.  For better
    % compatibility with MATLAB sparse matrices, the default for the MATLAB
    % interface for SuiteSparse:GraphBLAS is to store matrices by column.  This
    % has performance implications, and algorithms should be designed
    % accordingly.  The default format can be can changed via:
    %
    %   gb.format ('by row')
    %   gb.format ('by col')
    %
    % which changes the format of all subsequent GraphBLAS matrices.  Existing
    % gb matrices are not affected.
    %
    % The current default global format can be queried with
    %
    %   f = gb.format ;
    %
    % which returns the string 'by row' or 'by col'.
    %
    % Since MATLAB sparse and dense matrices are always 'by col', converting
    % them to a gb matrix 'by row' requires an internal transpose of the
    % format.  That is, if A is a MATLAB sparse or dense matrix,
    %
    %   gb.format ('by row')
    %   G = gb (A)
    %
    % Constructs a double gb matrix G that is held by row, but this takes more
    % work than if G is held by column:
    %
    %   gb.format ('by col')
    %   G = gb (A)
    %
    % If a subsequent algorithm works better with its matrices held by row,
    % then this transformation can save significant time in the long run.
    % Graph algorithms tend to be faster with their matrices held by row, since
    % the edge (i,j) is typically the entry G(i,j) in the matrix G, and most
    % graph algorithms need to know the outgoing edges of node i.  This is
    % G(i,:), which is very fast if G is held by row, but very slow if G is
    % held by column.
    %
    % When the gb.format (f) is changed, all subsequent matrices are created in
    % the given format f.  All prior matrices created before gb.format (f) are
    % kept in their same format; this setting only applies to new matrices.
    % Operations on matrices can be done with any mix of with different
    % formats.  The format only affects time and memory usage, not the results.
    %
    % This setting is reset to 'by col', by 'clear all' or by gb.clear.
    %
    % To query the format for a given GraphBLAS matrix G, use the following
    % (which does not affect the global format setting):
    %
    %   f = gb.format (G)
    %
    % Examples:
    %
    %   A = sparse (rand (4))
    %   gb.format ('by row') ;
    %   G = gb (A)
    %   gb.format (G)
    %   gb.format ('by col') ;      % set the format to 'by col'
    %   G = gb (A)
    %   gb.format (G)               % query the format of G
    %
    % See also gb.

    if (nargin == 0)
        % f = gb.format ; get the global format
        f = gbformat ;
    elseif (nargin == 1)
        if (isa (arg, 'gb'))
            % f = gb.format (G) ; get the format of the matrix G
            f = gbformat (arg.opaque) ;
        else
            % f = gb.format (f) ; set the global format for all future matrices
            f = gbformat (arg) ;
        end
    else
        error ('usage: f = gb.format or f = gb.format (f)') ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.expand: expand a scalar to a matrix
    %---------------------------------------------------------------------------

    function C = expand (scalar, S)
    %GB.EXPAND expand a scalar to a matrix
    % The scalar is expanded to the pattern of S, as in C = scalar*spones(S).
    % C has the same type as the scalar.  The numerical values of S are
    % ignored; only the pattern of S is used.
    C = gb.gbkron (['1st.' gb.type(scalar)], scalar, S) ;
    end

    %---------------------------------------------------------------------------
    % gb.eye: sparse identity matrix of any GraphBLAS matrix type
    %---------------------------------------------------------------------------

    function C = eye (varargin)
    %GB.EYE Sparse identity matrix, of any type supported by GraphBLAS
    % C = gb.eye (n) creates a sparse n-by-n identity matrix of type 'double'.
    %
    % C = gb.eye (m,n) or gb.eye ([m n]) is an m-by-n identity matrix.
    %
    % C = gb.eye (m,n,type) or gb.eye ([m n],type) creates a sparse m-by-n
    % identity matrix C of the given GraphBLAS type, either 'double', 'single',
    % 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    % 'uint32', 'uint64', or (in the future) 'complex'.
    %
    % See also spones, spdiags, speye, gb.speye, gb.

    % get the type
    type = 'double' ;
    nargs = nargin ;
    if (nargs > 1 & ischar (varargin {nargs}))
        type = varargin {nargs} ;
        nargs = nargs - 1 ;
    end

    % get the size
    if (nargs == 0)
        m = 1 ;
        n = 1 ;
    elseif (nargs == 1)
        % C = gb.eye (n) or gb.eye ([m n])
        arg1 = varargin {1} ;
        if (length (arg1) == 1)
            % C = gb.eye (n)
            m = arg1 ;
            n = m ;
        elseif (length (arg1) == 2)
            % C = gb.eye ([m n])
            m = arg1 (1) ;
            n = arg1 (2) ;
        else
            error ('only 2D arrays supported') ;
        end
    elseif (nargs == 2)
        % C = gb.eye (m,n)
        m = varargin {1} ;
        n = varargin {2} ;
    else
        error ('incorrect usage; only 2D arrays supported') ;
    end

    % construct the m-by-n identity matrix of the given type
    m = max (m, 0) ;
    n = max (n, 0) ;
    mn = min (m,n) ;
    I = int64 (0:mn-1) ;
    C = gb.build (I, I, ones (mn, 1, type), m, n, '1st', type) ;
    end

    %---------------------------------------------------------------------------
    % gb.speeye: same as gb.eye, just another name
    %---------------------------------------------------------------------------

    function C = speye (varargin)
    %GB.SPEYE Sparse identity matrix, of any type supported by GraphBLAS
    % Identical to gb.eye; see 'help gb.eye' for details.
    C = gb.eye (varargin {:}) ;
    end

    %---------------------------------------------------------------------------
    % gb.build: build a GraphBLAS sparse matrix from a list of entries
    %---------------------------------------------------------------------------

    function G = build (varargin)
    %GB.BUILD construct a GraphBLAS sparse matrix from a list of entries.
    %
    % Usage
    %
    %   G = gb.build (I, J, X, m, n, dup, type, desc)
    %
    % gb.build constructs an m-by-n GraphBLAS sparse matrix from a list of
    % entries, analogous to A = sparse (I, J, X, m, n) to construct a MATLAB
    % sparse matrix A.
    %
    % If not present or empty, m defaults to the largest row index in the list
    % I, and n defaults to the largest column index in the list J.  dup
    % defaults to '+', which gives the same behavior as the MATLAB sparse
    % function: duplicate entries are added together.
    %
    % dup is a string that defines a binary function; see 'help gb.binopinfo'
    % for a list of available binary operators.  The dup operator need not be
    % associative.  If two entries in [I,J,X] have the same row and column
    % index, the dup operator is applied to assemble them into a single entry.
    % Suppose (i,j,x1), (i,j,x2), and (i,j,x3) appear in that order in [I,J,X],
    % in any location (the arrays [I J] need not be sorted, and so these
    % entries need not be adjacent).  That is, i = I(k1) = I(k2) = I(k3) and j
    % = J(k1) = J(k2) = J(k3) for some k1 < k2 < k3.  Then G(i,j) is computed
    % as follows, in order:
    %
    %   x = X (k1) ;
    %   x = dup (x, X (k2)) ;
    %   x = dup (x, X (k3)) ;
    %   G (i,j) = x ;
    %
    % For example, if the dup operator is '1st', then G(i,j)=X(k1) is set, and
    % the subsequent entries are ignored.  If dup is '2nd', then G(i,j)=X(k3),
    % and the preceding entries are ignored.
    %
    % type is a string that defines the type of G (see 'help gb' for a list
    % of types).  The type need not be the same type as the dup operator
    % (unless one has a type of 'complex', in which case both must be
    % 'complex').  If the type is not specified, it defaults to the type of X.
    %
    % The integer arrays I and J may be double, in which case they contain
    % 1-based indices, in the range 1 to the dimension of the matrix.  This is
    % the same behavior as the MATLAB sparse function.  They may instead be
    % int64 or uint64 arrays, in which case they are treated as 0-based.
    % Entries in I are the range 0 to m-1, and J are in the range 0 to n-1.  If
    % I, J, and X are double, the following examples construct the same MATLAB
    % sparse matrix S:
    %
    %   S = sparse (I, J, X) ;
    %   S = gb.build (I, J, X, struct ('kind', 'sparse')) ;
    %   S = double (gb.build (I, J, X)) ;
    %   S = double (gb.build (uint64(I)-1, uint64(J)-1, X)) ;
    %
    % Using uint64 integers for I and J is faster and uses less memory.  I and
    % J need not be in any particular order, but gb.build is fastest if I and J
    % are provided in column-major order.
    %
    % Note: S = sparse (I,J,X) allows either I or J, and X to be scalars.  This
    % feature is not supported in gb.build.  All three arrays must be the same
    % size.
    %
    % See also sparse, find, gb.extracttuples.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        G = gb (gbbuild (args {:})) ;
    else
        G = gbbuild (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.extracttuples: extract all entries from a matrix
    %---------------------------------------------------------------------------

    function [I,J,X] = extracttuples (varargin)
    %GB.EXTRACTTUPLES extract a list of entries from a matrix
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
    % output for I and J.  The default is one-based, to be more compatible with
    % MATLAB.  If one-based, then I and J are returned as MATLAB double column
    % vectors, containing 1-based indices.  The indices in I are in the range 1
    % to m, and the indices in J are in the range 1 to n, if A is m-by-n.  This
    % usage is identical to [I,J,X] = find (A) for a MATLAB sparse or dense
    % matrix.  The array X has the same type as A (double, single, int8, ...,
    % uint8, or (in the future) complex).
    %
    % The default is 'one based', but 'zero based' is faster and uses less
    % memory.  In this case, I and J are returned as int64 arrays.  Entries in
    % I are in the range 0 to m-1, and entries in J are in the range 0 to n-1.
    %
    % This function corresponds to the GrB_*_extractTuples_* functions in
    % GraphBLAS.
    %
    % See also find, gb.build.

    [args, ~] = get_args (varargin {:}) ;
    if (nargout == 3)
        [I, J, X] = gbextracttuples (args {:}) ;
    elseif (nargout == 2)
        [I, J] = gbextracttuples (args {:}) ;
    else
        I = gbextracttuples (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.mxm: sparse matrix-matrix multiply
    %---------------------------------------------------------------------------

    function Cout = mxm (varargin)
    %GB.MXM sparse matrix-matrix multiplication
    %
    % gb.mxm computes C<M> = accum (C, A*B) using a given semiring.
    %
    % Usage:
    %
    %   Cout = gb.mxm (semiring, A, B)
    %   Cout = gb.mxm (semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, accum, semiring, A, B)
    %   Cout = gb.mxm (Cin, accum, semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, M, semiring, A, B)
    %   Cout = gb.mxm (Cin, M, semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, M, accum, semiring, A, B)
    %   Cout = gb.mxm (Cin, M, accum, semiring, A, B, desc)
    %
    % Not all inputs are required.
    %
    % Cin is an optional input matrix.  If Cin is not present or is an empty
    % matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of the
    % right size (which depends on A, B, and the descriptor).  Its type is the
    % output type of the accum operator, if it is present; otherwise, its type
    % is the type of the additive monoid of the semiring.
    %
    % M is the optional mask matrix.  If not present, or if empty, then no mask
    % is used.  If present, M must have the same size as C.
    %
    % If accum is not present, then the operation becomes C<...> = A*B.
    % Otherwise, accum (C,A*B) is computed.  The accum operator acts like a
    % sparse matrix addition (see gb.eadd).
    %
    % The semiring is a required string defining the semiring to use, in the
    % form 'add.mult.type', where '.type' is optional.  For example,
    % '+.*.double' is the conventional semiring for numerical linear algebra,
    % used in MATLAB for C=A*B when A and B are double.  If A or B are complex,
    % then the '+.*.complex' semiring is used (once complex matrice are
    % supported).  GraphBLAS has many more semirings it can use.  See 'help
    % gb.semiringinfo' for more details.
    %
    % A and B are the input matrices.  They are transposed on input if
    % desc.in0 = 'transpose' (which transposes A), and/or
    % desc.in1 = 'transpose' (which transposes B).
    %
    % The descriptor desc is optional.  If not present, all default settings are
    % used.  Fields not present are treated as their default values.  See
    % 'help gb.descriptorinfo' for more details.
    %
    % The input matrices Cin, M, A, and B can be MATLAB matrices or GraphBLAS
    % objects, in any combination.
    %
    % Examples:
    %
    %   A = sprand (4,5,0.5) ;
    %   B = sprand (5,3,0.5) ;
    %   C = gb.mxm ('+.*', A, B) ;
    %   norm (sparse(C)-A*B,1)
    %   E = sprand (4,3,0.7) ;
    %   M = logical (sprand (4,3,0.5)) ;
    %   C2 = gb.mxm (E, M, '+', '+.*', A, B) ;
    %   C3 = E ; AB = A*B ; C3 (M) = C3 (M) + AB (M) ;
    %   norm (sparse(C2)-C3,1)
    %
    %
    % See also gb.descriptorinfo, gb.add, mtimes.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbmxm (args {:})) ;
    else
        Cout = gbmxm (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.select: select entries from a GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function Cout = select (varargin)
    %GB.SELECT: select entries from a GraphBLAS sparse matrix
    %
    % Usage:
    %
    %   Cout = gb.select (selectop, A)
    %   Cout = gb.select (selectop, A, thunk)
    %   Cout = gb.select (selectop, A, thunk, desc)
    %
    %   Cout = gb.select (Cin, accum, selectop, A)
    %   Cout = gb.select (Cin, accum, selectop, A, thunk)
    %   Cout = gb.select (Cin, accum, selectop, A, thunk, desc)
    %
    %   Cout = gb.select (Cin, M, selectop, A)
    %   Cout = gb.select (Cin, M, selectop, A, thunk)
    %   Cout = gb.select (Cin, M, selectop, A, thunk, desc)
    %
    %   Cout = gb.select (Cin, M, accum, selectop, A)
    %   Cout = gb.select (Cin, M, accum, selectop, A, thunk)
    %   Cout = gb.select (Cin, M, accum, selectop, A, thunk, desc)
    %
    % gb.select selects a subset of entries from the matrix A, based on their
    % value or position.  For example, L = gb.select ('tril', A) returns the
    % lower triangular part of the GraphBLAS or MATLAB matrix A, just like L =
    % tril (A) for a MATLAB matrix A.  The select operators can also depend on
    % the values of the entries.  The thunk parameter is an optional input
    % scalar, used in many of the select operators.  For example, L = gb.select
    % ('tril', A, -1) is the same as L = tril (A, -1), which returns the
    % strictly lower triangular part of A.
    %
    % The selectop is a string defining the operator:
    %
    %   operator    MATLAB equivalent                   alternative strings
    %   --------    -----------------                   -------------------
    %   'tril'      C = tril (A,thunk)                  none
    %   'triu'      C = triu (A,thunk)                  none
    %   'diag'      C = diag (A,thunk), see note below  none
    %   'offdiag'   C = entries not in diag(A,k)        none
    %   'nonzero'   C = A (A ~= 0)                      '~=0'
    %   'eqzero'    C = A (A == 0)                      '==0'
    %   'gtzero'    C = A (A >  0)                      '>0'
    %   'gezero'    C = A (A >= 0)                      '>=0'
    %   'ltzero'    C = A (A <= 0)                      '<0'
    %   'lezero'    C = A (A <= 0)                      '<=0'
    %   'nethunk'   C = A (A ~= thunk)                  '~=thunk'
    %   'eqthunk'   C = A (A == thunk)                  '==thunk'
    %   'gtthunk'   C = A (A >  thunk)                  '>thunk'
    %   'gethunk'   C = A (A >= thunk)                  '>=thunk'
    %   'ltthunk'   C = A (A <= thunk)                  '<thunk'
    %   'lethunk'   C = A (A >= thunk)                  '<=thunk'
    %
    % Note that C = gb.select ('diag',A) does not returns a vector, but a
    % diagonal matrix.
    %
    % Many of the operations have equivalent synonyms, as listed above.
    %
    % Cin is an optional input matrix.  If Cin is not present or is an empty
    % matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of the
    % right size (which depends on A, and the descriptor).  Its type is the
    % output type of the accum operator, if it is present; otherwise, its type
    % is the type of the matrix A.
    %
    % M is the optional mask matrix.  If not present, or if empty, then no mask
    % is used.  If present, M must have the same size as C.
    %
    % If accum is not present, then the operation becomes C<...> = select(...).
    % Otherwise, accum (C, select(...)) is computed.  The accum operator acts
    % like a sparse matrix addition (see gb.eadd).
    %
    % The selectop is a required string defining the select operator to use.
    % All operators operate on all types (the select operators do not do any
    % typecasting of its inputs).
    %
    % A is the input matrix.  It is transposed on input if desc.in0 =
    % 'transpose'.
    %
    % The descriptor desc is optional.  If not present, all default settings are
    % used.  Fields not present are treated as their default values.  See
    % 'help gb.descriptorinfo' for more details.
    %
    % The input matrices Cin, M, and A, can be MATLAB matrices or GraphBLAS
    % objects, in any combination.
    %
    % See also tril, triu, diag.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbselect (args {:})) ;
    else
        Cout = gbselect (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.assign: sparse matrix assignment
    %---------------------------------------------------------------------------

    function Cout = assign (varargin)
    %GB.ASSIGN: assign a submatrix into a matrix
    %
    % gb.assign is an interface to GrB_Matrix_assign and
    % GrB_Matrix_assign_[TYPE], computing the GraphBLAS expression:
    %
    %   C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')
    %
    % where A can be a matrix or a scalar.
    %
    % Usage:
    %
    %   Cout = gb.assign (Cin, M, accum, A, I, J, desc)
    %
    % Cin and A are required parameters.  All others are optional.
    %
    % desc: see 'help gb.descriptorinfo' for details.
    %
    % I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
    %
    %       0:   { }    This is the MATLAB ':', like C(:,J), refering to all m
    %                   rows, if C is m-by-n.
    %
    %       1:   { I }  1D list of row indices, like C(I,J) in MATLAB.
    %                   If I is double, then it contains 1-based indices, in
    %                   the range 1 to m if C is m-by-n, so that C(1,1) refers
    %                   to the entry in the first row and column of C.  If I is
    %                   int64 or uint64, then it contains 0-based indices in
    %                   the range 0 to m-1, where C(0,0) is the same entry.
    %
    %       2:  { start,fini }  start and fini are scalars (either double,
    %                   int64, or uint64).  This defines I = start:fini in
    %                   MATLAB index notation.  Typically, start and fini have
    %                   type double and refer to 1-based indexing of C.  int64
    %                   or uint64 scalars are treated as 0-based.
    %
    %       3:  { start,inc,fini } start, inc, and fini are scalars (double,
    %                   int64, or uint64).  This defines I = start:inc:fini in
    %                   MATLAB notation.  The start and fini are 1-based if
    %                   double, 0-based if int64 or uint64.  inc is the same
    %                   for any type.
    %
    %       The J argument is identical, except that it is a list of column
    %       indices of C.  If only one cell array is provided, J = {  } is
    %       implied, refering to all n columns of C, like C(I,:) in MATLAB
    %       notation.  1D indexing of a matrix C, as in C(I) = A, is not yet
    %       supported.
    %
    %       If neither I nor J are provided on input, then this implies
    %       both I = { } and J = { }, or C(:,:) in MATLAB notation,
    %       refering to all rows and columns of C.
    %
    %  A: this argument either has size length(I)-by-length(J) (or A' if d.in0
    %       is 'transpose'), or it is 1-by-1 for scalar assignment (like
    %       C(1:2,1:2)=pi, which assigns the scalar pi to the leading 2-by-2
    %       submatrix of C).  For scalar assignment, A must contain an entry;
    %       it cannot be empty (for example, the MATLAB A = sparse (0)).
    %
    % accum: an optional binary operator, defined by a string ('+.double') for
    %       example.  This allows for C(I,J) = C(I,J) + A to be computed.  If
    %       not present, no accumulator is used and C(I,J)=A is computed.
    %
    % M: an optional mask matrix, the same size as C.
    %
    % Cin: a required input matrix, containing the initial content of the
    % matrix C.  Cout is the content of C after the assignment is made.
    %
    % Example:
    %
    %   A = sprand (5, 4, 0.5)
    %   AT = A'
    %   M = sparse (rand (4, 5)) > 0.5
    %   Cin = sprand (4, 5, 0.5)
    %
    %   d.in0 = 'transpose'
    %   d.mask = 'complement'
    %   Cout = gb.assign (Cin, M, A, d)
    %   C2 = Cin
    %   C2 (~M) = AT (~M)
    %   C2 - sparse (Cout)
    %
    %   I = [2 1 5]
    %   J = [3 3 1 2]
    %   B = sprandn (length (I), length (J), 0.5)
    %   Cin = sprand (6, 3, 0.5)
    %   Cout = gb.assign (Cin, B, {I}, {J})
    %   C2 = Cin
    %   C2 (I,J) = B
    %   C2 - sparse (Cout)
    %
    % See also gb.subassign, subsasgn

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbassign (args {:})) ;
    else
        Cout = gbassign (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.subassign: sparse matrix assignment
    %---------------------------------------------------------------------------

    function Cout = subassign (varargin)
    %GB.SUBASSIGN: assign a submatrix into a matrix
    %
    % gb.subassign is an interface to GxB_Matrix_subassign and
    % GxB_Matrix_subassign_[TYPE], computing the GraphBLAS expression:
    %
    %   C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')
    %
    % where A can be a matrix or a scalar.
    %
    % Usage:
    %
    %   Cout = gb.subassign (Cin, M, accum, A, I, J, desc)
    %
    % gb.subassign is identical to gb.assign, with two key differences:
    %
    %   (1) The mask is different.
    %       With gb.subassign, the mask M is length(I)-by-length(J),
    %       and M(i,j) controls how A(i,j) is assigned into C(I(i),J(j)).
    %       With gb.assign, the mask M has the same size as C,
    %       and M(i,j) controls how C(i,j) is assigned.
    %   (2) The d.out = 'replace' option differs.  gb.assign can clear
    %       entries outside the C(I,J) submatrix; gb.subassign cannot.
    %
    % If there is no mask, or if I and J are ':', then the two methods are
    % identical.  The examples shown in 'help gb.assign' also work with
    % gb.subassign.  Otherwise, gb.subassign is faster.  The two methods are
    % described below, where '+' is the optional accum operator.
    %
    %   step  | gb.assign       gb.subassign
    %   ----  | ---------       ------------
    %   1     | S = C(I,J)      S = C(I,J)
    %   2     | S = S + A       S<M> = S + A
    %   3     | Z = C           C(I,J) = S
    %   4     | Z(I,J) = S
    %   5     | C<M> = Z
    %
    % Refer to gb.assign for a description of the other input/outputs.
    %
    % See also gb.assign, subsasgn.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbsubassign (args {:})) ;
    else
        Cout = gbsubassign (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.vreduce: reduce a matrix to a vector
    %---------------------------------------------------------------------------

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
    end

    %---------------------------------------------------------------------------
    % gb.reduce: reduce a matrix to a scalar
    %---------------------------------------------------------------------------

    function Cout = reduce (varargin)
    %GB.REDUCE reduce a matrix to a scalar
    %
    % Usage:
    %
    %   cout = gb.reduce (monoid, A)
    %   cout = gb.reduce (monoid, A, desc)
    %   cout = gb.reduce (cin, accum, monoid, A)
    %   cout = gb.reduce (cin, accum, monoid, A, desc)
    %
    % gb.reduce reduces a matrix to a scalar, using the given monoid.  The
    % valid monoids are: '+', '*', 'max', and 'min' for all but the 'logical'
    % type, and '|', '&', 'xor', and 'ne' for the 'logical' type.  See 'help
    % gb.monoidinfo' for more details.
    %
    % The monoid and A arguments are required.  All others are optional.  The
    % op is applied to all entries of the matrix A to reduce them to a single
    % scalar result.
    %
    % accum: an optional binary operator (see 'help gb.binopinfo' for a list).
    %
    % cin: an optional input scalar into which the result can be accumulated
    %       with cout = accum (cin, result).
    %
    % See also gb.vreduce; sum, prod, max, min (with the 'all' parameter).

    % FUTURE: add complex monoids.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbreduce (args {:})) ;
    else
        Cout = gbreduce (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.gbkron: Kronecker product
    %---------------------------------------------------------------------------

    function Cout = gbkron (varargin)
    %GB.GBKRON sparse Kronecker product
    %
    % Usage:
    %
    %   Cout = gb.gbkron (op, A, B, desc)
    %   Cout = gb.gbkron (Cin, accum, op, A, B, desc)
    %   Cout = gb.gbkron (Cin, M, op, A, B, desc)
    %   Cout = gb.gbkron (Cin, M, accum, op, A, B, desc)
    %
    % gb.gbkron computes the Kronecker product, T=kron(A,B), using the given
    % binary operator op, in place of the conventional '*' operator for the
    % MATLAB built-in kron.  See also C = kron (A,B), which uses the default
    % semiring operators if A and/or B are gb matrices.
    %
    % T is then accumulated into C via C<#M,replace> = accum (C,T).
    %
    % See also kron, gb/kron.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbkron (args {:})) ;
    else
        Cout = gbkron (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.gbtranspose: transpose a matrix
    %---------------------------------------------------------------------------

    function Cout = gbtranspose (varargin)
    %GB.GBTRANSPOSE transpose a sparse matrix
    %
    % Usage:
    %
    %   Cout = gb.gbtranspose (A, desc)
    %   Cout = gb.gbtranspose (Cin, accum, A, desc)
    %   Cout = gb.gbtranspose (Cin, M, A, desc)
    %   Cout = gb.gbtranspose (Cin, M, accum, A, desc)
    %
    % The descriptor is optional.  If desc.in0 is 'transpose', then C<M>=A or
    % C<M>=accum(C,A) is computed, since the default behavior is to transpose
    % the input matrix.
    %
    % See also transpose, ctranspose.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbtranspose (args {:})) ;
    else
        Cout = gbtranspose (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.eadd: sparse matrix addition
    %---------------------------------------------------------------------------

    function Cout = eadd (varargin)
    %GB.EADD sparse matrix addition
    %
    % Usage:
    %
    %   Cout = gb.eadd (op, A, B, desc)
    %   Cout = gb.eadd (Cin, accum, op, A, B, desc)
    %   Cout = gb.eadd (Cin, M, op, A, B, desc)
    %   Cout = gb.eadd (Cin, M, accum, op, A, B, desc)
    %
    % gb.eadd computes the element-wise 'addition' T=A+B.  The result T has the
    % pattern of the union of A and B. The operator is used where A(i,j) and
    % B(i,j) are present.  Otherwise the entries in A and B are copied directly
    % into T:
    %
    %   if (A(i,j) and B(i,j) is present)
    %       T(i,j) = op (A(i,j), B(i,j))
    %   elseif (A(i,j) is present but B(i,j) is not)
    %       T(i,j) = A(i,j)
    %   elseif (B(i,j) is present but A(i,j) is not)
    %       T(i,j) = B(i,j)
    %
    % T is then accumulated into C via C<#M,replace> = accum (C,T).
    %
    % Cin, M, accum, and the descriptor desc are the same as all other
    % gb.methods; see gb.mxm and gb.descriptorinfo for more details.  For the
    % binary operator, see gb.binopinfo.
    %
    % See also gb.emult.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbeadd (args {:})) ;
    else
        Cout = gbeadd (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.emult: sparse matrix element-wise multiplication
    %---------------------------------------------------------------------------

    function Cout = emult (varargin)
    %GB.EMULT sparse element-wise 'multiplication'
    %
    % Usage:
    %
    %   Cout = gb.emult (op, A, B, desc)
    %   Cout = gb.emult (Cin, accum, op, A, B, desc)
    %   Cout = gb.emult (Cin, M, op, A, B, desc)
    %   Cout = gb.emult (Cin, M, accum, op, A, B, desc)
    %
    % gb.emult computes the element-wise 'multiplication' T=A.*B.  The result T
    % has the pattern of the intersection of A and B. The operator is used
    % where A(i,j) and B(i,j) are present.  Otherwise the entry does not
    % appear in T.
    %
    %   if (A(i,j) and B(i,j) is present)
    %       T(i,j) = op (A(i,j), B(i,j))
    %
    % T is then accumulated into C via C<#M,replace> = accum (C,T).
    %
    % Cin, M, accum, and the descriptor desc are the same as all other
    % gb.methods; see gb.mxm and gb.descriptorinfo for more details.  For the
    % binary operator, see gb.binopinfo.
    %
    % See also gb.eadd.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbemult (args {:})) ;
    else
        Cout = gbemult (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.apply: apply a unary operator to entries in a matrix
    %---------------------------------------------------------------------------

    function Cout = apply (varargin)
    %GB.APPLY apply a unary operator to a sparse matrix
    %
    % Usage:
    %
    %   Cout = gb.apply (op, A, desc)
    %   Cout = gb.apply (Cin, accum, op, A, desc)
    %   Cout = gb.apply (Cin, M, op, A, desc)
    %   Cout = gb.apply (Cin, M, accum, op, A, desc)
    %
    % gb.apply applies a unary operator to the entries in the input matrix A.
    % See 'help gb.unopinfo' for a list of available unary operators.
    %
    % The op and A arguments are required.
    %
    % accum: a binary operator to accumulate the results.
    %
    % Cin, and the mask matrix M, and the accum operator are optional.  If
    % either accum or M is present, then Cin is a required input. If desc.in0
    % is 'transpose' then A is transposed before applying the operator, as
    % C<M> = accum (C, f(A')) where f(...) is the unary operator.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbapply (args {:})) ;
    else
        Cout = gbapply (args {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.extract: extract a submatrix
    %---------------------------------------------------------------------------

    function Cout = extract (varargin)
    %GB.EXTRACT extract sparse submatrix
    %
    % gb.extract is an interface to GrB_Matrix_extract and
    % GrB_Matrix_extract_[TYPE], computing the GraphBLAS expression:
    %
    %   C<#M,replace> = accum (C, A(I,J)) or accum(C, A(J,I)')
    %
    % Usage:
    %
    %   Cout = gb.extract (Cin, M, accum, A, I, J, desc)
    %
    % A is a required parameters.  All others are optional, but if M or accum
    % appears, then Cin is also required.  If desc.in0 is 'transpose', then
    % the description below assumes A = A' is computed first before the
    % extraction (A is not changed on output, however).
    %
    % desc: see 'help gb.descriptorinfo' for details.
    %
    % I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
    %
    %       0:   { }    This is the MATLAB ':', like A(:,J), refering to all m
    %                   rows, if A is m-by-n.
    %
    %       1:   { I }  1D list of row indices, like A(I,J) in MATLAB.
    %                   If I is double, then it contains 1-based indices, in
    %                   the range 1 to m if A is m-by-n, so that A(1,1) refers
    %                   to the entry in the first row and column of A.  If I is
    %                   int64 or uint64, then it contains 0-based indices in
    %                   the range 0 to m-1, where A(0,0) is the same entry.
    %
    %       2:  { start,fini }  start and fini are scalars (either double,
    %                   int64, or uint64).  This defines I = start:fini in
    %                   MATLAB index notation.  Typically, start and fini have
    %                   type double and refer to 1-based indexing of A.  int64
    %                   or uint64 scalars are treated as 0-based.
    %
    %       3:  { start,inc,fini } start, inc, and fini are scalars (double,
    %                   int64, or uint64).  This defines I = start:inc:fini in
    %                   MATLAB notation.  The start and fini are 1-based if
    %                   double, 0-based if int64 or uint64.  inc is the same
    %                   for any type.
    %
    %       The J argument is identical, except that it is a list of column
    %       indices of A.  If only one cell array is provided, J = {  } is
    %       implied, refering to all n columns of A, like A(I,:) in MATLAB
    %       notation.  1D indexing of a matrix A, as in C = A(I), is not yet
    %       supported.
    %
    %       If neither I nor J are provided on input, then this implies
    %       both I = { } and J = { }, or A(:,:) in MATLAB notation,
    %       refering to all rows and columns of A.
    %
    % Cin: an optional input matrix, containing the initial content of the
    %       matrix C.  Cout is the content of C after the assignment is made.
    %       If present, Cin argument has size length(I)-by-length(J).
    %       If accum is present then Cin is a required input.
    %
    % accum: an optional binary operator, defined by a string ('+.double') for
    %       example.  This allows for Cout = Cin + A(I,J) to be computed.  If
    %       not present, no accumulator is used and Cout=A(I,J) is computed.
    %       If accum is present then Cin is a required input.
    %
    % M: an optional mask matrix, the same size as C.
    %
    % Example:
    %
    %   A = sprand (5, 4, 0.5)
    %   I = [2 1 5]
    %   J = [3 3 1 2]
    %   Cout = gb.extract (A, {I}, {J})
    %   C2 = A (I,J)
    %   C2 - Cout
    %
    % See also subsref.

    [args is_gb] = get_args (varargin {:}) ;
    if (is_gb)
        Cout = gb (gbextract (args {:})) ;
    else
        Cout = gbextract (args {:}) ;
    end
    end

end
end

%===============================================================================
% local functions ==============================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % get_args: get the arguments and descriptor
    %---------------------------------------------------------------------------

    function [args is_gb] = get_args (varargin)
    % Get the arguments and the descriptor for a gb.method.  Any input
    % arguments that are GraphBLAS sparse matrix objects are replaced with the
    % struct arg.opaque so that they can be passed to the underlying
    % mexFunction.  Next, the descriptor is modified to change the default
    % d.kind.
    %
    % All mexFunctions in private/mexFunction/*.c require the descriptor to be
    % present as the last argument.  They are not required for the user-
    % accessible gb.methods.  If the descriptor d is not present, then it is
    % created and appended to the argument list, with d.kind = 'gb'.  If the
    % descriptor is present and d.kind does not appear, then d.kind = 'gb' is
    % set.  Finally, is_gb is set true if d.kind is 'gb'.  If d.kind is 'gb',
    % then the underlying mexFunction returns a GraphBLAS struct, which is then
    % converted above to a GraphBLAS object.

    % get the args and extract any GraphBLAS matrix structs
    args = varargin ;
    for k = 1:length (args)
        if (isa (args {k}, 'gb'))
            args {k} = args {k}.opaque ;
        end
    end

    % find the descriptor
    is_gb = true ;
    if (length (args) > 0)
        % get the last input argument and see if it is a GraphBLAS descriptor
        d = args {end} ;
        if (isstruct (d) && ~isfield (d, 'GraphBLAS'))
            % found the descriptor.  If it does not have d.kind, add it.
            if (~isfield (d, 'kind'))
                d.kind = 'gb' ;
                args {end} = d ;
                is_gb = true ;
            else
                is_gb = isequal (d.kind, 'gb') || isequal (d.kind, 'default') ;
            end
        else
            % the descriptor is not present; add it
            args {end+1} = struct ('kind', 'gb') ;
            is_gb = true ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % get_scalar: get the first scalar from a matrix
    %---------------------------------------------------------------------------

    function x = get_scalar (A)
    [~, ~, x] = gb.extracttuples (A) ;
    if (length (x) == 0)
        x = 0 ;
    else
        x = x (1) ;
    end
    end

    %---------------------------------------------------------------------------
    % sparse_comparator: apply a comparator to two matrices
    %---------------------------------------------------------------------------

    function C = sparse_comparator (op, A, B)
    % The pattern of C is the set union of A and B.  A and B must first be
    % expanded to include explicit zeros in the set union of A and B.  For
    % example, with A < B for two matrices A and B:
    %
    %     in A        in B        A(i,j) < B (i,j)    true or false
    %     not in A    in B        0 < B(i,j)          true or false
    %     in A        not in B    A(i,j) < 0          true or false
    %     not in A    not in B    0 < 0               false, not in C
    %
    % expand A and B to the set union of A and B, with explicit zeros.
    % The type of the '1st' operator is the type of the first argument of
    % gbeadd, so the 2nd argument can be boolean to save space.
    A0 = gb.eadd ('1st', A, gb.expand (false, B)) ;
    B0 = gb.eadd ('1st', B, gb.expand (false, A)) ;
    C = gb.select ('nonzero', gb.eadd (op, A0, B0)) ;
    end

    %---------------------------------------------------------------------------
    % dense_comparator: apply a comparator to two matrices
    %---------------------------------------------------------------------------

    function C = dense_comparator (op, A, B)
    % The pattern of C is a full matrix.  A and B must first be expanded to to
    % a full matrix with explicit zeros.  For example, with A <= B for two
    % matrices A and B:
    %
    %     in A        in B        A(i,j) <= B (i,j)    true or false
    %     not in A    in B        0 <= B(i,j)          true or false
    %     in A        not in B    A(i,j) <= 0          true or false
    %     not in A    not in B    0 <= 0               true, in C
    %
    % expand A and B to full matrices with explicit zeros
    C = gb.select ('nonzero', gb.emult (op, full (A), full (B))) ;
    end

    %---------------------------------------------------------------------------
    % col_degree: count the number of entries in each column of G
    %---------------------------------------------------------------------------

    function D = col_degree (G)
    % D(j) = # of entries in G(:,j); result is a column vector
    D = gb.vreduce ('+.int64', spones (G), struct ('in0', 'transpose')) ;
    end

    %---------------------------------------------------------------------------
    % row_degree: count the number of entries in each row of G
    %---------------------------------------------------------------------------

    function D = row_degree (G)
    % D(i) = # of entries in G(i,:); result is a column vector
    D = gb.vreduce ('+.int64', spones (G)) ;
    end

    %---------------------------------------------------------------------------
    % compute_mpower: compute A^b
    %---------------------------------------------------------------------------

    function C = compute_mpower (A, b)
    % C = A^b where b > 0 is an integer
    if (b == 1)
        C = A ;
    else
        T = compute_mpower (A, floor (b/2)) ;
        C = T*T ;
        clear T ;
        if (mod (b, 2) == 1)
            C = C*A ;
        end
    end
    end

    %---------------------------------------------------------------------------
    % get_index: helper function for subsref and subsasgn
    %---------------------------------------------------------------------------

    function [I, whole] = get_index (I_input)
    whole = isequal (I_input, {':'}) ;
    if (whole)
        % C (:)
        I = { } ;
    elseif (iscell (I_input {1}))
        % C ({ }), C ({ list }), C ({start,fini}), or C ({start,inc,fini}).
        I = I_input {1} ;
    else
        % C (I) for an explicit list I, or MATLAB colon notation
        I = I_input ;
    end
    end

