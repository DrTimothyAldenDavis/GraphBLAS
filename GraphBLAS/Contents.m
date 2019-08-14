% MATLAB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Its MATLAB interface provides faster
% sparse matrix operations than the built-in methods in MATLAB, as well as
% sparse integer and single-precision matrices, and operations with arbitrary
% semirings.  Below is a short summary; see 'help gb' for details, for more
% help on the 'gb' class.
%
% Methods:
%
%   G = gb (S)              construct a gb matrix.
%   S = sparse (G)          convert a gb matrix G to a MATLAB sparse matrix
%   [I,J,X] = find (G)      extract all entries from a gb matrix
%   X = nonzeros (G)        extract all entries from a gb matrix
%   F = full (G)            convert a gb matrix G to a MATLAB dense matrix
%   C = double (G)          typecast a gb matrix G to double gb matrix C
%   C = single (G)          typecast a gb matrix G to single gb matrix C
%   C = complex (G)         FUTURE: typecast a gb matrix G to complex
%   C = logical (G)         typecast a gb matrix G to logical gb matrix C
%   C = int8 (G)            typecast a gb matrix G to int8 gb matrix C
%   C = int16 (G)           typecast a gb matrix G to int16 gb matrix C
%   C = int32 (G)           typecast a gb matrix G to int32 gb matrix C
%   C = int64 (G)           typecast a gb matrix G to int64 gb matrix C
%   C = uint8 (G)           typecast a gb matrix G to uint8 gb matrix C
%   C = uint16 (G)          typecast a gb matrix G to uint16 gb matrix C
%   C = uint32 (G)          typecast a gb matrix G to uint32 gb matrix C
%   C = uint64 (G)          typecast a gb matrix G to uint64 gb matrix C
%   C = cast (G,...)        typecast a gb matrix G to any of the above
%   C = spones (G)          return pattern of gb matrix
%   disp (G, level)         display a gb matrix G
%   display (G)             display a gb matrix G; same as disp(G,2)
%   mn = numel (G)          m*n for an m-by-n gb matrix G
%   e = nnz (G)             number of entries in a gb matrix G
%   [m n] = size (G)        size of a gb matrix G
%   n = length (G)          length of a gb vector
%   s = isempty (G)         true if any dimension of G is zero
%   s = issparse (G)        true for any gb matrix G
%   s = ismatrix (G)        true for any gb matrix G
%   s = isvector (G)        true if m=1 or n=1, for an m-by-n gb matrix G
%   s = isscalar (G)        true if G is a 1-by-1 gb matrix
%   s = isnumeric (G)       true for any gb matrix G
%   s = isfloat (G)         true if gb matrix is double, single, or complex
%   s = isreal (G)          true if gb matrix is not complex
%   s = isinteger (G)       true if gb matrix is int8, int16, ..., uint64
%   s = islogical (G)       true if gb matrix is logical
%   C = diag (G,k)          diagonal matrices and diagonals of a gb matrix G
%   L = tril (G,k)          lower triangular part of gb matrix G
%   U = triu (G,k)          upper triangular part of gb matrix G
%   C = kron (A,B)          Kronecker product
%   C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%   C = abs (G)             absolute value
%   s = istril (G)          true if G is lower triangular
%   s = istriu (G)          true if G is upper triangular
%   s = isbanded (G,...)    true if G is banded
%   s = isdiag (G)          true if G is diagonal
%   [lo,hi] = bandwidth (G) determine the lower & upper bandwidth of G
%   C = sqrt (G)            element-wise square root
%   C = sum (G, option)     reduce via sum, to vector or scalar
%   C = prod (G, option)    reduce via product, to vector or scalar
%   s = norm (G, kind)      1-norm or inf-norm of a gb matrix
%   C = max (G, ...)        reduce via max, to vector or scalar
%   C = min (G, ...)        reduce via min, to vector or scalar
%   C = any (G, ...)        reduce via '|', to vector or scalar
%   C = all (G, ...)        reduce via '&', to vector or scalar
%   C = eps (G)             floating-point spacing
%   C = ceil (G)            round towards infinity
%   C = floor (G)           round towards -infinity
%   C = round (G)           round towards nearest
%   C = fix (G)             round towards zero
%   C = isfinite (G)        test if finite
%   C = isinf (G)           test if infinite
%   C = isnan (G)           test if NaN
%
% Operator overloading:
%
%   C = plus (A, B)         C = A + B
%   C = minus (A, B)        C = A - B
%   C = uminus (A)          C = -A
%   C = uplus (A)           C = +A
%   C = times (A, B)        C = A .* B
%   C = mtimes (A, B)       C = A * B
%   C = rdivide (A, B)      C = A ./ B
%   C = ldivide (A, B)      C = A .\ B
%   C = mrdivide (A, B)     C = A / B
%   C = mldivide (A, B)     C = A \ B
%   C = power (A, B)        C = A .^ B
%   C = mpower (A, B)       C = A ^ B
%   C = lt (A, B)           C = A < B
%   C = gt (A, B)           C = A > B
%   C = le (A, B)           C = A <= B
%   C = ge (A, B)           C = A >= B
%   C = ne (A, B)           C = A ~= B
%   C = eq (A, B)           C = A == B
%   C = and (A, B)          C = A & B
%   C = or (A, B)           C = A | B
%   C = not (A)             C = ~A
%   C = ctranspose (A)      C = A'
%   C = transpose (A)       C = A.'
%   C = horzcat (A, B)      C = [A , B]
%   C = vertcat (A, B)      C = [A ; B]
%   C = subsref (A, I, J)   C = A (I,J)
%   C = subsasgn (A, I, J)  C (I,J) = A
%
% Static Methods:
%
%   gb.clear                    clear GraphBLAS workspace and settings
%   gb.descriptorinfo (d)       list properties of a descriptor
%   gb.unopinfo (op, type)      list properties of a unary operator
%   gb.binopinfo (op, type)     list properties of a binary operator
%   gb.monoidinfo (op, type)    list properties of a monoid
%   gb.semiringinfo (s, type)   list properties of a semiring
%   t = gb.threads (t)          set/get # of threads to use in GraphBLAS
%   c = gb.chunk (c)            set/get chunk size to use in GraphBLAS
%   e = gb.nvals (A)            number of entries in a matrix
%   G = gb.empty (m, n)         return an empty GraphBLAS matrix
%   s = gb.type (X)             get the type of a MATLAB or gb matrix X
%   f = gb.format (f)           set/get matrix format to use in GraphBLAS
%   C = expand (scalar, S)      expand a scalar (C = scalar*spones(S))
%   G = gb.build (...)          build a gb matrix from a list of entries
%   [I,J,X] = gb.extracttuples (A) extract all entries from a matrix
%
%   Static Methods with an optional mask M and/or operator accum, of the form:
%   C = gb.method (Cin, M, accum, ..., desc) where '...' is listed below.
%
%   C = gb.mxm (semiring, A, B)      C = A*B over a semiring
%   C = gb.select (op, A, thunk)     select subset of entries
%   C = gb.assign (A, I, J)          C<M>(I,J) = A ; M and C same size
%   C = gb.subassign (A, I, J)       C(I,J)<M> = A ; M and C(I,J) same size
%   C = gb.colassign (u, I, j)       C<M>(I,j) = u
%   C = gb.rowassign (u, i, J)       C<M'>(i,J) = u'
%   C = gb.vreduce (op, A)           reduce to a vector
%   C = gb.reduce (op, A)            reduce to a scalar (no mask M)
%   C = gb.gbkron (op, A, B)         Kronecker product
%   C = gb.gbtranspose (A)           transpose
%   C = gb.eadd (op, A, B)           element-wise addition
%   C = gb.emult (op, A, B)          element-wise multiplication
%   C = gb.apply (op, A)             apply a unary operator
%   C = gb.extract (A, I, J)         C = A (I,J)
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis/GraphBLAS

