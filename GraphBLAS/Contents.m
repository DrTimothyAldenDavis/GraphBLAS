% MATLAB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Its MATLAB interface provides faster
% sparse matrix operations than the built-in methods in MATLAB, as well as
% sparse integer and single-precision matrices, and operations with arbitrary
% semirings.  Below is a short summary; see 'help gb' for details.
%
% Methods:
%
%   G = gb (S)              construct a gb matrix
%   S = sparse (G)          makes a copy of a gb matrix
%   F = full (G)            adds explicit zeros to a gb matrix
%   F = full (G,id)         adds explicit identity values to a gb matrix
%   S = double (G)          cast gb matrix to MATLAB sparse double matrix
%   C = logical (G)         cast gb matrix to MATLAB sparse logical matrix
%   C = complex (G)         cast gb matrix to MATLAB sparse complex (FUTURE)
%   C = single (G)          cast gb matrix to MATLAB full single matrix
%   C = int8 (G)            cast gb matrix to MATLAB full int8 matrix
%   C = int16 (G)           cast gb matrix to MATLAB full int16 matrix
%   C = int32 (G)           cast gb matrix to MATLAB full int32 matrix
%   C = int64 (G)           cast gb matrix to MATLAB full int64 matrix
%   C = uint8 (G)           cast gb matrix to MATLAB full uint8 matrix
%   C = uint16 (G)          cast gb matrix to MATLAB full uint16 matrix
%   C = uint32 (G)          cast gb matrix to MATLAB full uint32 matrix
%   C = uint64 (G)          cast gb matrix to MATLAB full uint64 matrix
%   C = cast (G,...)        cast gb matrix to MATLAB matrix (as above)
%   X = nonzeros (G)        extract all entries from a gb matrix
%   [I,J,X] = find (G)      extract all entries from a gb matrix
%   C = spones (G)          return pattern of gb matrix
%   disp (G, level)         display a gb matrix G
%   display (G)             display a gb matrix G; same as disp(G,2)
%   mn = numel (G)          m*n for an m-by-n gb matrix G
%   e = nnz (G)             number of entries in a gb matrix G
%   e = nzmax (G)           number of entries in a gb matrix G
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
%   s = isa (G, classname)  determine if a gb matrix is of a specific class
%   C = diag (G,k)          diagonal matrices and diagonals of a gb matrix G
%   L = tril (G,k)          lower triangular part of gb matrix G
%   U = triu (G,k)          upper triangular part of gb matrix G
%   C = kron (A,B)          Kronecker product
%   C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%   C = reshape (G, ...)    reshape a GraphBLAS matrix
%   C = abs (G)             absolute value
%   C = sign (G)            signum function
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
%   C = spfun (fun, G)      evaluate a function on the entries of G
%   p = amd (G)             approximate minimum degree ordering
%   p = colamd (G)          column approximate minimum degree ordering
%   p = symamd (G)          approximate minimum degree ordering
%   p = symrcm (G)          reverse Cuthill-McKee ordering
%   [...] = dmperm (G)      Dulmage-Mendelsohn permutation
%   C = conj (G)            complex conjugate
%   C = real (G)            real part of a complex GraphBLAS matrix
%   [V, ...] = eig (G,...)  eigenvalues and eigenvectors
%
% Operator overloading:
%
%     A+B    A-B   A*B    A.*B   A./B   A.\B   A.^b    A/b    C=A(I,J)
%     -A     +A    ~A     A'     A.'    A&B    A|B     b\A    C(I,J)=A
%     A~=B   A>B   A==B   A<=B   A>=B   A<B    [A,B]   [A;B]  
%     A(1:end,1:end)
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
%   C = gb.subassign (A, I, J)       C(I,J)<M> = A ; M and A same size
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

