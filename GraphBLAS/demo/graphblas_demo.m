%% GraphBLAS: graph algorithms in the language of linear algebra
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Visit http://graphblas.org for more
% details and resources.  See also the SuiteSparse:GraphBLAS User Guide in this
% package.
%
% SuiteSparse:GraphBLAS, (c) 2017-2019, Tim Davis, Texas A&M University,
% http://faculty.cse.tamu.edu/davis
%
% See also sparse, doc sparse, and https://twitter.com/DocSparse

%% GraphBLAS provides faster and more general sparse matrices for MATLAB
% GraphBLAS is not only useful for creating graph algorithms; it also supports
% a wide range of sparse matrix data types and operations.  MATLAB can compute
% C=A*B with just two semirings: 'plus.times.double' and 'plus.times.complex'
% for complex matrices.  GraphBLAS has 1,040 unique built-in semirings, such
% as 'max.plus' (https://en.wikipedia.org/wiki/Tropical_semiring).  These
% semirings can be used to construct a wide variety of graph algorithms, based
% on operations on sparse adjacency matrices.
%
% GraphBLAS supports sparse double and single precision matrices, logical, and
% sparse integer matrices: int8, int16, int32, int64, uint8, uint16, uint32,
% and uint64.  Complex matrices will be added in the future.

clear all
format compact
rng ('default') ;
X = 100 * rand (2)
G = gb (X)              % GraphBLAS copy of a matrix X, same type

%% Sparse integer matrices
% Here's an int8 version of the same matrix:

S = sparse (G)          % convert a gb matrix to a MATLAB sparse matrix
G = gb (X, 'int8')      % GraphBLAS typecasted copy of matrix X

%% Sparse single-precision matrices
% Matrix operations in GraphBLAS are typically as fast, or faster than MATLAB.
% Here's an unfair comparison: computing X^2 with MATLAB in double precision
% and with GraphBLAS in single precision.  You would naturally expect GraphBLAS
% to be faster. 
%
% Please wait ...

n = 1e5 
X = spdiags (rand (n, 201), -100:100, n, n) ;
G = gb (X, 'single') ;
tic
G2 = G^2 ;
gb_time = toc ;
tic
X2 = X^2 ;
matlab_time = toc ;
fprintf ('\nGraphBLAS time: %g sec (in single precision)\n', gb_time) ;
fprintf ('MATLAB time:    %g sec (in double precision)\n', matlab_time) ;
fprintf ('Speedup of GraphBLAS over MATLAB: %g\n', matlab_time / gb_time) ;

%% Mixing MATLAB and GraphBLAS matrices
% The error in the last computation is about eps('single') since GraphBLAS did
% its computation in single precision, while MATLAB used double precision.
% MATLAB and GraphBLAS matrices can be easily combined, as in X2-G2.  The
% sparse single precision matrices take less memory space.

err = norm (X2 - G2, 1) / norm (X2,1)
eps ('single')
whos

%% Faster matrix operations
% But even with standard double precision sparse matrices, GraphBLAS is
% typically faster than the built-in MATLAB methods.  Here's a fair comparison:

G = gb (X) ;
tic
G2 = G^2 ;
gb_time = toc ;
err = norm (X2 - G2, 1) / norm (X2,1)
fprintf ('\nGraphBLAS time: %g sec (in double precision)\n', gb_time) ;
fprintf ('MATLAB time:    %g sec (in double precision)\n', matlab_time) ;
fprintf ('Speedup of GraphBLAS over MATLAB: %g\n', matlab_time / gb_time) ;

%% A wide range of semirings
% MATLAB can only compute C=A*B using the standard '+.*.double' and
% '+.*.complex' semirings.  A semiring is defined in terms of a string,
% 'add.mult.type', where 'add' is a monoid that takes the place of the additive
% operator, 'mult' is the multiplicative operator, and 'type' is the data type
% for the two inputs to the mult operator (the type defaults to the type of
% A for C=A*B).
%
% In the standard semiring, C(i,j) = sum (A(i,:).' .* B(:,j)), using 'plus' as
% the monoid and 'times' as the multiplicative operator.  But in a more general
% semiring, 'sum' can be any monoid, which is an associative and commutative
% operator that has an identity value.  For example, in the 'max.plus' tropical
% algebra, C(i,j) for C=A*B is defined as C(i,j) = max (A(i,:).' + B(:,j)).

n = 3 ;
A = rand (n) ;
B = rand (n) ;
C = zeros (n) ;
for i = 1:n
    for j = 1:n
        C(i,j) = max (A (i,:).' + B (:,j)) ;
    end
end
C2 = gb.mxm ('max.+', A, B) ;
fprintf ('err = norm (C-C2,1) = %g\n', norm (C-C2,1)) ;

%% The max.plus tropical semiring
% Here are details of the "max.plus" tropical semiring.  The identity value
% is -inf since max(x,-inf) = max (-inf,x) = -inf for any x.

gb.semiringinfo ('max.+.double') ;

%% A boolean semiring
% MATLAB cannot multiply two logical matrices; it converts them to double and
% uses the conventional +.*.double semiring instead.  In GraphBLAS, this is the
% common Boolean 'or.and.logical' semiring, which is widely used in linear
% algebraic graph algorithms.

clear
gb.semiringinfo ('|.&.logical') ;

%%
A = sparse (rand (3) > 0.5)
B = sparse (rand (3) > 0.2)

%%
C1 = A*B
C2 = gb (A) * gb (B)

%%
% Note that C1 is a MATLAB sparse double matrix, and contains non-binary
% values.  C2 is a GraphBLAS logical matrix.
whos
gb.type (C2)

%% The GraphBLAS operators, monoids, and semirings
% The C interface for SuiteSparse:GraphBLAS allows for arbitrary types and
% operators to be constructed.  However, the MATLAB interface is restricted to
% pre-defined types and operators.  See 'help gb.binopinfo' for a list of the
% 25 binary operators, and 'help gb.monoidinfo' for the ones that can be used
% as the additive monoid in a semiring.

%% 
help gb.binopinfo

%% 
help gb.monoidinfo

%% Element-wise operations
% Binary operators can be used in element-wise matrix operations, like C=A+B
% and C=A.*B.  For the matrix addition C=A+B, the pattern of C is the set
% union of A and B, and the '+' operator is applied for entries in the
% intersection.  Entries in A but not B, or in B but not A, are assigned to
% C without using the operator.  The '+' operator is used for C=A+B but
% any operator can be used with gb.eadd.

%%
A = gb (sprand (3, 3, 0.5)) ;
B = gb (sprand (3, 3, 0.5)) ;
C1 = A + B
C2 = gb.eadd ('+', A, B)
sparse (C1-C2)

%% Subtracting two matrices
% A-B and gb.eadd ('-', A, B) are not the same thing, since the '-' operator
% is not applied to an entry that is in B but not A.

C1 = A-B 
C2 = gb.eadd ('-', A, B)

%% 
% But these give the same result

C1 = A-B 
C2 = gb.eadd ('+', A, gb.apply ('-', B))
sparse (C1-C2)

%% Element-wise 'multiplication'
% For C = A.*B, the result C is the set intersection of the pattern of A and B.
% The operator is applied to entries in both A and B.  Entries in A but not B,
% or B but not A, do not appear in the result C.

C1 = A.*B
C2 = gb.emult ('*', A, B) 
C3 = sparse(A) .* sparse (B)

%%
% Just as in gb.eadd, any operator can be used in gb.emult:

A
B
C2 = gb.emult ('max', A, B) 

%% Overloaded operators
% The following operators all work as you would expect for any matrix.  The
% matrices A and B can be GraphBLAS matrices, or MATLAB sparse or dense
% matrices, in any combination, or scalars where appropriate:
%
%   A+B    A-B   A*B    A.*B   A./B   A.\B   A.^b    A/b    C=A(I,J)
%   -A     +A    ~A     A'     A.'    A&B    A|B     b\A    C(I,J)=A
%   A~=B   A>B   A==B   A<=B   A>=B   A<B    [A,B]   [A;B]
%
% For A/b and b\A, b must be a scalar.  For A^b, b must be a non-negative
% integer.

C1 = [A B] ;
C2 = [sparse(A) sparse(B)] ;
assert (isequal (sparse (C1), C2))
C1 = A^2 ;
C2 = sparse (A)^2 ;
assert (isequal (sparse (C1), C2))
C1 = A (1:2,2:3)
A = sparse (A) ;
C2 = A (1:2,2:3)

%% Overloaded functions
% Many MATLAB built-in functions can be used with GraphBLAS matrices:
%
% abs        display int16     isinf      issparse min      single  uint8
% all        double  int32     isinteger  istril   nnz      size    uint16
% any        eps     int64     islogical  istriu   nonzeros sparse  uint32
% bandwidth  find    isbanded  ismatrix   isvector norm     spones  uint64
% cast       fix     isdiag    isnan      kron     numel    sqrt   
% ceil       floor   isempty   isnumeric  length   prod     sum       
% diag       full    isfinite  isreal     logical  repmat   tril     
% disp       int8    isfloat   isscalar   max      round    triu
%
% A few differences with the built-in functions:
%   S = sparse (G)      convert a gb matrix G to a MATLAB sparse matrix
%   F = full (G)        convert a gb matrix G to a MATLAB dense matrix
%   disp (G, level)     display a gb matrix G; level=2 is the default.
%   e = nnz (G)         number of entries in a gb matrix G; some can be zero
%   X = nonzeros (G)    all the entries of G; some can be zero

%% Zeros are handled differently
% Explicit zeros cannot be dropped from a GraphBLAS matrix.  In a shortest-path
% problem, for example, an edge A(i,j) that is missing has an infinite weight,
% (the monoid identity of min(x,y) is +inf).  A zero edge weight A(i,j)=0 is
% very different from an entry that is not present in A.  However, if a
% GraphBLAS matrix is converted into a MATLAB sparse matrix, explicit zeros are
% dropped, which is the convention for a MATLAB sparse matrix.

G = gb (magic (3)) - 1
A = sparse (G)
fprintf ('nnz (G): %d  nnz (A): %g\n', nnz (G), nnz (A)) ;

%% Displaying contents of a GraphBLAS matrix
% Unlike MATLAB, the default is to display just a few entries of a gb matrix.
% Here are all 100 entries of a 10-by-10 matrix, using a non-default disp(G,3):

%%
G = gb (rand (10)) ;
% display everything:
disp (G,3)

%%
% That was disp(G,3), so every entry was printed.

%%
% With the default display (level = 2):
G

%%
% That was disp(G,2) or just display(G), which is what is printed by a
% MATLAB statement that doesn't have a trailing semicolon.  With
% level = 1, disp(G,1) gives just a terse summary:
disp (G,1)

%% Storing a matrix by row or by column
% MATLAB stores its sparse matrices by column, refered to as 'standard CSC' in
% SuiteSparse:GraphBLAS.  In the CSC (compressed sparse column) format, each
% column of the matrix is stored as a list of entries, with their value and row
% index.  In the CSR (compressed sparse row) format, each row is stored as a
% list of values and their column indices.  GraphBLAS uses both CSC and CSR,
% and the two formats can be intermixed arbitrarily.  In its C interface, the
% default format is CSR.  However, for better compatibility with MATLAB, this
% MATLAB interface for SuiteSparse:GraphBLAS uses CSC by default instead. 

rng ('default') ;
gb.clear ;                      % clear all prior GraphBLAS settings
default_format_is = gb.format
C = sparse (rand (2))
G = gb (C)
gb.format (G)

%%
% Many graph algorithms work better in CSR format, with matrices stored by row.
% For example, it is common to use A(i,j) for the edge (i,j), and many graph
% algorithms need to access the out-adjacencies of nodes, which is the row
% A(i,;) for node i.  If the CSR format is desired, gb.format ('by row') tells
% GraphBLAS to create all subsequent matrices in the CSR format.  Converting
% from a MATLAB sparse matrix (in standard CSC format) takes a little more time
% (requiring a transpose), but subsequent graph algorithms can be faster.

gb.format ('by row') ;
default_format_is = gb.format
G = gb (C)
default_format_is = gb.format ('by col')
G = gb (C)

%% Hypersparse matrices
% SuiteSparse:GraphBLAS can use two kinds of sparse matrix data structures:
% standard and hypersparse, for both CSC and CSR formats.  In the standard CSC
% format used in MATLAB, an m-by-n matrix A takes O(n+nnz(A)) space.  MATLAB
% can create huge column vectors, but not huge matrices (when n is huge).

clear all
[c, huge] = computer ;
C = sparse (huge, 1)            % MATLAB can create a huge-by-1 sparse column
try
    C = sparse (huge, huge)     % but this fails
catch me
    error_expected = me
end

%%
% In a GraphBLAS hypersparse matrix, an m-by-n matrix A takes only O(nnz(A))
% space.  The difference can be huge if nnz (A) << n.

G = gb (huge, 1)            % no problem for GraphBLAS
H = gb (huge, huge)         % this works in GraphBLAS too

%%
% Operations on huge hypersparse matrices are very fast; no component of
% the time or space complexity is Omega(n).

I = randperm (huge, 2) ;
J = randperm (huge, 2) ;
H (I,J) = 42 ;              % add 4 nonzeros to random locations in H
H = (H' * 2) ;              % transpose H and double the entries
K = gb.expand (pi, H) ;     % K = pi * spones (H)
H = gb.eadd ('+', H, K)     % add pi to each entry in H
numel (H)

%%
% All of these matrices take very little memory space:
whos

%% The mask and accumulator
% When not used in overloaded operators or built-in functions, many GraphBLAS
% methods of the form gb.method ( ... ) can optionally use a mask and/or an
% accumulator operator.  If the accumulator is '+' in gb.mxm, for example, then
% C = C + A*B is computed.  The mask acts much like logical indexing in MATLAB.
% With a logical mask matrix M, C<M>=A*B allows only part of C to be assigned.
% If M(i,j) is true, then C(i,j) can be modified.  If false, then C(i,j) is not
% modified.
%
% For example, to set all values in C that are greater than 0.5 to 3, use:

C = rand (3) 
C1 = gb.assign (C, C > 0.5, 3)      % in GraphBLAS
C (C > .5) = 3                      % in MATLAB
assert (isequal (sparse (C), sparse (C1)))

%% The descriptor
% Most GraphBLAS functions of the form gb.method ( ... ) take an optional last
% argument, called the descriptor.  It is a MATLAB struct that can modify the
% computations performed by the method.  'help gb.descriptorinfo' gives all the
% details.  The following is a short summary of the primary settings:
%
% d.out  = 'default' or 'replace', clears C after the accum op is used
% d.mask = 'default' or 'complement', to use M or ~M as the mask matrix
% d.in0  = 'default' or 'transpose', to transpose A for C=A*B, C=A+B, ...
% d.in1  = 'default' or 'transpose', to transpose B for C=A*B, C=A+B, ...
% d.kind = 'default', 'gb', 'sparse', or 'full'; the output of gb.method.

A = sparse (rand (2)) ;
B = sparse (rand (2)) ;
C1 = A'*B ;
C2 = gb.mxm ('+.*', A, B, struct ('in0', 'transpose')) ;
err = norm (C1-C2,1)

%% Integer arithmetic is different in GraphBLAS
% MATLAB supports integer arithmetic on its full matrices, using int8, int16,
% int32, int64, uint8, uint16, uint32, or uint64 data types.  None of these
% integer data types can be used to construct a MATLAB sparse matrix, which can
% only be double, double complex, or logical.  Furthermore, C=A*B is not
% defined for integer types in MATLAB, except when A and/or B are scalars.
%
% GraphBLAS supports all of those types for its sparse matrices (except for
% complex, which will be added in the future).  All operations are supported,
% including C=A*B when A or B are any integer type, for all 1,040 semirings.
%
% However, integer arithmetic differs in GraphBLAS and MATLAB.  In MATLAB,
% integer values saturate if they exceed their maximum value.  In GraphBLAS,
% integer operators act in a modular fashion.  The latter is essential when
% computing C=A*B over a semiring.  A saturating integer operator cannot be
% used as a monoid since it is not associative.

%%
C = uint8 (magic (3)) ;
G = gb (C) ;
C1 = C * 40
C2 = G * 40
C3 = double (G) * 40 ;
S = double (C1 < 255) ;
assert (isequal (sparse (double (C1).*S), sparse (C2.*S)))
assert (isequal (nonzeros (C2), double (mod (nonzeros (C3), 256))))

%% Example graph algorithm: breadth-first search in MATLAB and GraphBLAS
% The breadth-first search of a graph finds all nodes reachable from the
% source node, and their level, v.  v = bfs_gb(A,s) or v = bfs_matlab(A,s)
% compute the same thing, but bfs_gb uses GraphBLAS matrices and operations,
% while bfs_matlab uses pure MATLAB operations.  v is defined as v(s) = 1 for
% the source node, v(i) = 2 for nodes adjacent to the source, and so on.
%
% Please wait ...

clear all
rng ('default') ;
n = 1e5 ;
A = logical (sprandn (n, n, 1e-3)) ;

tic
v1 = bfs_gb (A, 1) ;
gb_time = toc ;

tic
v2 = bfs_matlab (A, 1) ;
matlab_time = toc ;

assert (isequal (sparse (v1), sparse (v2)))
fprintf ('\nnodes reached: %d of %d\n', nnz (v2), n) ;
fprintf ('GraphBLAS time: %g sec\n', gb_time) ;
fprintf ('MATLAB time:    %g sec\n', matlab_time) ;
fprintf ('Speedup of GraphBLAS over MATLAB: %g\n', matlab_time / gb_time) ;

%% Example graph algorithm: Luby's method in GraphBLAS
% The mis_gb.m function is variant of Luby's randomized algorithm [Luby 1985]. 
% It is a parallel method for finding an maximal independent set of nodes,
% where no two nodes are adjacent.  See the GraphBLAS/demo/mis_gb.m function
% for details.

A = gb (A) ;
A = A|A' ;              % the graph must be symmetric with a zero-free diagonal
A = tril (A, -1) ;
A = A|A' ;
tic
s = mis_gb (A) ;
toc

% make sure it's independent
p = find (s == 1) ;
S = A (p,p) ;
assert (nnz (S) == 0)

% make sure it's maximal
notp = find (s == 0) ;
S = A (notp, p) ;
deg = gb.vreduce ('+.int64', S) ;
assert (full (all (deg > 0)))

%% Sparse deep neural network
% The 2019 GraphChallenge (see http://graphchallenge.org) was to solve a
% set of large sparse deep neural network problems.  In this demo, the MATLAB
% reference solution is compared with a solution using GraphBLAS.  See the
% dnn_gb.m and dnn_matlab.m functions for details.

clear all
rng ('default') ;
nlayers = 16 ;
nneurons = 4096 ;
nfeatures = 30000 ;
fprintf ('# layers:   %d\n', nlayers) ;
fprintf ('# neurons:  %d\n', nneurons) ;
fprintf ('# features: %d\n', nfeatures) ;

tic
Y0 = sprand (nfeatures, nneurons, 0.1) ;
for layer = 1:nlayers
    W {layer} = sprand (nneurons, nneurons, 0.01) * 0.2 ;
    bias {layer} = -0.2 * ones (1, nneurons) ;
end
t_setup = toc ;
fprintf ('construct problem time: %g sec\n', t_setup) ;

%% Solving the sparse deep neural network problem with GraphbLAS
%
% Please wait ...

tic
Y1 = dnn_gb (W, bias, Y0) ;
gb_time = toc ;
fprintf ('total time in GraphBLAS: %g sec\n', gb_time) ;

%% Solving the sparse deep neural network problem with MATLAB
%
% Please wait ...

tic
Y2 = dnn_matlab (W, bias, Y0) ;
matlab_time = toc ;
fprintf ('total time in MATLAB:    %g sec\n', matlab_time) ;
fprintf ('Speedup of GraphBLAS over MATLAB: %g\n', matlab_time / gb_time) ;

err = norm (Y1-Y2,1)

%% Extreme performance differences between GraphBLAS and MATLAB.
% The GraphBLAS operations used so far are perhaps 2x to 50x faster than the
% corresponding MATLAB operations, depending on how many cores your computer
% has.  To run a demo illustrating a 500x or more speedup versus MATLAB,
% run this demo:
%
%    gb_slow_demo
%
% It will illustrate an assignment C(I,J)=A that can take under a second in
% GraphBLAS but several minutes in MATLAB.  To make the comparsion even more
% dramatic, try:
%
%    gb_slow_demo (10000)
%
% assuming you have enough memory.

%% Limitations
% GraphBLAS has a 'non-blocking' mode, in which operations can be left pending
% and completed later.  SuiteSparse:GraphBLAS uses the non-blocking mode to
% speed up a sequence of assignment operations, such as C(I,J)=A.  However, in
% its MATLAB interface, this would require a MATLAB mexFunction to modify its
% inputs.  That breaks the MATLAB API standard, so it cannot be safely done.
% As a result, using GraphBLAS via its MATLAB interface can be slower than when
% using its C API.
%
% As mentioned earlier, GraphBLAS can operate on matrices with arbitrary
% user-defined types and operators.  The only constraint is that the type be a
% fixed sized.  However, in this MATLAB interface, SuiteSparse:GraphBLAS has
% access to only predefined types, operators, and semirings.  Complex types and
% operators will be added to this MATLAB in the future.  They already appear
% in the C version of GraphBLAS, with "user-defined" operators in
% GraphBLAS/Demo/Source/usercomplex.c.

%% GraphBLAS operations
% In addition to the overloaded operators (such as C=A*B) and overloaded
% functions (such as L=tril(A)), GraphBLAS also has methods of the form
% gb.method, listed on the next page.  Most of them take an optional input
% matrix Cin, which is the initial value of the matrix C for the expression
% below, an optional mask matrix M, and an optional accumulator operator.
%
%   C<#M,replace> = accum (C, T)
%
% In the above expression, #M is either empty (no mask), M (with a mask matrix)
% or ~M (with a complemented mask matrix), as determined by the descriptor.
% 'replace' can be used to clear C after it is used in accum(C,T) but before it
% is assigned with C<...> = Z, where Z=accum(C,T).  The matrix T is the result
% of some operation, such as T=A*B for gb.mxm, or T=op(A,B) for gb.eadd.
%
% A short summary of these gb.methods is on the next page.

%% List of gb.methods
%   gb.clear                    clear GraphBLAS workspace and settings
%   gb.descriptorinfo (d)       list properties of a descriptor d
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
%
%   G = gb.build (I, J, X, m, n, dup, type, d)      build a matrix
%   [I,J,X] = gb.extracttuples (A, d)               extract all entries
%
%   C = gb.mxm (Cin, M, accum, semiring, A, B, d)   matrix multiply
%   C = gb.select (Cin, M, accum, op, A, thunk, d)  select entries
%   C = gb.assign (Cin, M, accum, A, I, J, d)       assign, like C(I,J)=A
%   C = gb.subassign (Cin, M, accum, A, I, J, d)    assign, with different M
%   C = gb.colassign (Cin, M, accum, u, I, j, d)    assign to column
%   C = gb.rowassign (Cin, M, accum, u, i, J, d)    assign to row
%   C = gb.vreduce (Cin, M, accum, op, A, d)        reduce to vector
%   C = gb.reduce (Cin, accum, op, A, d)            reduce to scalar
%   C = gb.gbkron (Cin, M, accum, op, A, B, d)      Kronecker product
%   C = gb.gbtranspose (Cin, M, accum, A, d)        transpose
%   C = gb.eadd (Cin, M, accum, op, A, B, d)        element-wise addition
%   C = gb.emult (Cin, M, accum, op, A, B, d)       element-wise multiplication
%   C = gb.apply (Cin, M, accum, op, A, d)          apply unary operator
%   C = gb.extract (Cin, M, accum, A, I, J, d)      extract, like C=A(I,J)
%
% For more details type 'help graphblas' or 'help gb'.
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis
% See also sparse, doc sparse, and https://twitter.com/DocSparse

