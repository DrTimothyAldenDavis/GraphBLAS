classdef GrB
%GrB GraphBLAS sparse matrices for MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Visit http://graphblas.org
% for more details and resources.  See also the SuiteSparse:GraphBLAS
% User Guide in this package.
%
% The MATLAB GrB class represents a GraphBLAS sparse matrix.  The GrB
% method creates a GraphBLAS sparse matrix from a MATLAB matrix.  Other
% methods also generate GrB matrices.  For example G = GrB.subassign (C, M,
% A) constructs a GraphBLAS matrix G, which is the result of C<M>=A in
% GraphBLAS notation (like C(M)=A(M) in MATLAB).  The matrices used any
% GrB.method may be MATLAB matrices (sparse or dense) or GraphBLAS sparse
% matrices, in any combination.
%
% The GrB constructor:
%
%   The GrB constructor creates a GraphBLAS matrix.  The input A may be any
%   MATLAB or GraphBLAS matrix:
%
%   C = GrB (A) ;            GraphBLAS copy of a matrix A, same type
%   C = GrB (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   C = GrB (..., type) ;    create or typecast to a different type
%   C = GrB (..., format) ;  create in a specified format
%
%   The m and n parameters above are MATLAB scalars.  The type and format
%   parameters are strings.  The default format is 'by col', to match the
%   format used in MATLAB (see also GrB.format), but many graph algorithms
%   are faster if the format is 'by row'.
%
%   The usage C = GrB (m, n, type) is analgous to A = sparse (m, n), which
%   creates an empty MATLAB sparse matrix A.  The type parameter is a
%   string, which defaults to 'double' if not present.
%
%   For the usage C = GrB (A, type), A is either a MATLAB sparse or dense
%   matrix, or a GraphBLAS sparse matrix object.  C is created as a
%   GraphBLAS sparse matrix object that contains a copy of A, typecasted
%   to the given type if the type string does not match the type of A.
%   If the type string is not present it defaults to 'double'.
%
% Matrix types:
% 
%   Most of the valid type strings correspond to MATLAB class of the same
%   name (see 'help class'):
%
%       'double'    64-bit floating-point (real, not complex)
%       'single'    32-bit floating-point (real, not complex)
%       'logical'   8-bit boolean
%       'int8'      8-bit signed integer
%       'int16'     16-bit signed integer
%       'int32'     32-bit signed integer
%       'int64'     64-bit signed integer
%       'uint8'     8-bit unsigned integer
%       'uint16'    16-bit unsigned integer
%       'uint32'    32-bit unsigned integer
%       'uint64'    64-bit unsigned integer
%       'complex'   64-bit double complex (not yet implemented).
%
% Matrix formats:
%
%   The format of a GraphBLAS matrix can have a large impact on
%   performance.  GraphBLAS matrices can be stored by column or by row.
%   The corresponding format string is 'by col' or 'by row',
%   respectively.  Since the only format that MATLAB supports for its
%   sparse and full matrices is 'by col', that is the default format for
%   GraphBLAS matrices via this MATLAB interfance.  However, the default
%   for the C API is 'by row' since graph algorithms tend to be faster
%   with that format.
%
%   Column vectors are always stored 'by col', and row vectors are always
%   stored 'by row'.  The format for new matrices propagates from the
%   format of their inputs.  For example with C=A*B, C takes on the same
%   format as A, unless A is a vector, in which case C takes on the
%   format of B.  If both A and B are vectors, then the format of C is
%   determined by the descriptor (if present), or by the default format
%   (see GrB.format).
%
%   When a GraphBLAS matrix is converted into a MATLAB sparse or full
%   matrix, it is always returned to MATLAB 'by col'.
%
% Integer operations:
%
%   Operations on integer values differ from MATLAB.  In MATLAB,
%   uint9(255)+1 is 255, since the arithmetic saturates.  This is not
%   possible in matrix operations such as C=A*B, since saturation of
%   integer arithmetic would render most of the monoids useless.
%   GraphBLAS instead computes a result modulo the word size, so that
%   GrB(uint8(255))+1 is zero.  However, new unary and binary operators
%   could be added so that element-wise operations saturate.  The C
%   interface allows for arbitrary creation of user-defined operators, so
%   this could be added in the future.
%
% Methods for the GrB class:
%
%   These methods operate on GraphBLAS matrices only, and they overload
%   the existing MATLAB functions of the same name.
%
%   C = GrB (...)           construct a GraphBLAS matrix
%   C = sparse (G)          makes a copy of a GrB matrix
%   C = full (G, ...)       adds explicit zeros or id values to a GrB matrix
%   C = double (G)          cast GrB matrix to MATLAB sparse double matrix
%   C = logical (G)         cast GrB matrix to MATLAB sparse logical matrix
%   C = complex (G)         cast GrB matrix to MATLAB sparse complex
%   C = single (G)          cast GrB matrix to MATLAB full single matrix
%   C = int8 (G)            cast GrB matrix to MATLAB full int8 matrix
%   C = int16 (G)           cast GrB matrix to MATLAB full int16 matrix
%   C = int32 (G)           cast GrB matrix to MATLAB full int32 matrix
%   C = int64 (G)           cast GrB matrix to MATLAB full int64 matrix
%   C = uint8 (G)           cast GrB matrix to MATLAB full uint8 matrix
%   C = uint16 (G)          cast GrB matrix to MATLAB full uint16 matrix
%   C = uint32 (G)          cast GrB matrix to MATLAB full uint32 matrix
%   C = uint64 (G)          cast GrB matrix to MATLAB full uint64 matrix
%   C = cast (G,...)        cast GrB matrix to MATLAB matrix (as above)
%   X = nonzeros (G)        extract all entries from a GrB matrix
%   [I,J,X] = find (G,...)  extract all entries from a GrB matrix
%   C = spones (G)          return pattern of GrB matrix
%   disp (G, level)         display a GrB matrix G
%   display (G)             display a GrB matrix G; same as disp(G,2)
%   mn = numel (G)          m*n for an m-by-n GrB matrix G
%   e = nnz (G)             number of entries in a GrB matrix G
%   e = nzmax (G)           number of entries in a GrB matrix G
%   [m n] = size (G)        size of a GrB matrix G
%   n = length (G)          length of a GrB vector
%   s = isempty (G)         true if any dimension of G is zero
%   s = issparse (G)        true for any GrB matrix G
%   s = ismatrix (G)        true for any GrB matrix G
%   s = isvector (G)        true if m=1 or n=1, for an m-by-n GrB matrix G
%   s = iscolumn (G)        true if n=1, for an m-by-n GrB matrix G
%   s = isrow (G)           true if m=1, for an m-by-n GrB matrix G
%   s = isscalar (G)        true if G is a 1-by-1 GrB matrix
%   s = isnumeric (G)       true for any GrB matrix G (even logical)
%   s = isfloat (G)         true if GrB matrix is double, single, complex
%   s = isreal (G)          true if GrB matrix is not complex
%   s = isinteger (G)       true if GrB matrix is int8, int16, ..., uint64
%   s = islogical (G)       true if GrB matrix is logical
%   s = isa (G, classname)  check if a GrB matrix is of a specific class
%   C = diag (G,k)          diagonal matrices and diagonals of GrB matrix G
%   L = tril (G,k)          lower triangular part of GrB matrix G
%   U = triu (G,k)          upper triangular part of GrB matrix G
%   C = kron (A,B)          Kronecker product
%   C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%   C = reshape (G, ...)    reshape a GraphBLAS matrix
%   C = abs (G)             absolute value
%   C = sign (G)            signum function
%   s = istril (G)          true if G is lower triangular
%   s = istriu (G)          true if G is upper triangular
%   s = isbanded (G,...)    true if G is banded
%   s = isdiag (G)          true if G is diagonal
%   s = ishermitian (G)     true if G is Hermitian
%   s = issymmetric (G)     true if G is symmetric
%   [lo,hi] = bandwidth (G) determine the lower & upper bandwidth of G
%   C = sum (G, option)     reduce via sum, to vector or scalar
%   C = prod (G, option)    reduce via product, to vector or scalar
%   s = norm (G, kind)      1-norm or inf-norm of a GrB matrix
%   [C,I] = max (G, ...)    reduce via max, to vector or scalar
%   C = min (G, ...)        reduce via min, to vector or scalar
%   C = any (G, ...)        reduce via '|', to vector or scalar
%   C = all (G, ...)        reduce via '&', to vector or scalar
%   C = sqrt (G)            element-wise square root
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
%   parent = etree (G)      elimination tree
%   C = conj (G)            complex conjugate
%   C = real (G)            real part of a complex GraphBLAS matrix
%   [V, ...] = eig (G,...)  eigenvalues and eigenvectors
%   assert (G)              generate an error if G is false
%   C = zeros (...,'like',G)   all-zero matrix, same type as G
%   C = false (...,'like',G)   all-false logical matrix
%   C = ones (...,'like',G)    matrix with all ones, same type as G
%   c = fprintf (...)       print to a file or to the Command Window
%   c = sprintf (...)       print to a string
%   C = flip (G, dim)       flip the order of entries
%   C = sprand (G)          random GraphBLAS matrix
%   C = sprandn (G)         random GraphBLAS matrix, normal distribution
%   C = sprandsym (G, ...)  random symmetric GraphBLAS matrix
%
%   operator overloading:
%
%   C = plus (A, B)         C = A + B
%   C = minus (A, B)        C = A - B
%   C = uminus (G)          C = -G
%   C = uplus (G)           C = +G
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
%   C = not (G)             C = ~G
%   C = ctranspose (G)      C = G'
%   C = transpose (G)       C = G.'
%   C = horzcat (A, B)      C = [A , B]
%   C = vertcat (A, B)      C = [A ; B]
%   C = subsref (A, I, J)   C = A (I,J) or C = A (M)
%   C = subsasgn (A, I, J)  C (I,J) = A
%   index = end (A, k, n)   for object indexing, A(1:end,1:end)
%
% Static Methods:
%
%   The Static Methods for the GrB class can be used on input matrices of
%   any kind: GraphBLAS sparse matrices, MATLAB sparse matrices, or
%   MATLAB dense matrices, in any combination.  The output matrix Cout is
%   a GraphBLAS matrix, by default, but can be optionally returned as a
%   MATLAB sparse or dense matrix.  The static methods divide into two
%   categories: those that perform basic functions, and the GraphBLAS
%   operations that use the mask/accum.
%
%   GraphBLAS basic functions:
%
%   GrB.clear                    clear GraphBLAS workspace and settings
%   GrB.descriptorinfo (d)       list properties of a descriptor
%   GrB.unopinfo (op, type)      list properties of a unary operator
%   GrB.binopinfo (op, type)     list properties of a binary operator
%   GrB.monoidinfo (op, type)    list properties of a monoid
%   GrB.semiringinfo (s, type)   list properties of a semiring
%   t = GrB.threads (t)          set/get # of threads to use in GraphBLAS
%   c = GrB.chunk (c)            set/get chunk size to use in GraphBLAS
%   result = GrB.entries (G,...) count or query entries in a matrix
%   result = GrB.nonz (G,...)    count or query nonzeros in a matrix
%   C = GrB.prune(A, id)         prune entries equal to id
%   C = GrB.offdiag (A)          prune diagonal entries
%   s = GrB.isfull (A)           true if all entries present
%   [C,I,J] = GrB.compact (A,id) remove empty rows and columns
%   G = GrB.empty (m, n)         return an empty GraphBLAS matrix
%   s = GrB.type (A)             get the type of a MATLAB or GrB matrix A
%   s = GrB.issigned (type)      true if type is signed
%   f = GrB.format (f)           set/get matrix format to use in GraphBLAS
%   s = GrB.isbyrow (A)          true if format f A is 'by row'
%   s = GrB.isbycol (A)          true if format f A is 'by col'
%   C = GrB.expand (scalar, A)   expand a scalar (C = scalar*spones(A))
%   C = GrB.eye                  identity matrix of any type
%   C = GrB.speye                identity matrix (of type 'double')
%   C = GrB.random (varargin)    random GraphBLAS matrix
%   C = GrB.build (I, J, X, m, n, dup, type, desc)
%                               build a GrB matrix from list of entries
%   [I,J,X] = GrB.extracttuples (A, desc)
%                               extract all entries from a matrix
%
%   GraphBLAS operations (as Static methods) with Cout, mask M, and accum:
%
%   Cout = GrB.mxm (Cin, M, accum, semiring, A, B, desc)
%                   sparse matrix-matrix multiplication over a semiring
%   Cout = GrB.select (Cin, M, accum, op, A, thunk, desc)
%                   select a subset of entries from a matrix
%   Cout = GrB.assign (Cin, M, accum, A, I, J, desc)
%                   sparse matrix assignment, such as C(I,J)=A
%   Cout = GrB.subassign (Cin, M, accum, A, I, J, desc)
%                   sparse matrix assignment, such as C(I,J)=A
%   Cout = GrB.vreduce (Cin, M, accum, op, A, desc)
%                   reduce a matrix to a vector
%   Cout = GrB.reduce (Cin, accum, op, A, desc)
%                   reduce a matrix to a scalar
%   Cout = GrB.kronecker (Cin, M, accum, op, A, B, desc)
%                   Kronecker product
%   Cout = GrB.trans (Cin, M, accum, A, desc)
%                   transpose a matrix
%   Cout = GrB.eadd (Cin, M, accum, op, A, B, desc)
%                   element-wise addition
%   Cout = GrB.emult (Cin, M, accum, op, A, B, desc)
%                   element-wise multiplication
%   Cout = GrB.apply (Cin, M, accum, op, A, desc)
%                   apply a unary operator
%   Cout = GrB.extract (Cin, M, accum, A, I, J, desc)
%                   extract submatrix, like C=A(I,J) in MATLAB
%
% GraphBLAS operations (with Cout, Cin arguments) take the following form:
%
%   C<#M,replace> = accum (C, operation (A or A', B or B'))
%
%   C is both an input and output matrix.  In this MATLAB interface to
%   GraphBLAS, it is split into Cin (the value of C on input) and Cout
%   (the value of C on output).  M is the optional mask matrix, and #M is
%   either M or !M depending on whether or not the mask is complemented
%   via the desc.mask option.  The replace option is determined by
%   desc.out; if present, C is cleared after it is used in the accum
%   operation but before the final assignment.  A and/or B may optionally
%   be transposed via the descriptor fields desc.in0 and desc.in1,
%   respectively.  To select the format of Cout, use desc.format.  See
%   GrB.descriptorinfo for more details.
%
%   accum is optional; if not is not present, then the operation becomes
%   C<...> = operation(A,B).  Otherwise, C = C + operation(A,B) is
%   computed where '+' is the accum operator.  It acts like a sparse
%   matrix addition (see GrB.eadd), in terms of the structure of the
%   result C, but any binary operator can be used.
%
%   The mask M acts like MATLAB logical indexing.  If M(i,j)=1 then
%   C(i,j) can be modified; if zero, it cannot be modified by the
%   operation.
%
% Static Methods for graph algorithms:
%
%   r = GrB.pagerank (A, opts) ;            % PageRank of a matrix
%   C = GrB.ktruss (A, k, check) ;          % k-truss
%   s = GrB.tricount (A, check) ;           % triangle count
%   L = GrB.laplacian (A, type, check) ;    % Laplacian graph
%   C = GrB.incidence (A, ...) ;            % incidence matrix
%   [v, parent] = GrB.bfs (A, s, ...) ;     % breadth-first search
%   iset = GrB.mis (A, check) ;             % maximal independent set
%   Y = GrB.dnn (W, bias, Y0) ;             % deep neural network
%
% See also sparse.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

properties (SetAccess = private, GetAccess = private)
    % The struct contains the entire opaque content of a GraphBLAS GrB_Matrix.
    opaque = [ ] ;
end

methods

    %---------------------------------------------------------------------------
    % GrB: GraphBLAS matrix constructor
    %---------------------------------------------------------------------------

    function C = GrB (varargin)
    %GRB GraphBLAS constructor: create a GraphBLAS sparse matrix.
    %
    % C = GrB (A) ;          GrB copy of a matrix A, same type and format
    %
    % C = GrB (A, type) ;    GrB typecasted copy of a matrix A, same format
    % C = GrB (A, format) ;  GrB copy of a matrix A, with given format
    % C = GrB (m, n) ;       empty m-by-n GrB double matrix, default format
    %
    % C = GrB (A, type, format) ;   GrB copy of A, new type and format
    % C = GrB (A, format, type) ;   ditto
    %
    % C = GrB (m, n, type) ;   empty m-by-n GrB type matrix, default format
    % C = GrB (m, n, format) ; empty m-by-n GrB double matrix of given format
    %
    % C = GrB (m, n, type, format) ;   empty m-by-n matrix, given type & format
    % C = GrB (m, n, format, type) ;   ditto
    %
    % See also sparse.
    if (nargin == 1 && ...
        (isstruct (varargin {1}) && isfield (varargin {1}, 'GraphBLAS')))
        % C = GrB (A), where the input A is a GraphBLAS struct as returned by
        % another GrB* function, but this usage is not meant for the end-user.
        % It is only used internally in @GrB.  See for @GrB/mxm, which uses
        % C = GrB (gbmxm (args)), and the typecasting methods, C = double (C),
        % etc.  The output of GrB is a GraphBLAS object.
        C.opaque = varargin {1} ;
    else
        if (isa (varargin {1}, 'GrB'))
            % extract the contents of the GrB object as its opaque struct so
            % the gbnew mexFunction can access it.
            varargin {1} = varargin {1}.opaque ;
        end
        C.opaque = gbnew (varargin {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % implicitly-defined methods
    %---------------------------------------------------------------------------

    % The following methods work without any implemention needed here:
    %
    %   flipdim fliplr flipud cast isrow iscolumn ndims sprank etreeplot spy
    %   gplot
    %
    %   bicgstabl bicgstab cgs minres gmres bicg pcg qmr rjr tfqmr lsqr

    %---------------------------------------------------------------------------
    % FUTURE:: many these could also be overloaded:
    %---------------------------------------------------------------------------

    % Some of these are trivial (like sin and cos, which would be unary
    % operators defined for GrB matrices of type 'double', 'single', or
    % 'complex').  Others are not appropriate for sparse matrices (such as
    % svd), but the inputs to them could be typecasted to MATLAB full matrices
    % ('double', 'single', or 'complex').  Still more have no matrix inputs
    % (linspace, ...) and thus cannot be overloaded.

    % methods 'double' that are not yet implemented here:
    %
    %   accumarray acos acosd acosh acot acotd acoth acsc acscd acsch airy asec
    %   asecd asech asin asind asinh atan atan2 atan2d atand atanh bernoulli
    %   besselh besseli besselj besselk bessely betainc betaincinv bsxfun
    %   charpoly chebyshevT chebyshevU cos cosd cosh coshint cosint cot cotd
    %   coth csc cscd csch cummax cummin cumprod cumsum dawson det diff dilog
    %   dirac ei ellipticCE ellipticCK ellipticCPi ellipticE ellipticF
    %   ellipticK ellipticNome ellipticPi erf erfc erfcinv erfcx erfi erfinv
    %   euler exp expm1 fresnelc fresnels gamma gammainc gammaincinv gammaln
    %   gegenbauerC harmonic hermiteH hess hypot ichol igamma ilu imag inv
    %   issorted issortedrows jacobiP jordan kummerU laguerreL legendreP
    %   linsolve log log10 log1p log2 logint ltitr maxk mink minpoly mod ordeig
    %   permute pochhammer poly2sym polylog pow2 psi qrupdate rcond reallog
    %   realpow realsqrt rem sec secd sech signIm sin sind sinh sinhint sinint
    %   sort sortrowsc ssinint superiorfloat tan tand tanh whittakerM
    %   whittakerW wrightOmega zeta
    %
    %   not needed: colon factor divisors delete triangularPulse
    %   rectangularPulse 

    % methods in MATLAB/matfun not implemented here:
    %
    %   balance cdf2rdf chol cholupdate condeig condest cond decomposition det
    %   expm funm gsvd hess inv ldl linsolve logm lscov lsqminnorm ltitr lu
    %   normest1 normest null ordeig ordqz ordschur orth pinv planerot polyeig
    %   qrdelete qrinsert qr qrupdate qz rank rcond rref rsf2csf schur sqrtm
    %   svd sylvester trace vecnorm

    % methods in MATLAB/sparfun not implemented here:
    %
    %   colperm delsq dissect eigs ichol ilu spalloc spaugment spconvert
    %   spdiags svds symbfact symmlq unmesh
    %
    %   not needed: treeplot treelayout numgrid nested spparms

    % methods in MATLAB/elmat not implemented here:
    %
    %   accumarray blkdiag bsxfun cat circshift compan gallery hadamard hankel
    %   hilb inf invhilb ipermute isequaln isequalwithequalnans nan ndgrid
    %   pascal permute repelem rot90 shiftdim toeplitz vander wilkinson
    %
    %   not needed: linspace logspace ind2sub sub2ind meshgrid pi freqspace
    %   flintmax intmax intmin squeeze realmin realmax i j magic rosser 

    % methods for classes graph and digraph not yet implemented:
    %
    %   addedge addnode bfsearch centrality conncomp dfsearch distances
    %   findedge findnode isisomorphic isomorphism maxflow nearest outedges
    %   rmedge rmnode shortestpath shortestpathtree simplify
    %
    %   gbgraph/bfs is like graph/bfsearch and graph/shortestpathtree.

    % methods for class graph (not in digraph class) not yet implemented:
    %
    %   bctree biconncomp minspantree neighbors

    % methods for class digraph (not in graph class) not yet implemented:
    %
    %   condensation inedges isdag predecessors successors toposort
    %   transclosure transreduction

    % methods in LAGraph:
    %
    %   betweeness-centrality, etc ...

    %---------------------------------------------------------------------------
    % Methods that overload built-in MATLAB functions:
    %---------------------------------------------------------------------------

    %   In the list below, G is always a GraphBLAS matrix.  The inputs A and B
    %   can be a mix of GraphBLAS and/or MATLAB matrices, but at least one will
    %   be a GraphBLAS matrix because these are all methods that are overloaded
    %   from the MATLAB versions.  If all inputs are MATLAB matrices, these
    %   methods are not used.  The output matrix (C, L, or U) is always a
    %   GraphBLAS matrix.  Lower case variables i, e, s, and n are scalars.
    %   Outputs p, parent, I, J, and X are MATLAB vectors.  Graph is a MATLAB
    %   undirected graph.  DiGraph is a MATLAB directed digraph.

    C = abs (G) ;
    C = all (G, option) ;
    p = amd (G, varargin) ;
    C = and (A, B) ;
    C = any (G, option) ;
    assert (G) ;
    [arg1, arg2] = bandwidth (G, uplo) ;
    C = ceil (G) ;
    [p, varargout] = colamd (G, varargin) ;
    C = complex (A, B) ;
    C = conj (G) ;
    C = diag (G, k) ;
    DiGraph = digraph (G, option) ;
    display (G) ;
    disp (G, level) ;
    [p, varargout] = dmperm (G) ;
    C = double (G) ;
    [V, varargout] = eig (G, varargin) ;
    i = end (G, k, ndims) ;
    C = eps (G) ;
    [parent, varargout] = etree (G, varargin) ;
    C = false (varargin) ;
    [I, J, X] = find (G, k, search) ;
    C = fix (G) ;
    C = flip (G, dim) ;
    C = floor (G) ;
    c = fprintf (varargin) ;
    C = full (A, type, identity) ;
    Graph = graph (G, varargin) ;
    C = int16 (G) ;
    C = int32 (G) ;
    C = int64 (G) ;
    C = int8 (G) ;
    s = isa (G, classname) ;
    s = isbanded (G, lo, hi) ;
    s = isdiag (G) ;
    s = isempty (G) ;
    s = isequal (A, B) ;
    C = isfinite (G) ;
    s = isfloat (G) ;
    s = ishermitian (G, option) ;
    C = isinf (G) ;
    s = isinteger (G) ;
    s = islogical (G) ;
    s = ismatrix (G) ;
    C = isnan (G) ;
    s = isnumeric (G) ;
    s = isreal (G) ;
    s = isscalar (G) ;
    s = issparse (G) ;
    s = issymmetric (G, option) ;
    s = istril (G) ;
    s = istriu (G) ;
    s = isvector (G) ;
    C = kron (A, B) ;
    n = length (G) ;
    C = logical (G) ;
    [C, I] = max (varargin) ;
    C = min (varargin) ;
    e = nnz (G) ;
    X = nonzeros (G) ;
    s = norm (G,kind) ;
    s = numel (G) ;
    e = nzmax (G) ;
    C = ones (varargin) ;
    C = prod (G, option) ;
    C = real (G) ;
    C = repmat (G, m, n) ;
    C = reshape (G, arg1, arg2) ;
    C = round (G) ;
    C = sign (G) ;
    C = single (G) ;
    [arg1, n] = size (G, dim) ;
    C = sparse (G) ;
    C = spfun (fun, G) ;
    C = spones (G, type) ;
    C = sprand (G) ;
    C = sprandn (G) ;
    C = sprandsym (G, varargin) ;
    c = sprintf (varargin) ;
    C = sqrt (G) ;
    C = sum (G, option) ;
    [p, varargout] = symamd (G, varargin) ;
    p = symrcm (G) ;
    L = tril (G, k) ;
    U = triu (G, k) ;
    C = true (varargin) ;
    C = uint16 (G) ;
    C = uint32 (G) ;
    C = uint64 (G) ;
    C = uint8 (G) ;
    C = xor (A, B) ;
    C = zeros (varargin) ;

    %---------------------------------------------------------------------------
    % MATLAB operator overloading
    %---------------------------------------------------------------------------

    C = ctranspose (A) ;            % C = A'
    C = eq (A, B) ;                 % C = (A == B)
    C = ge (A, B) ;                 % C = (A >= B)
    C = gt (A, B) ;                 % C = (A > B)
    C = horzcat (varargin) ;        % C = [A , B]
    C = ldivide (A, B) ;            % C = A .\ B
    C = le (A, B) ;                 % C = (A <= B)
    C = lt (A, B) ;                 % C = (A < B)
    C = minus (A, B) ;              % C = A - B
    C = mldivide (A, B) ;           % C = A \ B
    C = mpower (A, B) ;             % C = A^B
    C = mrdivide (A, B) ;           % C = A / B
    C = mtimes (A, B) ;             % C = A * B
    C = ne (A, B) ;                 % C = (A ~= B)
    C = not (G) ;                   % C = ~A
    C = or (A, B) ;                 % C = (A | B)
    C = plus (A, B) ;               % C = A + B
    C = power (A, B) ;              % C = A .^ B
    C = rdivide (A, B) ;            % C = A ./ B
    C = subsasgn (C, S, A) ;        % C (I,J) = A
    C = subsref (A, S) ;            % C = A (I,J)
    C = times (A, B) ;              % C = A .* B
    C = transpose (G) ;             % C = A.'
    C = uminus (G) ;                % C = -A
    C = uplus (G) ;                 % C = +A
    C = vertcat (varargin) ;        % C = [A ; B]

end

methods (Static)

    %---------------------------------------------------------------------------
    % Static Methods:
    %---------------------------------------------------------------------------

    % All of these are used as GrB.method (...), with the "GrB." prefix.  The
    % input matrices (A, B, C, M, ...) are of any kind (GraphBLAS, MATLAB
    % sparse, or MATLAB full).  The output matrix C or Cout is a GraphBLAS
    % matrix.

    clear ;
    descriptorinfo (d) ;
    unopinfo (op, type) ;
    binopinfo (op, type) ;
    monoidinfo (monoid, type) ;
    semiringinfo (s, type) ;
    nthreads = threads (varargin) ;
    c = chunk (varargin) ;
    C = empty (arg1, arg2) ;
    s = type (A) ;
    s = issigned (type) ;
    s = isfull (A) ;
    f = format (arg) ;
    s = isbyrow (A) ;
    s = isbycol (A) ;
    C = expand (scalar, A) ;
    C = prune (A, identity) ;
    C = offdiag (A) ;
    C = eye (varargin) ;
    C = speye (varargin) ;
    C = random (varargin) ;
    C = build (varargin) ;
    [I,J,X] = extracttuples (varargin) ;
    Cout = mxm (varargin) ;
    Cout = select (varargin) ;
    Cout = assign (varargin) ;
    Cout = subassign (varargin) ;
    Cout = vreduce (varargin) ;
    Cout = reduce (varargin) ;
    Cout = kronecker (varargin) ;
    Cout = trans (varargin) ;
    Cout = eadd (varargin) ;
    Cout = emult (varargin) ;
    Cout = apply (varargin) ;
    Cout = extract (varargin) ;
    r = pagerank (A, opts) ;
    C = ktruss (A, k, check) ;
    s = tricount (A, check) ;
    L = laplacian (A, type, check) ;
    C = incidence (A, varargin) ;
    [v, parent] = bfs (A, s, varargin) ;
    iset = mis (A, check) ;
    Y = dnn (W, bias, Y0) ;
    result = entries (A, varargin) ;
    result = nonz (A, varargin) ;
    [C, I, J] = compact (A, id) ;

end
end

