classdef gb
%GB GraphBLAS sparse matrices for MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Visit http://graphblas.org
% for more details and resources.  See also the SuiteSparse:GraphBLAS
% User Guide in this package.
%
% The MATLAB gb class represents a GraphBLAS sparse matrix.  The gb
% method creates a GraphBLAS sparse matrix from a MATLAB matrix.  Other
% methods also generate gb matrices.  For example G = gb.subassign (C, M,
% A) constructs a GraphBLAS matrix G, which is the result of C<M>=A in
% GraphBLAS notation (like C(M)=A(M) in MATLAB).  The matrices used any
% gb.method may be MATLAB matrices (sparse or dense) or GraphBLAS sparse
% matrices, in any combination.
%
% The gb constructor:
%
%   The gb constructor creates a GraphBLAS matrix.  The input X may be any
%   MATLAB or GraphBLAS matrix:
%
%   G = gb (X) ;            GraphBLAS copy of a matrix X, same type
%   G = gb (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   G = gb (..., type) ;    create or typecast to a different type
%   G = gb (..., format) ;  create in a specified format
%
%   The m and n parameters above are MATLAB scalars.  The type and format
%   parameters are strings.  The default format is 'by col', to match the
%   format used in MATLAB (see also gb.format), but many graph algorithms
%   are faster if the format is 'by row'.
%
%   The usage G = gb (m, n, type) is analgous to X = sparse (m, n), which
%   creates an empty MATLAB sparse matrix X.  The type parameter is a
%   string, which defaults to 'double' if not present.
%
%   For the usage G = gb (X, type), X is either a MATLAB sparse or dense
%   matrix, or a GraphBLAS sparse matrix object.  G is created as a
%   GraphBLAS sparse matrix object that contains a copy of X, typecasted
%   to the given type if the type string does not match the type of X.
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
%   (see gb.format).
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
%   gb(uint8(255))+1 is zero.  However, new unary and binary operators
%   could be added so that element-wise operations saturate.  The C
%   interface allows for arbitrary creation of user-defined operators, so
%   this could be added in the future.
%
%
% Methods for the gb class:
%
%   These methods operate on GraphBLAS matrices only, and they overload
%   the existing MATLAB functions of the same name.
%
%   G = gb (...)            construct a GraphBLAS matrix
%   C = sparse (G)          makes a copy of a gb matrix
%   C = full (G, ...)       adds explicit zeros or id values to a gb matrix
%   C = double (G)          cast gb matrix to MATLAB sparse double matrix
%   C = logical (G)         cast gb matrix to MATLAB sparse logical matrix
%   C = complex (G)         cast gb matrix to MATLAB sparse complex
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
%   = nzmax (G)           number of entries in a gb matrix G
%   [m n] = size (G)        size of a gb matrix G
%   n = length (G)          length of a gb vector
%   s = isempty (G)         true if any dimension of G is zero
%   s = issparse (G)        true for any gb matrix G
%   s = ismatrix (G)        true for any gb matrix G
%   s = isvector (G)        true if m=1 or n=1, for an m-by-n gb matrix G
%   s = iscolumn (G)        true if n=1, for an m-by-n gb matrix G
%   s = isrow (G)           true if m=1, for an m-by-n gb matrix G
%   s = isscalar (G)        true if G is a 1-by-1 gb matrix
%   s = isnumeric (G)       true for any gb matrix G (even logical)
%   s = isfloat (G)         true if gb matrix is double, single, complex
%   s = isreal (G)          true if gb matrix is not complex
%   s = isinteger (G)       true if gb matrix is int8, int16, ..., uint64
%   s = islogical (G)       true if gb matrix is logical
%   s = isa (G, classname)  check if a gb matrix is of a specific class
%   C = diag (G,k)          diagonal matrices and diagonals of gb matrix G
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
%   s = ishermitian (G)     true if G is Hermitian
%   s = issymmetric (G)     true if G is symmetric
%   [lo,hi] = bandwidth (G) determine the lower & upper bandwidth of G
%   C = sum (G, option)     reduce via sum, to vector or scalar
%   C = prod (G, option)    reduce via product, to vector or scalar
%   s = norm (G, kind)      1-norm or inf-norm of a gb matrix
%   C = max (G, ...)        reduce via max, to vector or scalar
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
%   L = laplacian (G)       graph Laplacian matrix
%   d = degree (G)          degree of G
%   d = indegree (G)        in-degree of G
%   d = outdegree (G)       out-degree of G
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
%   The Static Methods for the gb class can be used on input matrices of
%   any kind: GraphBLAS sparse matrices, MATLAB sparse matrices, or
%   MATLAB dense matrices, in any combination.  The output matrix Cout is
%   a GraphBLAS matrix, by default, but can be optionally returned as a
%   MATLAB sparse or dense matrix.  The static methods divide into two
%   categories: those that perform basic functions, and the GraphBLAS
%   operations that use the mask/accum.
%
%   GraphBLAS basic functions:
%
%   gb.clear                    clear GraphBLAS workspace and settings
%   gb.descriptorinfo (d)       list properties of a descriptor
%   gb.unopinfo (op, type)      list properties of a unary operator
%   gb.binopinfo (op, type)     list properties of a binary operator
%   gb.monoidinfo (op, type)    list properties of a monoid
%   gb.semiringinfo (s, type)   list properties of a semiring
%   t = gb.threads (t)          set/get # of threads to use in GraphBLAS
%   c = gb.chunk (c)            set/get chunk size to use in GraphBLAS
%   e = gb.nvals (G)            number of entries in a matrix
%   G = gb.empty (m, n)         return an empty GraphBLAS matrix
%   s = gb.type (X)             get the type of a MATLAB or gb matrix X
%   s = gb.issigned (type)      true if type is signed
%   f = gb.format (f)           set/get matrix format to use in GraphBLAS
%   C = gb.expand (scalar, S)   expand a scalar (C = scalar*spones(S))
%   C = gb.eye                  identity matrix of any type
%   C = gb.speye                identity matrix (of type 'double')
%   D = gb.coldegree (G)        column degree
%   D = gb.rowdegree (G)        row degree
%   G = gb.build (I, J, X, m, n, dup, type, desc)
%                               build a gb matrix from list of entries
%   [I,J,X] = gb.extracttuples (A, desc)
%                               extract all entries from a matrix
%
%   GraphBLAS operations (as Static methods) with Cout, mask M, and accum:
%
%   Cout = gb.mxm (Cin, M, accum, semiring, A, B, desc)
%                   sparse matrix-matrix multiplication over a semiring
%   Cout = gb.select (Cin, M, accum, op, A, thunk, desc)
%                   select a subset of entries from a matrix
%   Cout = gb.assign (Cin, M, accum, A, I, J, desc)
%                   sparse matrix assignment, such as C(I,J)=A
%   Cout = gb.subassign (Cin, M, accum, A, I, J, desc)
%                   sparse matrix assignment, such as C(I,J)=A
%   Cout = gb.vreduce (Cin, M, accum, op, A, desc)
%                   reduce a matrix to a vector
%   Cout = gb.reduce (Cin, accum, op, A, desc)
%                   reduce a matrix to a scalar
%   Cout = gb.gbkron (Cin, M, accum, op, A, B, desc)
%                   Kronecker product
%   Cout = gb.gbtranspose (Cin, M, accum, A, desc)
%                   transpose a matrix
%   Cout = gb.eadd (Cin, M, accum, op, A, B, desc)
%                   element-wise addition
%   Cout = gb.emult (Cin, M, accum, op, A, B, desc)
%                   element-wise multiplication
%   Cout = gb.apply (Cin, M, accum, op, A, desc)
%                   apply a unary operator
%   Cout = gb.extract (Cin, M, accum, A, I, J, desc)
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
%   gb.descriptorinfo for more details.
%
%   accum is optional; if not is not present, then the operation becomes
%   C<...> = operation(A,B).  Otherwise, C = C + operation(A,B) is
%   computed where '+' is the accum operator.  It acts like a sparse
%   matrix addition (see gb.eadd), in terms of the structure of the
%   result C, but any binary operator can be used.
%
%   The mask M acts like MATLAB logical indexing.  If M(i,j)=1 then
%   C(i,j) can be modified; if zero, it cannot be modified by the
%   operation.
%
% See also sparse, doc sparse, and https://twitter.com/DocSparse .

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

properties (SetAccess = private, GetAccess = private)
    % The struct contains the entire opaque content of a GraphBLAS GrB_Matrix.
    opaque = [ ] ;
end

methods

    %---------------------------------------------------------------------------
    % gb: GraphBLAS matrix constructor
    %---------------------------------------------------------------------------

    function G = gb (varargin)
    %GB GraphBLAS constructor: create a GraphBLAS sparse matrix.
    %
    % G = gb (X) ;          gb copy of a matrix X, same type and format
    %
    % G = gb (X, type) ;    gb typecasted copy of a matrix X, same format
    % G = gb (X, format) ;  gb copy of a matrix X, with given format
    % G = gb (m, n) ;       empty m-by-n gb double matrix, default format
    %
    % G = gb (X, type, format) ;   gb copy of X, new type and format
    % G = gb (X, format, type) ;   ditto
    %
    % G = gb (m, n, type) ;   empty m-by-n gb type matrix, default format
    % G = gb (m, n, format) ; empty m-by-n gb double matrix of given format
    %
    % G = gb (m, n, type, format) ;     empty m-by-n matrix, given type & format
    % G = gb (m, n, format, type) ;     ditto
    %
    % See also sparse.
    if (nargin < 1)
        error ('not enough input arguments') ;
    elseif (nargin == 1 && ...
        (isstruct (varargin {1}) && isfield (varargin {1}, 'GraphBLAS')))
        % G = gb (X), where the input X is a GraphBLAS struct as returned by
        % another gb* function, but this usage is not meant for the end-user.
        % It is only used internally in @gb.  See for @gb/mxm, which uses G =
        % gb (gbmxm (args)), and the typecasting methods, C = double (G), etc.
        % The output of gb is a GraphBLAS object.
        G.opaque = varargin {1} ;
    else
        if (isa (varargin {1}, 'gb'))
            % extract the contents of the gb object as its opaque struct so
            % the gbnew mexFunction can access it.
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % implicitly-defined methods
    %---------------------------------------------------------------------------

    % The following methods work without any implemention needed here:
    %
    %   cast isrow iscolumn ndims

    %---------------------------------------------------------------------------
    % FUTURE:: many these could also be overloaded:
    %---------------------------------------------------------------------------

    % Some of these are trivial (like sin and cos, which would be unary
    % operators defined for gb matrices of type 'double', 'single', or
    % 'complex').  Others are not appropriate for sparse matrices (such as
    % svd), but the inputs to them could be typecasted to MATLAB full matrices
    % ('double', 'single', or 'complex').  Still more have no matrix inputs
    % (sprand, linspace, ...) and thus cannot be overloaded, but gb.sprand
    % would be useful instead as a static method.

    % methods 'double' that are not yet implemented here:
    %
    %    accumarray acos acosd acosh acot acotd acoth acsc acscd acsch airy
    %    asec asecd asech asin asind asinh atan atan2 atan2d atand atanh
    %    bernoulli besselh besseli besselj besselk bessely betainc betaincinv
    %    bsxfun charpoly chebyshevT chebyshevU cos cosd cosh coshint cosint cot
    %    cotd coth csc cscd csch cummax cummin cumprod cumsum dawson det diff
    %    dilog dirac ei ellipticCE ellipticCK ellipticCPi ellipticE ellipticF
    %    ellipticK ellipticNome ellipticPi erf erfc erfcinv erfcx erfi erfinv
    %    euler exp expm1 fresnelc fresnels gamma gammainc gammaincinv gammaln
    %    gegenbauerC harmonic hermiteH hess hypot ichol igamma ilu imag inv
    %    issorted issortedrows jacobiP jordan kummerU laguerreL legendreP
    %    linsolve log log10 log1p log2 logint ltitr maxk mink minpoly mod
    %    ordeig permute pochhammer poly2sym polylog pow2 psi qrupdate rcond
    %    reallog realpow realsqrt rem sec secd sech signIm sin sind sinh
    %    sinhint sinint sort sortrowsc ssinint superiorfloat tan tand tanh
    %    whittakerM whittakerW wrightOmega zeta
    %
    %    double methods never needed: colon factor divisors delete
    %    triangularPulse rectangularPulse 

    % methods in MATLAB/matfun not implemented here:
    %
    %    balance cdf2rdf chol cholupdate condeig condest cond decomposition det
    %    expm funm gsvd hess inv ldl linsolve logm lscov lsqminnorm ltitr lu
    %    normest1 normest null ordeig ordqz ordschur orth pinv planerot polyeig
    %    qrdelete qrinsert qr qrupdate qz rank rcond rref rsf2csf schur sqrtm
    %    svd sylvester trace vecnorm

    % methods in MATLAB/sparsfun not implemented here:
    %
    %    md bicg bicgstabl bicgstab cgs colperm delsq dissect eigs etreeplot
    %    gmres gplot ichol ilu lsqr minres nested numgrid pcg qmr rjr spalloc
    %    spaugment spconvert spdiags spparms sprand sprandn sprandsym sprank
    %    spy svds symbfact symmlq symrcm tfqmr treelayout treeplot unmesh

    % methods in MATLAB/elmat not implemented here:
    %
    %    accumarray blkdiag bsxfun cat circshift compan flipdim fliplr flip
    %    flipud gallery hadamard hankel hilb inf invhilb ipermute isequaln
    %    isequalwithequalnans nan ndgrid pascal permute repelem rot90 shiftdim
    %    toeplitz vander wilkinson
    %
    %    elmat methods never needed: linspace logspace ind2sub sub2ind meshgrid
    %    pi freqspace flintmax intmax intmin squeeze realmin realmax i j magic
    %    rosser 

    %---------------------------------------------------------------------------
    % overloaded methods
    %---------------------------------------------------------------------------

    %   In the list below, G is always a GraphBLAS matrix.  The inputs A and B
    %   can be a mix of GraphBLAS and/or MATLAB matrices, but at least one will
    %   be a GraphBLAS matrix because these are all methods that are overloaded
    %   from the MATLAB versions.  If all inputs are MATLAB matrices, these
    %   methods are not used.  The input X is a matrix of any kind (GraphBLAS
    %   or MATLAB).

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
    C = ctranspose (G) ;
    C = diag (G, k) ;
    display (G) ;
    disp (G, level) ;
    [p, varargout] = dmperm (G) ;
    C = double (G) ;
    [V, varargout] = eig (G, varargin) ;
    i = end (G, k, ndims) ;
    C = eps (G) ;
    C = eq (A, B) ;
    [parent, varargout] = etree (G, varargin) ;
    C = false (varargin) ;
    [I, J, X] = find (G) ;
    C = fix (G) ;
    C = floor (G) ;
    C = full (X, type, identity) ;
    C = ge (A, B) ;
    C = gt (A, B) ;
    C = horzcat (varargin) ;
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
    C = ldivide (A, B) ;
    C = le (A, B) ;
    n = length (G) ;
    C = logical (G) ;
    C = lt (A, B) ;
    C = max (varargin) ;
    C = min (varargin) ;
    C = minus (A, B) ;
    C = mldivide (A, B) ;
    C = mpower (A, B) ;
    C = mrdivide (A, B) ;
    C = mtimes (A, B) ;
    C = ne (A, B) ;
    e = nnz (G) ;
    X = nonzeros (G) ;
    s = norm (G,kind) ;
    C = not (G) ;
    s = numel (G) ;
    e = nzmax (G) ;
    C = ones (varargin) ;
    C = or (A, B) ;
    C = plus (A, B) ;
    C = power (A, B) ;
    C = prod (G, option) ;
    C = rdivide (A, B) ;
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
    C = sqrt (G) ;
    C = subsasgn (C, S, A) ;
    C = subsref (A, S) ;
    C = sum (G, option) ;
    [p, varargout] = symamd (G, varargin) ;
    p = symrcm (G) ;
    C = times (A, B) ;
    C = transpose (G) ;
    L = tril (G, k) ;
    U = triu (G, k) ;
    C = true (varargin) ;
    C = uint16 (G) ;
    C = uint32 (G) ;
    C = uint64 (G) ;
    C = uint8 (G) ;
    C = uminus (G) ;
    C = uplus (G) ;
    C = vertcat (varargin) ;
    C = xor (A, B) ;
    C = zeros (varargin) ;

end

methods (Access = protected, Hidden)

    function disp_helper (G, level)
    %DISP_HELPER display a GraphBLAS matrix for gbgraph/display
    gbdisp (G.opaque, level) ;
    end

end

methods (Static)

    %---------------------------------------------------------------------------
    % Static Methods:
    %---------------------------------------------------------------------------

    % All of these are used as gb.method (...), with the "gb." prefix.

    clear ;
    descriptorinfo (d) ;
    unopinfo (op, type) ;
    binopinfo (op, type) ;
    monoidinfo (monoid, type) ;
    semiringinfo (s, type) ;
    nthreads = threads (varargin) ;
    c = chunk (varargin) ;
    e = nvals (X) ;
    C = empty (arg1, arg2) ;
    s = type (X) ;
    s = issigned (type) ;
    f = format (arg) ;
    C = expand (scalar, S) ;
    C = prune (G, identity) ;
    C = eye (varargin) ;
    C = speye (varargin) ;
    D = coldegree (X) ;
    D = rowdegree (X) ;
    C = build (varargin) ;
    [I,J,X] = extracttuples (varargin) ;
    Cout = mxm (varargin) ;
    Cout = select (varargin) ;
    Cout = assign (varargin) ;
    Cout = subassign (varargin) ;
    Cout = vreduce (varargin) ;
    Cout = reduce (varargin) ;
    Cout = gbkron (varargin) ;
    Cout = gbtranspose (varargin) ;
    Cout = eadd (varargin) ;
    Cout = emult (varargin) ;
    Cout = apply (varargin) ;
    Cout = extract (varargin) ;

end
end

