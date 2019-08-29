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
%       G = gb (...)            construct a GraphBLAS matrix
%       S = sparse (G)          makes a copy of a gb matrix
%       F = full (G, ...)       adds explicit zeros or id values to a gb matrix
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
%       X = nonzeros (G)        extract all entries from a gb matrix
%       [I,J,X] = find (G)      extract all entries from a gb matrix
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
%       s = iscolumn (G)        true if n=1, for an m-by-n gb matrix G
%       s = isrow (G)           true if m=1, for an m-by-n gb matrix G
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
%       C = sum (G, option)     reduce via sum, to vector or scalar
%       C = prod (G, option)    reduce via product, to vector or scalar
%       s = norm (G, kind)      1-norm or inf-norm of a gb matrix
%       C = max (G, ...)        reduce via max, to vector or scalar
%       C = min (G, ...)        reduce via min, to vector or scalar
%       C = any (G, ...)        reduce via '|', to vector or scalar
%       C = all (G, ...)        reduce via '&', to vector or scalar
%       C = sqrt (G)            element-wise square root
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
%       parent = etree (G)      elimination tree
%       C = conj (G)            complex conjugate
%       C = real (G)            real part of a complex GraphBLAS matrix
%       [V, ...] = eig (G,...)  eigenvalues and eigenvectors
%       assert (G)              generate an error if G is false
%       C = zeros (...,'like',G)   all-zero matrix, same type as G
%       C = false (...,'like',G)   all-false logical matrix
%       C = ones (...,'like',G)    matrix with all ones, same type as G
%       L = laplacian (G)       graph Laplacian matrix
%       d = degree (G)          degree of G
%       d = indegree (G)        in-degree of G
%       d = outdegree (G)       out-degree of G
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

    %---------------------------------------------------------------------------
    % implicitly-defined methods
    %---------------------------------------------------------------------------

    % The following methods work without any implemention needed here:
    %
    %   cast isrow iscolumn

    %---------------------------------------------------------------------------
    % FUTURE:: most these could also be overloaded:
    %---------------------------------------------------------------------------

    % methods 'double' that are not yet implemented here:
    %
    %    accumarray acos acosd acosh acot acotd acoth acsc acscd acsch airy
    %    asec asecd asech asin asind asinh atan atan2 atan2d atand atanh
    %    bernoulli besselh besseli besselj besselk bessely betainc betaincinv
    %    bsxfun charpoly chebyshevT chebyshevU colon cos cosd cosh coshint
    %    cosint cot cotd coth csc cscd csch cummax cummin cumprod cumsum dawson
    %    delete det diff dilog dirac divisors ei ellipticCE ellipticCK
    %    ellipticCPi ellipticE ellipticF ellipticK ellipticNome ellipticPi erf
    %    erfc erfcinv erfcx erfi erfinv euler exp expm1 fresnelc fresnels gamma
    %    gammainc gammaincinv gammaln gegenbauerC harmonic hermiteH hess hypot
    %    ichol igamma ilu imag inv issorted issortedrows jacobiP jordan kummerU
    %    laguerreL legendreP linsolve log log10 log1p log2 logint ltitr maxk
    %    mink minpoly mod ordeig permute pochhammer poly2sym polylog pow2 psi
    %    qrupdate rcond reallog realpow realsqrt rectangularPulse rem sec secd
    %    sech signIm sin sind sinh sinhint sinint sort sortrowsc ssinint
    %    superiorfloat tan tand tanh triangularPulse whittakerM whittakerW
    %    wrightOmega xor zeta

    % methods in MATLAB/matfun not implemented here:
    %
    %    balance cdf2rdf chol cholupdate condeig condest cond decomposition det
    %    expm funm gsvd hess inv ldl linsolve logm lscov lsqminnorm ltitr lu
    %    normest1 normest null ordeig ordqz ordschur orth pinv planerot polyeig
    %    private qrdelete qrinsert qr qrupdate qz rank rcond rref rsf2csf schur
    %    sqrtm svd sylvester trace vecnorm

    % methods in MATLAB/sparsfun not implemented here:
    %
    %    md bicg bicgstabl bicgstab cgs colperm delsq dissect eigs etreeplot
    %    gmres gplot ichol ilu lsqr minres nested numgrid pcg private qmr rjr
    %    spalloc sparse spaugment spconvert spdiags spparms sprand sprandn
    %    sprandsym sprank spy svds symbfact symmlq symrcm tfqmr treelayout
    %    treeplot unmesh

    % methods in MATLAB/elmat not implemented here:
    %
    %    accumarray blkdiag bsxfun cat circshift compan flintmax flipdim fliplr
    %    flip flipud freqspace gallery hadamard hankel hilb ind2sub inf intmax
    %    intmin invhilb ipermute isequal isequaln isequalwithequalnans linspace
    %    logspace meshgrid nan ndgrid ndims pascal permute pi repelem rosser
    %    rot90 shiftdim squeeze sub2ind toeplitz vander wilkinson

    % methods for both classes graph and digraph not implemented here:
    %
    %    addedge addnode adjacency bfsearch centrality conncomp dfsearch
    %    distances edgecount findedge findnode incidence isisomorphic
    %    ismultigraph isomorphism maxflow nearest numedges numnodes outedges
    %    plot reordernodes rmedge rmnode shortestpath shortestpathtree simplify

    % methods for class graph (not in digraph class) not implemented here:
    %
    %    bctree biconncomp minspantree neighbors

    % methods for class digraph (not in graph class) not implemented here:
    %
    %    condensation flipedge inedges isdag predecessors successors toposort
    %    transclosure transreduction

end

methods (Static)

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
        Cout = gb (gbkronmex (args {:})) ;
    else
        Cout = gbkronmex (args {:}) ;
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
        Cout = gb (gbtransposemex (args {:})) ;
    else
        Cout = gbtransposemex (args {:}) ;
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

