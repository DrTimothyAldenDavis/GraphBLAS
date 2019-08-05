classdef gb
%GB GraphBLAS sparse matrices for MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Visit http://graphblas.org for more
% details and resources.  See also the SuiteSparse:GraphBLAS User Guide in this
% package.
%
% The 'gb' class is a MATLAB object that represents a GraphBLAS sparse matrix.
% The gb method creates a GraphBLAS sparse matrix from a MATLAB matrix.
% Other methods also generate gb matrices.  For example G = gb.assign (C, M, A)
% constructs a GraphBLAS matrix G, which is the result of C<M>=A in GraphBLAS
% notation (like C(M)=A(M) in MATLAB).  The matrices used any gb.method may
% be MATLAB matrices (sparse or dense) or GraphBLAS sparse matrices, in any
% combination.
%
% The gb constructor creates a GraphBLAS matrix.  The input X may be any
% MATLAB or GraphBLAS matrix:
%
%   G = gb (X) ;            GraphBLAS copy of a matrix X, same type
%   G = gb (X, type) ;      GraphBLAS typecasted copy of matrix X
%   G = gb (m, n) ;         empty m-by-n GraphBLAS double matrix
%   G = gb (m, n, type) ;   empty m-by-n GraphBLAS matrix of given type
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
%               class name, but instead a property of a MATLAB sparse double
%               matrix.  In GraphBLAS, 'complex' is treated as a type.
%
% To free a GraphBLAS sparse matrix G, simply use 'clear G'.
%
% User-visible Properties of the gb class:
%
%       G.semiring      This is a string that defines the default semiring
%                       when using gb matrices via operator overloading.
%                       It is ignored when using any gb.method.  When the
%           object G is created, G.semiring defaults to '+.*'.  It can be
%           changed, as G.semiring = 'min.first', for example.  Computing A*B
%           with two gb matrices uses B.semiring; with one gb matrix, A*B uses
%           the semiring from the gb matrix.  C=A+B uses the additive monoid.
%           C=A.*B uses the multiplicative operator of B.semiring (if both are
%           gb; if just one, then that semiring is used).  C=A-B becomes
%           A+(-B), and C=-B applies the inverse of the additive monoid of B to
%           each entry.  If the monoid is '+', then the operator is the
%           inverse, this is the conventional negation, -B.  If the monoid is
%           '*', the operator computes the inverse 1/b(i,j).
%
% Methods for the gb class:
%
%   These methods operate on GraphBLAS matrices only, and they overload the
%   existing MATLAB functions of the same name.
%
%       S = sparse (G)      convert a gb matrix G to a MATLAB sparse matrix
%       [I,J,X] = find (G)  extract all entries from a gb matrix
%       F = full (G)        convert a gb matrix G to a MATLAB dense matrix
%       C = double (G)      typecast a gb matrix G to double gb matrix C
%       C = single (G)      typecast a gb matrix G to single gb matrix C
%       C = complex (G)     typecast a gb matrix G to complex gb matrix C
%       C = logical (G)     typecast a gb matrix G to logical gb matrix C
%       C = int8 (G)        typecast a gb matrix G to int8 gb matrix C
%       C = int16 (G)       typecast a gb matrix G to int16 gb matrix C
%       C = int32 (G)       typecast a gb matrix G to int32 gb matrix C
%       C = int64 (G)       typecast a gb matrix G to int64 gb matrix C
%       C = uint8 (G)       typecast a gb matrix G to uint8 gb matrix C
%       C = uint16 (G)      typecast a gb matrix G to uint16 gb matrix C
%       C = uint32 (G)      typecast a gb matrix G to uint32 gb matrix C
%       C = uint64 (G)      typecast a gb matrix G to uint64 gb matrix C
%       s = type (G)        get the type of a gb matrix G
%       disp (G, level)     display a gb matrix G
%       display (G)         display a gb matrix G; same as disp(G,2)
%       mn = numel (G)      m*n for an m-by-n gb matrix G
%       e = nnz (G)         number of entries in a gb matrix G
%       [m n] = size (G)    size of a gb matrix G
%       s = ismatrix (G)    true for any gb matrix G
%       s = isvector (G)    true if m=1 or n=1, for an m-by-n gb matrix G
%       s = isscalar (G)    true if G is a 1-by-1 gb matrix
%       s = isnumeric (G)   true for any gb matrix G
%       s = isfloat (G)     true if gb matrix is double, single, or complex
%       s = isreal (G)      true if gb matrix is not complex
%       s = isinteger (G)   true if gb matrix is int8, int16, ..., uint64
%       s = islogical (G)   true if gb matrix is logical
%       L = tril (G,k)      lower triangular part of gb matrix G
%       U = triu (G,k)      upper triangular part of gb matrix G
%       C = kron (A,B)      Kronecker product
%       C = permute (G, order)  
%       C = ipermute (G, order)
%       C = repmat (G, varargin)
%
%   operator overloading:
%
%       C = plus (A, B)     C = A + B
%       C = minus (A, B)    C = A - B
%       C = uminus (A)      C = -A
%       C = uplus (A)       C = +A
%       C = times (A, B)    C = A .* B
%       C = mtimes (A, B)   C = A * B
%       C = rdivide (A, B)  C = A ./ B
%       C = ldivide (A, B)  C = A .\ B
%       C = mrdivide (A, B) C = A / B
%       C = mldivide (A, B) C = A \ B
%       C = power (A, B)    C = A .^ B
%       C = mpower (A, B)   C = A ^ B
%       C = lt (A, B)       C = A < B
%       C = gt (A, B)       C = A > B
%       C = le (A, B)       C = A <= B
%       C = ge (A, B)       C = A >= B
%       C = ne (A, B)       C = A ~= B
%       C = eq (A, B)       C = A == B
%       C = and (A, B)      C = A & B
%       C = or (A, B)       C = A | B
%       C = not (A)         C = ~A
%       C = ctranspose (A)  C = A'
%       C = transpose (A)   C = A.'
%       C = horzcat (A, B)  C = [A , B]
%       C = vertcat (A, B)  C = [A ; B]
%       C = subsref (A, I, J)   C = A (I,J)
%       C = subsasgn (A, I, J)  C(I,J) = A
%       C = subsindex (A, B)    C = B (A)
%
% Static Methods:
%
%       The Static Methods can be used on input matrices of any kind: GraphBLAS
%       sparse matrices, MATLAB sparse matrices, or MATLAB dense matrices, in
%       any combination.  The output matrix Cout is a GraphBLAS matrix, by
%       default, but can be optionally returned as a MATLAB sparse matrix.  The
%       static methods divide into two categories: those that perform basic
%       functions, and the GraphBLAS operations that use the mask/accum.
%
%   GraphBLAS basic functions:
%        
%       gb.clear                    clear GraphBLAS workspace and settings
%       gb.descriptorinfo (d)       list properties of a descriptor
%       gb.unopinfo (s, type)       list properties of a unary operator
%       gb.binopinfo (s, type)      list properties of a binary operator
%       gb.semiringinfo (s, type)   list properties of a semiring
%       t = gb.threads (t)          set/get # of threads to use in GraphBLAS
%       c = gb.chunk (c)            set/get chunk size to use in GraphBLAS
%       f = gb.format (f)           set/get matrix format to use in GraphBLAS
%       e = gb.nvals (A)            number of entries in a matrix
%       G = gb.empty (m, n)         return an empty GraphBLAS matrix
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
%       Cout = gb.colassign (Cin, M, accum, u, I, j, desc)
%                           sparse matrix assignment to a single column
%       Cout = gb.rowassign (Cin, M, accum, u, i, J, desc)
%                           sparse matrix assignment to a single row
%       Cout = gb.reduce (Cin, M, accum, op, A, desc)
%                           reduce a matrix to a vector or scalar
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

% The following methods are the only common ones that are not overloaded:
% colon, char, loadobj, saveobj

properties
    % The default semiring for overloaded operators is '+.*'.  The G.semiring
    % can be modified directly by the MATLAB user.
    semiring = '+.*' ;
end

properties (SetAccess = private, GetAccess = private)
    % The object properties are a single struct, containing the opaque content
    % of a GraphBLAS GrB_Matrix.
    opaque = [ ] ;
end

%===============================================================================
methods %=======================================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % gb: construct a GraphBLAS sparse matrix object
    %---------------------------------------------------------------------------

    function G = gb (varargin)
    %GB GraphBLAS constructor.

    if (nargin == 0)
        % G = gb ; return an empty GraphBLAS object
        G.opaque = gbnew ;
    elseif (nargin == 1 && ...
        (isstruct (varargin {1}) && isfield (varargin {1}, 'GraphBLAS')))
        % G = gb (X), where the input X is a GraphBLAS struct as returned by
        % another gb* function, but this usage is not meant for the end-user.
        % It is only used in gb.m.  See for example mxm below, which uses G =
        % gb (gbmxm (args)), and the typecasting methods, C = double (G), etc.
        % The output of gb is a GraphBLAS object.
        G.opaque = varargin {1} ;
    else
        %   G = gb (X) ;            gb copy of a matrix X, same type
        %   G = gb (X, type) ;      gb typecasted copy of a matrix X
        %   G = gb (m, n) ;         empty m-by-n gb double matrix
        %   G = gb (m, n, type) ;   empty m-by-n gb matrix of give type
        if (isa (varargin {1}, 'gb'))
            % extract the contents of the gb object as its opaque struct so
            % the gbnew function can access it.
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

    %---------------------------------------------------------------------------

    % TODO: this can all be overloaded (not static) methods:
    % TODO: cast
    % TODO abs, max, min, prod, sum,
    % TODO ceil, floor, fix
    % TODO sqrt? bsxfun?  cummin? cummax? cumprod?  diff?  TODO inv?
    % TODO isbanded, isdiag, isfinite, isinf, isnan, issorted, issortedrows
    % TODO istril, istriu, reshape, sort
    % TODO diag? spdiags?
    % TODO spones
    % TODO ... see 'methods double'

    % gb.methods:
    % gb.maxmax, gb.minmin, gb.sumsum, gb.prodprod

    %---------------------------------------------------------------------------
    % sparse: convert a GraphBLAS sparse matrix into a MATLAB sparse matrix
    %---------------------------------------------------------------------------

    function S = sparse (G)
    %SPARSE convert a GraphBLAS sparse matrix into a MATLAB sparse matrix.
    % S = sparse (G) converts the GraphBLAS matrix G into a MATLAB sparse
    % matrix S, typecasting if needed.  MATLAB supports double, complex, and
    % logical sparse matrices.  If G has a different type (int8, ... uint64),
    % it is typecasted to a MATLAB sparse double matrix.
    S = gbsparse (G.opaque) ;
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
    % not supported.
    %
    % See also gb.extracttuples.
    if (nargout == 3)
        [I, J, X] = gbextracttuples (G.opaque) ;
    elseif (nargout == 2)
        [I, J] = gbextracttuples (G.opaque) ;
    else
        I = gbextracttuples (G.opaque) ;
    end
    end

    %---------------------------------------------------------------------------
    % full: convert a GraphBLAS sparse matrix into a MATLAB dense matrix
    %---------------------------------------------------------------------------

    function F = full (G, identity)
    %FULL convert a GraphBLAS sparse matrix into a MATLAB dense matrix.
    % F = full (G) converts the GraphBLAS matrix G into a MATLAB dense matrix
    % F.  It assumes the identity value is zero.  F = full (G,id) allows the
    % identity value to be specified.  No typecasting is done.
    if (nargin == 2)
        F = gbfull (G.opaque, identity) ;
    else
        F = gbfull (G.opaque) ;
    end
    end

    %---------------------------------------------------------------------------
    % double, single, etc: typecast a GraphBLAS sparse matrix to double, etc
    %---------------------------------------------------------------------------

    function C = double (G)
    %DOUBLE typecast a GraphBLAS sparse matrix to double.
    % C = double (G) typecasts the gb matrix G to double.
    C = gb (G, 'double') ;
    end

    function C = single (G)
    %SINGLE typecast a GraphBLAS sparse matrix to single.
    % C = single (G) typecasts the gb matrix G to single.
    C = gb (G, 'single') ;
    end

    function C = complex (G)
    %COMPLEX typecast a GraphBLAS sparse matrix to complex.
    % C = complex (G) typecasts the gb matrix G to complex.
    error ('complex type not yet supported') ;  % TODO
    end

    function C = logical (G)
    %LOGICAL typecast a GraphBLAS sparse matrix to logical.
    % C = logical (G) typecasts the gb matrix G to logical.
    C = gb (G, 'logical') ;
    end

    function C = int8 (G)
    %INT8 typecast a GraphBLAS sparse matrix to int8.
    % C = int8 (G) typecasts the gb matrix G to int8.
    C = gb (G, 'int8') ;
    end

    function C = int16 (G)
    %INT16 typecast a GraphBLAS sparse matrix to int16.
    % C = int16 (G) typecasts the gb matrix G to int16.
    C = gb (G, 'int16') ;
    end

    function C = int32 (G)
    %INT32 typecast a GraphBLAS sparse matrix to int32.
    % C = int32 (G) typecasts the gb matrix G to int32.
    C = gb (G, 'int32') ;
    end

    function C = int64 (G)
    %INT64 typecast a GraphBLAS sparse matrix to int64.
    % C = int64 (G) typecasts the gb matrix G to int64.
    C = gb (G, 'int64') ;
    end

    function C = uint8 (G)
    %UINT8 typecast a GraphBLAS sparse matrix to uint8.
    % C = uint8 (G) typecasts the gb matrix G to uint8.
    C = gb (G, 'uint8') ;
    end

    function C = uint16 (G)
    %UINT16 typecast a GraphBLAS sparse matrix to uint16.
    % C = uint16 (G) typecasts the gb matrix G to uint16.
    C = gb (G, 'uint16') ;
    end

    function C = uint32 (G)
    %UINT32 typecast a GraphBLAS sparse matrix to uint32.
    % C = uint32 (G) typecasts the gb matrix G to uint32.
    C = gb (G, 'uint32') ; end

    function C = uint64 (G)
    %UINT64 typecast a GraphBLAS sparse matrix to uint64.
    % C = uint64 (G) typecasts the gb matrix G to uint64.
    C = gb (G, 'uint64') ;
    end

    %---------------------------------------------------------------------------
    % type: get the type of GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function s = type (G)
    %TYPE get the type of a GraphBLAS matrix.
    % s = type (G) returns the type of G as a string: 'double', 'single',
    % 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    % 'logical', or 'complex'.  Note that 'complex' is treated as a type,
    % not an attribute.
    %
    % See also class, gb.
    s = gbtype (G.opaque) ;
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
    if (level > 0)
        fprintf ('   default semiring: %s\n\n', G.semiring) ;
    end
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
    fprintf ('   default semiring: %s\n\n', G.semiring) ;
    end

    %---------------------------------------------------------------------------
    % numel: number of elements in a GraphBLAS matrix, m * n
    %---------------------------------------------------------------------------

    function mn = numel (G)
    %NUMEL the maximum number of entries a GraphBLAS matrix can hold.
    % numel (G) is m*n for the m-by-n GraphBLAS matrix G.
    mn = prod (size (G)) ;
    end

    %---------------------------------------------------------------------------
    % nnz: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function e = nnz (G)
    %NNZ the number of entries in a GraphBLAS matrix.
    % nnz (G) is the same as gb.nvals (G); some of the entries may actually be
    % explicit zero-valued entries.  See 'help gb.nvals' for more details.
    %
    % See also gb.nvals, nnz, numel.
    e = gbnvals (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % size: number of rows and columns in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function [m n] = size (G)
    %SIZE the dimensions of a GraphBLAS matrix.
    % [m n] = size (G) is the size of an m-by-n GraphBLAS sparse matrix.
    if (nargout <= 1)
        m = gbsize (G.opaque) ;
    else
        [m n] = gbsize (G.opaque) ;
    end
    end

    %---------------------------------------------------------------------------
    % ismatrix: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = ismatrix (G)
    %ISMATRIX always true for any GraphBLAS matrix.
    % ismatrix (G) is always true for any GraphBLAS matrix G.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isvector: determine if row or column vector
    %---------------------------------------------------------------------------

    function s = isvector (G)
    %ISVECTOR determine if the GraphBLAS matrix is a row or column vector.
    % isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) || (n == 1) ;
    end

    %---------------------------------------------------------------------------
    % isscalar: determine if scalar
    %---------------------------------------------------------------------------

    function s = isscalar (G)
    %ISSCALAR determine if the GraphBLAS matrix is a scalar.
    % isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
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
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isfloat: determine if a GraphBLAS matrix has a floating-point type
    %---------------------------------------------------------------------------

    function s = isfloat (G)
    %ISFLOAT true for floating-point GraphBLAS matrices.
    t = gbtype (G.opaque) ;
    s = isequal (t, 'double') || isequal (t, 'single') || ...
        isequal (t, 'complex') ;
    end

    %---------------------------------------------------------------------------
    % isreal: determine if a GraphBLAS matrix is real (not complex)
    %---------------------------------------------------------------------------

    function s = isreal (G)
    %ISREAL true for real GraphBLAS matrices.
    s = ~isequal (gbtype (G.opaque), 'complex') ;
    end

    %---------------------------------------------------------------------------
    % isinteger: determine if a GraphBLAS matrix has an integer type
    %---------------------------------------------------------------------------

    function s = isinteger (G)
    %ISINTEGER true for integer GraphBLAS matrices.
    t = gbtype (G.opaque) ;
    s = isequal (t (1:3), 'int') || isequal (t, (1:4), 'uint') ;
    end

    %---------------------------------------------------------------------------
    % islogical: determine if a GraphBLAS matrix has a logical type
    %---------------------------------------------------------------------------

    function s = islogical (G)
    %ISINTEGER true for logical GraphBLAS matrices.
    t = gbtype (G.opaque) ;
    s = isequal (t, 'logical') ;
    end

    %---------------------------------------------------------------------------
    % tril: lower triangular part
    %---------------------------------------------------------------------------

    function L = tril (G, k)
    %TRIL lower triangular part of a GraphBLAS matrix.
    % L = tril (G) returns the lower triangular part of G. L = tril (G,k)
    % returns the entries on and below the kth diagonal of G, where k=0 is
    % the main diagonal.
    if (nargin < 2)
        k = 0 ;
    end
    L = gb (gbselect ('tril', G.opaque, k, struct ('kind', 'gb'))) ;
    end

    %---------------------------------------------------------------------------
    % triu: upper triangular part
    %---------------------------------------------------------------------------

    function U = triu (G, k)
    %TRIU upper triangular part of a GraphBLAS matrix.
    % U = triu (G) returns the upper triangular part of G. U = triu (G,k)
    % returns the entries on and above the kth diagonal of X, where k=0 is
    % the main diagonal.
    if (nargin < 2)
        k = 0 ;
    end
    U = gb (gbselect ('triu', G.opaque, k, struct ('kind', 'gb'))) ;
    end

    %---------------------------------------------------------------------------
    % kron: Kronecker product
    %---------------------------------------------------------------------------

    function C = kron (A, B)
    error ('TODO') ;        % kron: use gb.gbkron
    end

    %---------------------------------------------------------------------------
    % permute: C = permute (A, order)
    %---------------------------------------------------------------------------

    function C = permute (A, order)
    % transpose?
    error ('TODO') ;        % permute indices (use transpose)
    end

    %---------------------------------------------------------------------------
    % ipermute: C = ipermute (A, order)
    %---------------------------------------------------------------------------

    function C = ipermute (A, order)
    % transpose?
    error ('TODO') ;        % permute indices (use transpose)
    end

    %---------------------------------------------------------------------------
    % repmat: C = repmat (A, ...)
    %---------------------------------------------------------------------------

    function C = repmat (A, varargin)
    error ('TODO') ;    % repmat: use kron
    end

    %---------------------------------------------------------------------------
    % plus: C = A + B
    %---------------------------------------------------------------------------

    function C = plus (A, B)
    error ('TODO') ;        % plus
    end

    %---------------------------------------------------------------------------
    % minus: C = A - B
    %---------------------------------------------------------------------------

    function C = minus (A, B)
    C = A+(-B) ;
    end

    %---------------------------------------------------------------------------
    % uminus: C = -A
    %---------------------------------------------------------------------------

    function C = uminus (A)
    error ('TODO') ;        % uminus
    end

    %---------------------------------------------------------------------------
    % uplus: C = +A
    %---------------------------------------------------------------------------

    function C = uplus (A)
    error ('TODO') ;        % uplus
    end

    %---------------------------------------------------------------------------
    % times: C = A .* B
    %---------------------------------------------------------------------------

    function C = times (A, B)
    error ('TODO') ;        % A.*B
    % C = gb.emult (gb_get_multop (B.semiring), A, B) ;
    end

    %---------------------------------------------------------------------------
    % mtimes: C = A * B
    %---------------------------------------------------------------------------

    function C = mtimes (A, B)
    %MTIMES sparse matrix-matrix multiplication over a semiring.
    % TODO C=A*B: expand a scalar into a diagonal matrix
    if (isa (B, 'gb'))
        C = gb.mxm (B.semiring, A, B) ;
    elseif (isa (A, 'gb'))
        C = gb.mxm (A.semiring, A, B) ;
    else
        C = gb.mxm ('+.*', A, B) ;
    end
    end

    %---------------------------------------------------------------------------
    % rdivide: C = A ./ B
    %---------------------------------------------------------------------------

    function C = rdivide (A, B)
    % get the multiplicative operator of B.semiring.  If '+' use '-';
    % if '*', use '/', for gb.emult (op, A, B)
    error ('TODO') ;    % A ./ B
    end

    %---------------------------------------------------------------------------
    % ldivide: C = A .\ B
    %---------------------------------------------------------------------------

    function C = ldivide (A, B)
    % get the multiplicative operator of A.semiring.  If '+' use 'rminus';
    % if '*', use '\', for gb.emult (op, A, B)
    error ('TODO') ;    % A .\ B
    end

    %---------------------------------------------------------------------------
    % mrdivide: C = A / B
    %---------------------------------------------------------------------------

    function C = mrdivide (A, B)
    % typecast to double, leave complex as-is, and do
    % C = sparse(A)/sparse(B)
    error ('TODO') ;    % A/B
    end

    %---------------------------------------------------------------------------
    % mldivide: C = A \ B
    %---------------------------------------------------------------------------

    function C = mldivide (A, B)
    % typecast to double, leave complex as-is, and do
    % C = sparse(A)\sparse(B)
    error ('TODO') ;    % A\B
    end

    %---------------------------------------------------------------------------
    % power: C = A .^ B
    %---------------------------------------------------------------------------

    function C = power (A, B)
    % need a new binary operator for this
    error ('TODO') ;    % A.^B
    end

    %---------------------------------------------------------------------------
    % mpower: C = A ^ B
    %---------------------------------------------------------------------------

    function C = mpower (A, B)
    % what should this be?
    error ('TODO') ;    % A^B
    end

    %---------------------------------------------------------------------------
    % lt: C = (A < B)
    %---------------------------------------------------------------------------

    function C = lt (A, B)
    % gb.emult ('<', A, B)?
    error ('TODO') ;    % A<B
    end

    %---------------------------------------------------------------------------
    % gt: C = (A > B)
    %---------------------------------------------------------------------------

    function C = gt (A, B)
    % gb.emult ('>', A, B)?
    error ('TODO') ;    % A>B
    end

    %---------------------------------------------------------------------------
    % le: C = (A <= B)
    %---------------------------------------------------------------------------

    function C = le (A, B)
    % gb.emult ('<=', A, B)?
    error ('TODO') ;    % A<=B
    end

    %---------------------------------------------------------------------------
    % ge: C = (A >= B)
    %---------------------------------------------------------------------------

    function C = ge (A, B)
    % gb.emult ('>=', A, B)?
    error ('TODO') ;    % A>=B
    end

    %---------------------------------------------------------------------------
    % ne: C = (A ~= B)
    %---------------------------------------------------------------------------

    function C = ne (A, B)
    % gb.emult ('~=', A, B)?
    error ('TODO') ;    % A~=B
    end

    %---------------------------------------------------------------------------
    % eq: C = (A == B)
    %---------------------------------------------------------------------------

    function C = eq (A, B)
    % gb.emult ('~=', A, B)?
    error ('TODO') ;    % A==B
    end

    %---------------------------------------------------------------------------
    % and: C = (A & B)
    %---------------------------------------------------------------------------

    function C = and (A, B)
    % gb.emult ('&', A, B)?
    error ('TODO') ;    % A&B
    end

    %---------------------------------------------------------------------------
    % or: C = (A | B)
    %---------------------------------------------------------------------------

    function C = or (A, B)
    % gb.emult ('|', A, B)?
    error ('TODO') ;    % A|B
    end

    %---------------------------------------------------------------------------
    % not: C = (~A)
    %---------------------------------------------------------------------------

    function C = not (A)
    % gb.apply ('not', A)
    error ('TODO') ;    % ~A
    end

    %---------------------------------------------------------------------------
    % ctranspose: C = A' 
    %---------------------------------------------------------------------------

    function C = ctranspose (A)
    % gb.transpose (A) but use gb.apply ('conj', A) first if complex
    error ('TODO') ;    % A'
    end

    %---------------------------------------------------------------------------
    % transpose: C = A' 
    %---------------------------------------------------------------------------

    function C = transpose (A)
    % gb.transpose (A)
    error ('TODO') ;    % A.'
    end

    %---------------------------------------------------------------------------
    % horzcat: C = [A1, A2, ..., An]
    %---------------------------------------------------------------------------

    function C = horzcat (varargin)
    nargin
    args = varargin
    length (args)
    error ('TODO') ;    % [A B]
    end

    %---------------------------------------------------------------------------
    % vertcat: C = [A1 ; A2 ; ... ; An]
    %---------------------------------------------------------------------------

    function C = vertcat (varargin)
    error ('TODO') ;    % [A ; B]
    end

    %---------------------------------------------------------------------------
    % subsref: C = A (I,J)
    %---------------------------------------------------------------------------

    function C = subsref (A, I, J)
    % C = gb.extract (A, {I}, {J}) ;
    error ('TODO') ;    % C=A(I,J)
    end

    %---------------------------------------------------------------------------
    % subsasgn: C (I,J) = A
    %---------------------------------------------------------------------------

%   function C = subsasgn (A, I, J)
%   % C = gb.assign (C, A, {I}, {J}) ;
%   error ('TODO') ;    % C(I,J)=A
%   end

    %---------------------------------------------------------------------------
    % subsindex: C = B (A)
    %---------------------------------------------------------------------------

    function C = subsindex (A, B)
    % C = gb.assign using A as the mask?
    error ('TODO') ;    % C = B(A)
    end

    %---------------------------------------------------------------------------
    % end: ??
    %---------------------------------------------------------------------------

end

%===============================================================================
methods (Static) %==============================================================
%===============================================================================

%-------------------------------------------------------------------------------
% Static methods: user-accessible utility functions
%-------------------------------------------------------------------------------

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
    % This method is optional.  Simply terminating the MATLAB session, or
    % typing 'clear all' will do the same thing.  However, if you are finished
    % with GraphBLAS and wish to free its internal resources, but do not wish
    % to free everything else freed by 'clear all', then use this method.
    % gb.clear also clears any non-default setting of gb.threads, gb.chunk,
    % and gb.format.
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
    %   d.kind  'default', 'gb', 'sparse', or 'full'.  The default is d.kind =
    %            'gb', where the GraphBLAS operation returns an object, which
    %            is preferred since GraphBLAS sparse matrices are faster and
    %            can represent many more data types.  However, if you want a
    %            standard MATLAB sparse matrix, use d.kind='sparse'.  Use
    %            d.kind='full' for a MATLAB dense matrix.  For any gb.method
    %            that takes a descriptor, the following uses are the same, but
    %            the first method is faster and takes less temporary workspace:
    %
    %               d.kind = 'sparse' ;
    %               S = gb.method (..., d) ;
    %
    %               % with no d, or d.kind = 'default'
    %               S = sparse (gb.method (...)) :
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
    % See also gb, gb.unopinfo, gb.binopinfo, gb.semiringinfo.

    if (nargin == 0)
        help gb.descriptorinfo
    else
        gbdescriptorinfo (d) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.unopinfo: list the details of a GraphBLAS unary operator
    %---------------------------------------------------------------------------

    function unopinfo (s, type)
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
    % matrices.  However, this method does not have a default type and thus
    % one must be provided, either in the op as gb.unopinfo ('abs.double'), or
    % in the second argument, gb.unopinfo ('abs', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for TODO different operators,
    % each of which may be used with any of the 11 types, for a total of
    % TODO*11 = TODO valid unary operators.  Unary operators are defined by a
    % string of the form 'op.type', or just 'op'.  In the latter case, the type
    % defaults to the the type of the matrix inputs to the GraphBLAS operation.
    %
    % The following operators are available.
    %
    %   operator name(s) f(x,y)         |   operator names(s) f(x,y)
    %   ---------------- ------         |   ----------------- ------
    %   TODO
    %
    % TODO add complex unary operators
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
    %   % invalid unary operator (generates an error); this is a binary op:
    %   gb.unopinfo ('+.double') ;
    %
    % gb.unopinfo generates an error for an invalid op, so user code can test
    % the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.binopinfo, gb.semiringinfo, gb.descriptorinfo.

    % TODO gbunopinfo

    if (nargin == 0)
        help gb.unopinfo
    elseif (nargin == 1)
        gbunopinfo (s) ;
    else
        gbunopinfo (s, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.binopinfo: list the details of a GraphBLAS binary operator
    %---------------------------------------------------------------------------

    function binopinfo (s, type)
    %GB.BINOPINFO list the details of a GraphBLAS binary operator
    %
    % Usage
    %
    %   gb.binopinfo
    %   gb.binopinfo (op)
    %   gb.binopinfo (op, type)
    %
    % For gb.binopinfo(op), the op must be a string of the form 'op.type',
    % where 'op' is listed below.  The second usage allows the type to be
    % omitted from the first argument, as just 'op'.  This is valid for all
    % GraphBLAS operations, since the type defaults to the type of the input
    % matrices.  However, this method does not have a default type and thus
    % one must be provided, either in the op as gb.binopinfo ('+.double'), or
    % in the second argument, gb.binopinfo ('+', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 25 different operators,
    % each of which may be used with any of the 11 types, for a total of 25*11
    % = 275 valid binary operators.  Binary operators are defined by a string
    % of the form 'op.type', or just 'op'.  In the latter case, the type
    % defaults to the the type of the matrix inputs to the GraphBLAS operation.
    %
    % The 6 comparator operators come in two flavors.  For the is* operators,
    % the result has the same type as the inputs, x and y, with 1 for true and
    % 0 for false.  For example isgt.double (pi, 3.0) is the double value 1.0.
    % For the second set of 6 operators (eq, ne, gt, lt, ge, le), the result
    % always has a logical type (true or false).  In a semiring, the type of
    % the add monoid must exactly match the type of the output of the multiply
    % operator, and thus 'plus.iseq.double' is valid (counting how many terms
    % are equal).  The 'plus.eq.double' semiring is valid, but not the same
    % semiring since the 'plus' of 'plus.eq.double' has a logical type and is
    % thus equivalent to 'or.eq.double'.   The 'or.eq' is true if any terms are
    % equal and false otherwise (it does not count the number of terms that are
    % equal).
    %
    % The following operators are available.  Many have equivalent synonyms, so
    % that '1st' and 'first' both define the first(x,y) = x operator.
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
    %   xor lxor                        |
    %
    % TODO add complex operators
    %
    % The three logical operators, lor, land, and lxor, also come in 11 types.
    % z = lor.double (x,y) tests the condition (x ~= 0) || (y ~= 0), and returns
    % the double value 1.0 if true, or 0.0 if false.
    %
    % Example:
    %
    %   % valid binary operators
    %   gb.binopinfo ('+.double') ;
    %   gb.binopinfo ('1st.int32') ;
    %
    %   % invalid binary operator (generates an error); this is a unary op:
    %   gb.binopinfo ('abs.double') ;
    %
    % gb.binopinfo generates an error for an invalid op, so user code can test
    % the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.unopinfo, gb.semiringinfo, gb.descriptorinfo.

    if (nargin == 0)
        help gb.binopinfo
    elseif (nargin == 1)
        gbbinopinfo (s) ;
    else
        gbbinopinfo (s, type) ;
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
    % defaults to the type of the input matrices.  However, this method does
    % not have a default type and thus one must be provided, either in the
    % semiring as gb.semiringinfo ('+.*.double'), or in the second argument,
    % gb.semiringinfo ('+.*', 'double').
    %
    % The add operator must be a valid monoid: plus, times, min, max, and the
    % boolean operators or.logical, and.logical, ne.logical, and xor.logical.
    % The binary operator z=f(x,y) of a monoid must be associative and
    % commutative, with an identity value id such that f(x,id) = f(id,x) = x.
    % Furthermore, the types of x, y, and z must all be the same.  Thus, the
    % '<.double' is not a valid operator for a monoid, since its output type
    % (logical) does not match its inputs (double), and since it is neither
    % associative nor commutative.  Thus, <.*.double is not a valid semiring.
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
    % gb.chunk).  The setting persists for the current MATLAB session, or until
    % 'clear all' or gb.clear is used, at which point the setting reverts to
    % the default number of threads.
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
    % format: get/set the default GraphBLAS matrix format
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
    % Graph algorithms tend to be faster with their matrices held by row,
    % since the edge (i,j) is typically the entry G(i,j) in the matrix G,
    % and most graph algorithms need to know the outgoing edges of node i.
    % This is G(i,:), which is very fast if G is held by row, but very slow
    % if G is held by column.
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
    %   f = format (G)
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

    if (nargin == 1)
        if (isa (arg, 'gb'))
            % f = gb.format (G) ; get the format of the matrix G
            f = gbformat (arg.opaque) ;
        else
            % f = gb.format (f) ; set the global format for all future matrices
            f = gbformat (arg) ;
        end
    else
        % f = gb.format ; get the global format
        f = gbformat ;
    end
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
    % For a MATLAB dense matrix F, gb.nvals (F) and numel (F) are the same.
    %
    % See also nnz, numel
    if (isa (A, 'gb'))
        e = gbnvals (A.opaque) ;
    else
        e = gbnvals (A) ;
    end
    end

%-------------------------------------------------------------------------------
% Static methods that return a GraphBLAS matrix or a MATLAB matrix
%-------------------------------------------------------------------------------

    %---------------------------------------------------------------------------
    % gb.empty: construct an empty GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function G = empty (arg1, arg2)
    %GB.EMPTY construct an empty GraphBLAS sparse matrix
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
    %   S = sparse (gb.build (I, J, X)) ;
    %   S = sparse (gb.build (uint64(I)-1, uint64(J)-1, X)) ;
    %
    % Using uint64 integers for I and J is faster and uses less memory.  I and
    % J need not be in any particular order, but gb.build is fastest if I and J
    % are provided in column-major order.
    %
    % See also sparse, find, gb.extractuples.

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
    % uint8, or complex).
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
    % then the '+.*.complex' semiring is used.  GraphBLAS has many more
    % semirings it can use.  See 'help gb.semiringinfo' for more details.
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
    % Note that C = gb.select ('diag',A) does return a vector, but a diagonal
    % matrix.
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
    %GB.ASSIGN: assign a submatrix of a GraphBLAS sparse matrix
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
    %       1:   { Ilist }  1D list of row indices, like C(I,J) in MATLAB.
    %                   If I is double, then it contains 1-based indices, in
    %                   the range 1 to m if C is m-by-n, so that C(1,1) refers
    %                   to the entry in the first row and column of C.  If I is
    %                   int64 or uint64, then it contains 0-based indices in
    %                   the range 0 to m-1, where C(0,0) is the same entry.
    %
    %       2:  { start,fini }  start and fini are scalars (either double,
    %                   int64, or uint64).  This defines I = start:fini) in
    %                   MATLAB index notation.  Typically, start and fini have
    %                   type double and refer to 1-based indexing of C.  int64
    %                   or uint64 scalars are treated as 0-based.
    %
    %       3:  { start,inc,fini } start, inc, and fini are scalars (double,
    %       int64, or uint64).  This defines I = start:inc:fini in MATLAB
    %       notation.  The start and fini are 1-based if double, 0-based if
    %       int64 or uint64.
    %
    %       The J argument is identical, except that it is a list of column
    %       indices of C.  If only one cell array is provided, J = {  } is
    %       implied, refering to all n columns of C (like C(I,:) in MATLAB
    %       notation.  1D indexing of a matrix C, as in C(I) = A, is not
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
    % M: an optional mask matrix.
    % 
    % Cin: a required input matrix, containing the initial content of the
    % matrix C.  Cout is the content of C after the assignment is made.
    %
    % Example:
    %
    %   A = sprand (5, 4, 0.5)
    %   AT = A'
    %   M = sparse (rand (n)) > 0.5
    %   Cin = sprand (n, n, 0.5)
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
    %GB.SUBASSIGN subassign sparse submatrix
    %   Cout = gb.subassign (Cin, M, accum, A, I, J, desc)
    % See also gb.assign, subsasgn.

    error ('gb.subassign not yet implemented') ;    % TODO
    end

    %---------------------------------------------------------------------------
    % gb.colassign: sparse matrix assignment to a single column
    %---------------------------------------------------------------------------

    function Cout = colassign (varargin)
    %GB.COLASSIGN sparse matrix assignment to a single column
    %   Cout = gb.colassign (Cin, M, accum, u, I, j, desc)
    error ('gb.colassign not yet implemented') ;    % TODO
    end

    %---------------------------------------------------------------------------
    % gb.rowassign: sparse matrix assignment to a single row
    %---------------------------------------------------------------------------

    function Cout = rowassign (varargin)
    %GB.ROWASSIGN sparse matrix assignment to a single row
    %   Cout = gb.rowassign (Cin, M, accum, u, i, J, desc)
    error ('gb.rowassign not yet implemented') ;    % TODO
    end

    %---------------------------------------------------------------------------
    % gb.reduce: reduce a matrix to a vector or scalar
    %---------------------------------------------------------------------------

    function Cout = reduce (varargin)
    %GB.REDUCE reduce a matrix to a vector or scalar
    %
    % Usage:
    %
    %   Cout = gb.reduce (Cin, M, accum, op, A, desc)
    %
    % TODO: separate functions for reduce-to-vector and reduce-to-scalar?
    %
    % See also sum, prod, accumarry, max, min.

    error ('gb.reduce not yet implemented') ;   % TODO
    end

    %---------------------------------------------------------------------------
    % gb.kron: Kronecker product
    %---------------------------------------------------------------------------

    function Cout = gbkron (varargin)
    %GB.GBKRON sparse Kronecker product
    %
    % Usage:
    %
    %   Cout = gb.gbkron (Cin, M, accum, op, A, B, desc)

    error ('gb.gbkron not yet implemented') ;   % TODO
    end

    %---------------------------------------------------------------------------
    % gb.gbtranspose: transpose a matrix
    %---------------------------------------------------------------------------

    function Cout = gbtranspose (varargin)
    %GB.GBTRANSPOSE transpose a matrix
    %
    % Usage:
    %
    %   Cout = gb.gbtranspose (Cin, M, accum, A, desc)
    %
    % See also transpose, ctranspose.

    error ('gb.gbtranspose not yet implemented') ;  % TODO
    end

    %---------------------------------------------------------------------------
    % gb.eadd: sparse matrix addition
    %---------------------------------------------------------------------------

    function Cout = eadd (varargin)
    %GB.EADD sparse matrix addition
    %
    % Usage:
    %
    %   Cout = gb.eadd (Cin, M, accum, op, A, B, desc)
    %
    % gb.eadd computes the element-wise 'addition' T=A+B.  The result T has the
    % pattern of the union of A and B. The operator is used where A(i,j) and
    % B(i,j) are present.  Otherwise the entries in A and B are copied directly
    % into T:
    %
    %   if (A(i,j) and B(i,j) is present)
    %       T(i,j) = op (A(i,j), B(i,j))
    %   else if (A(i,j) is present but B(i,j) is not)
    %       T(i,j) = A(i,j)
    %   else if (B(i,j) is present but A(i,j) is not)
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
    %   Cout = gb.apply (Cin, M, accum, op, A, desc)

    error ('gb.apply not yet implemented') ;    % TODO
    end

    %---------------------------------------------------------------------------
    % gb.extract: extract a submatrix
    %---------------------------------------------------------------------------

    function Cout = extract (varargin)
    %GB.EXTRACT extract sparse submatrix
    %
    % Usage:
    %
    %   Cout = gb.extract (Cin, M, accum, A, I, J, desc)
    %
    % See also subsref.

    error ('gb.extract not yet implemented') ;  % TODO
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
    %GET_ARGS get the arguments and the descriptor for a gb.method.
    %
    % Any input arguments that are GraphBLAS sparse matrix objects are replaced
    % with the struct arg.opaque so that they can be passed to the underlying
    % mexFunction.
    %
    % Next, the descriptor is modified to change the default d.kind.
    %
    % All mexFunctions in private/mexFunction/*.c require the descriptor to be
    % present as the last argument.  They are not required for the user-
    % accessible gb.methods.
    %
    % If the descriptor d is not present, then it is created and appended to
    % the argument list, with d.kind = 'gb'.  If the descriptor is present and
    % d.kind does not appear, then d.kind = 'gb' is set.  Finally, is_gb is set
    % true if d.kind is 'gb'.  If d.kind is 'gb', then the underlying
    % mexFunction returns a GraphBLAS struct, which is then converted above to
    % a GraphBLAS object.  See for example G = gb (gbmxm (args {:})) above.

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

