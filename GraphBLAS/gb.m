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
% generate gb matrices.  For example G = gb.assign (C, M, A) constructs a
% GraphBLAS matrix G, which is the result of C<M>=A in GraphBLAS notation (like
% C(M)=A(M) in MATLAB).  The matrices used any gb.method may be MATLAB matrices
% (sparse or dense) or GraphBLAS sparse matrices, in any combination.
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
% Methods for the gb class:
%
%   These methods operate on GraphBLAS matrices only, and they overload the
%   existing MATLAB functions of the same name.
%
%       S = sparse (G)          convert a gb matrix G to a MATLAB sparse matrix
%       [I,J,X] = find (G)      extract all entries from a gb matrix
%       X = nonzeros (G)        extract all entries from a gb matrix
%       F = full (G)            convert a gb matrix G to a MATLAB dense matrix
%       C = double (G)          typecast a gb matrix G to double gb matrix C
%       C = single (G)          typecast a gb matrix G to single gb matrix C
%       C = complex (G)         typecast a gb matrix G to complex gb matrix C
%       C = logical (G)         typecast a gb matrix G to logical gb matrix C
%       C = int8 (G)            typecast a gb matrix G to int8 gb matrix C
%       C = int16 (G)           typecast a gb matrix G to int16 gb matrix C
%       C = int32 (G)           typecast a gb matrix G to int32 gb matrix C
%       C = int64 (G)           typecast a gb matrix G to int64 gb matrix C
%       C = uint8 (G)           typecast a gb matrix G to uint8 gb matrix C
%       C = uint16 (G)          typecast a gb matrix G to uint16 gb matrix C
%       C = uint32 (G)          typecast a gb matrix G to uint32 gb matrix C
%       C = uint64 (G)          typecast a gb matrix G to uint64 gb matrix C
%       C = cast (G,...)        typecast a gb matrix G to any of the above
%       C = spones (G)          return pattern of gb matrix
%       s = type (G)            get the type of a gb matrix G
%       disp (G, level)         display a gb matrix G
%       display (G)             display a gb matrix G; same as disp(G,2)
%       mn = numel (G)          m*n for an m-by-n gb matrix G
%       e = nnz (G)             number of entries in a gb matrix G
%       [m n] = size (G)        size of a gb matrix G
%       s = ismatrix (G)        true for any gb matrix G
%       s = isvector (G)        true if m=1 or n=1, for an m-by-n gb matrix G
%       s = isscalar (G)        true if G is a 1-by-1 gb matrix
%       s = isnumeric (G)       true for any gb matrix G
%       s = isfloat (G)         true if gb matrix is double, single, or complex %       s = isreal (G)          true if gb matrix is not complex
%       s = isinteger (G)       true if gb matrix is int8, int16, ..., uint64
%       s = islogical (G)       true if gb matrix is logical
%       L = tril (G,k)          lower triangular part of gb matrix G
%       U = triu (G,k)          upper triangular part of gb matrix G
%       C = kron (A,B)          Kronecker product
%       C = permute (G, ...)    TODO
%       C = ipermute (G, ...)   TODO
%       C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%       C = abs (G)             absolute value
%       s = istril (G)          true if G is lower triangular
%       s = istriu (G)          true if G is upper triangular
%       s = isbanded (G,...)    true if G is banded
%       s = isdiag (G)          true if G is diagonal
%       [lo,hi] = bandwidth (G) determine the lower & upper bandwidth of G
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

properties (SetAccess = private, GetAccess = private)
    % The struct contains the entire opaque content of a GraphBLAS GrB_Matrix.
    opaque = [ ] ;
end

%===============================================================================
methods %=======================================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % gb: construct a GraphBLAS sparse matrix object
    %---------------------------------------------------------------------------

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

    % TODO: these can all be overloaded (not static) methods:
    % TODO norm(G,1), norm(G,inf)
    % TODO max, min, prod, sum, eps, any, all, inf, nan.
    % TODO sin, cos, tan, tanh, ... 
    % TODO ceil, floor, fix
    % TODO sqrt, bsxfun, cummin, cummax, cumprod, diff, ... inv
    % TODO isfinite, isinf, isnan, issorted, issortedrows
    % TODO reshape, sort
    % TODO diag, spdiags
    % TODO ... see 'methods double', 'help datatypes' for more options.

    % add these as gb.methods:
    % TODO gb.maxmax, gb.minmin, gb.sumsum, gb.prodprod, ...

    %---------------------------------------------------------------------------
    % sparse: convert a GraphBLAS sparse matrix into a MATLAB sparse matrix
    %---------------------------------------------------------------------------

    function S = sparse (G)
    %SPARSE convert a GraphBLAS sparse matrix into a MATLAB sparse matrix.
    % S = sparse (G) converts the GraphBLAS matrix G into a MATLAB sparse
    % matrix S, typecasting if needed.  Explicit zeros are dropped from G.
    % MATLAB supports double, complex, and logical sparse matrices.  If G has a
    % different type (int8, ... uint64), it is typecasted to a MATLAB sparse
    % double matrix.
    %
    % See also issparse, full, gb.build, gb.extracttuples, find, gb.
    S = gbselect ('nonzero', G.opaque, struct ('kind', 'sparse')) ;
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
    [~, ~, X] = find (G) ;  % TODO write gbnonzeros to extract just X
    end

    %---------------------------------------------------------------------------
    % full: convert a GraphBLAS sparse matrix into a MATLAB dense matrix
    %---------------------------------------------------------------------------

    function F = full (G, identity)
    %FULL convert a GraphBLAS sparse matrix into a MATLAB dense matrix.
    % F = full (G) converts the GraphBLAS matrix G into a MATLAB dense matrix
    % F.  It assumes the identity value is zero.  F = full (G,id) allows the
    % identity value to be specified, which defines the value of F(i,j) when
    % the entry G(i,j) is not present.  No typecasting is done; F and G have
    % the same type ('double', 'single', 'int8', ...).
    %
    % See also issparse, sparse, gb.build.
    if (nargin == 2)
        F = gbfull (G.opaque, identity) ;
    else
        F = gbfull (G.opaque) ;
    end
    end

    %---------------------------------------------------------------------------
    % double, single, etc: typecast a GraphBLAS sparse matrix to double, etc
    %---------------------------------------------------------------------------

    % Note that these make the built-in 'cast' function work as well, with no
    % further code.  Try C = cast (G, 'like', int16(1)), for example.

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
    C = gb (G, 'uint32') ;
    end

    function C = uint64 (G)
    %UINT64 typecast a GraphBLAS sparse matrix to uint64.
    % C = uint64 (G) typecasts the gb matrix G to uint64.
    C = gb (G, 'uint64') ;
    end

    %---------------------------------------------------------------------------
    % spones: return pattern of GraphBLAS matrix
    %---------------------------------------------------------------------------

    function C = spones (G, type)
    %SPONES return pattern of GraphBLAS matrix.
    % The behavior of spones (G) for a gb matrix differs from spones (S) for a
    % MATLAB matrix S.  An explicit entry G(i,j) that has a value of zero is
    % converted to the explicit entry C(i,j)=1; these entries never appear in
    % spones (S) for a MATLAB matrix S.  C = spones (G) returns C as double
    % (just like the MATLAB spones (S)).  C = spones (G,type) returns C in the
    % requested type ('double', 'single', 'int8', ...).  For example, use
    % C = spones (G, 'logical') to return the pattern of G as a sparse logical
    % matrix.
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
    % To count the number of entries of G that have a nonzero value, use
    % nnz (sparse (G)).
    %
    % See also gb.nvals, nonzeros, size, numel.
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
    % issparse: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = issparse (G)
    %ISSPARSE always true for any GraphBLAS matrix.
    % issparse (G) is always true for any GraphBLAS matrix G.
    %
    % See also ismatrix, isvector, isscalar, sparse, full, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % ismatrix: true for any GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = ismatrix (G)
    %ISMATRIX always true for any GraphBLAS matrix.
    % ismatrix (G) is always true for any GraphBLAS matrix G.
    %
    % See also issparse, isvector, isscalar, sparse, full, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isvector: determine if row or column vector
    %---------------------------------------------------------------------------

    function s = isvector (G)
    %ISVECTOR determine if the GraphBLAS matrix is a row or column vector.
    % isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.
    %
    % See also issparse, ismatrix, isscalar, sparse, full, gb.
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
    % See also issparse, ismatrix, isvector, sparse, full, gb.
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
    % See also isfloat, isreal, isinteger, islogical, type, gb.
    s = true ;
    end

    %---------------------------------------------------------------------------
    % isfloat: determine if a GraphBLAS matrix has a floating-point type
    %---------------------------------------------------------------------------

    function s = isfloat (G)
    %ISFLOAT true for floating-point GraphBLAS matrices.
    %
    % See also isnumeric, isreal, isinteger, islogical, type, gb.
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
    % See also isnumeric, isfloat, isinteger, islogical, type, gb.
    s = ~isequal (gbtype (G.opaque), 'complex') ;
    end

    %---------------------------------------------------------------------------
    % isinteger: determine if a GraphBLAS matrix has an integer type
    %---------------------------------------------------------------------------

    function s = isinteger (G)
    %ISINTEGER true for integer GraphBLAS matrices.
    %
    % See also isnumeric, isfloat, isreal, islogical, type, gb.
    t = gbtype (G.opaque) ;
    s = isequal (t (1:3), 'int') || isequal (t, (1:4), 'uint') ;
    end

    %---------------------------------------------------------------------------
    % islogical: determine if a GraphBLAS matrix has a logical type
    %---------------------------------------------------------------------------

    function s = islogical (G)
    %ISINTEGER true for logical GraphBLAS matrices.
    %
    % See also isnumeric, isfloat, isreal, isinteger, type, gb.
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
    % permute: C = permute (A, order)
    %---------------------------------------------------------------------------

    function C = permute (A, order)
    error ('permute(...) not yet implemented') ;    % TODO permute
    end

    %---------------------------------------------------------------------------
    % ipermute: C = ipermute (A, order)
    %---------------------------------------------------------------------------

    function C = ipermute (A, order)
    error ('ipermute(...) not yet implemented') ;   % TODO ipermute
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
    C = gb.gbkron (['2nd.' type(G)], R, G) ;
    end

    %---------------------------------------------------------------------------
    % abs: absolute value
    %---------------------------------------------------------------------------

    function C = abs (G)
    %ABS Absolute value.
    C = gb.apply ('abs', G) ;
    end

    %---------------------------------------------------------------------------
    % istril: check if lower triangular
    %---------------------------------------------------------------------------

    function result = istril (G)
    %ISTRIL  Determine if a matrix is lower triangular.
    % A GraphBLAS matrix G may have explicit zeros.  If these appear in the
    % upper triangular part of G, then istril (G) is false, but
    % istril (sparse (G)) can be true since the sparse (G) drops those entries.
    result = (gb.nvals (triu (G, 1)) == 0) ;
    end

    %---------------------------------------------------------------------------
    % istriu: check if upper triangular
    %---------------------------------------------------------------------------

    function result = istriu (G)
    %ISTRIU  Determine if a matrix is upper triangular.
    % A GraphBLAS matrix G may have explicit zeros.  If these appear in the
    % lower triangular part of G, then istriu (G) is false, but
    % istriu (sparse (G)) can be true since the sparse (G) drops those entries.
    result = (gb.nvals (tril (G, -1)) == 0) ;
    end

    %---------------------------------------------------------------------------
    % isbanded: check if banded
    %---------------------------------------------------------------------------

    function result = isbanded (G, lo, hi)
    error ('TODO') ;
    end

    %---------------------------------------------------------------------------
    % isdiag: check if diagonal
    %---------------------------------------------------------------------------

    function result = isdiag (G)
    [lo, hi] = bandwidth (G) ;
    result = (lo == 0) && (hi == 0) ;
    end

    %---------------------------------------------------------------------------
    % bandwidth: determine the lower & upper bandwidth
    %---------------------------------------------------------------------------

    function [arg1,arg2] = bandwidth (G,uplo)
    [i j] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
    b = j - i ;
    hi =  double (max (b)) ;
    lo = -double (min (b)) ;
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
    % plus: C = A + B
    %---------------------------------------------------------------------------

    function C = plus (A, B)
    %PLUS sparse matrix addition, C = A+B.
    % A and B can be GraphBLAS matrices or MATLAB sparse or full matrices, in
    % any combination.  If A and B are matrices, the pattern of C is the set
    % union of A and B.  If one of A or B is a scalar, the scalar is expanded
    % into a dense matrix the size of the other matrix, and the result is a
    % dense matrix.  If the type of A and B differ, the type of A is used, as:
    % C = A + gb (B, type (gb (A))).
    %
    % See also gb.eadd, minus, uminus.
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars.  Result is also a scalar.
            C = gb.eadd ('+', A, B) ;
        else
            % A is a scalar, B is a matrix.  Result is full.
            C = gb.eadd ('+', expand_scalar (A, true (size (B))), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar.  Result is full.
            C = gb.eadd ('+', A, expand_scalar (B, true (size (A)))) ;
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
    % C = A + gb (B, type (gb (A))).
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
            C = gb.emult ('*', expand_scalar (A, B), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            C = gb.emult ('*', A, expand_scalar (B, A)) ;
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
    % same as C = gb.mxm (['+.*' type(gb(A))], A, B).  A and B can be GraphBLAS
    % matrices or MATLAB sparse or full matrices, in any combination.
    % 
    % See also gb.mxm.
    C = gb.mxm ('+.*', A, B) ;
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
            C = gb.eadd ('/', expand_scalar (A, true (size (B))), B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            b = get_scalar (B) ;
            if (b == 0 & isfloat (A))
                % 0/0 is Nan, and thus must be computed computed if A is
                % floating-point.  The result is a dense matrix.
                C = gb.eadd ('/', A, expand_scalar (B, true (size (A)))) ;
            else
                % b is nonzero so just compute A/b in the pattern of A.
                % The result is sparse (the pattern of A).
                C = gb.emult ('/', A, expand_scalar (B, A)) ;
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
    % This is not yet supported for GraphBLAS matrices, except when B is a
    % scalar.
    if (isscalar (B))
        C = rdivide (A, B) ;
    else
        error ('A/B not yet supported for GraphBLAS matrices') ;
    end
    end

    %---------------------------------------------------------------------------
    % mldivide: C = A \ B
    %---------------------------------------------------------------------------

    function C = mldivide (A, B)
    % C = A\B, matrix left division
    % This is not yet supported for GraphBLAS matrices, except when A is a
    % scalar.
    if (isscalar (A))
        C = rdivide (B, A) ;
    else
        error ('A\B not yet supported for GraphBLAS matrices') ;
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
            A = expand_scalar (A, true (size (B))) ;
            B = full (B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) <= 0)
                % so the result is full
                A = full (A) ;
                B = expand_scalar (B, true (size (A))) ;
            else
                % b is > 0, and thus 0.^b is zero.  The result is sparse.
                % B is expanded to a matrix wit the same pattern as A.
                B = expand_scalar (B, A) ;
            end
        else
            % both A and B are matrices.
            A = full (A) ;
            B = full (B) ;
        end
    end

    % GraphBLAS does not have a binary operator f(x,y)=x^y.  It could be
    % constructed as a user-defined operator, but this is reasonably fast.
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
            C = gb.build (1:n, 1:n, ones (1, n, get_type (A)), n, n) ;
        else
            % C = A^b where b > 0 is an integer
            C = compute_mpower (A, b) ;
        end
    else
        error ('For C=A^B, B must be a non-negative integer scalar') ;
    end
    end

    function C = compute_mpower (A, b)
    % C = A^b where b > 0 is an integer
    if (b == 1)
        C = A ;
    else
        T = compute_mpower (A, floor (b/2)) ;
        C = T*T ;
        if (mod (b, 2) == 1)
            C = C*A ;
        end
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
                A = expand_scalar (A, true (size (B))) ;
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
                B = expand_scalar (B, true (size (A))) ;
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
                A = expand_scalar (A, true (size (B))) ;
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
                B = expand_scalar (B, true (size (A))) ;
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
                A = expand_scalar (A, true (size (B))) ;
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
                B = expand_scalar (B, true (size (A))) ;
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
                A = expand_scalar (A, true (size (B))) ;
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
                B = expand_scalar (B, true (size (A))) ;
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
    A
    C = gb.gbtranspose (A) ;
    end

    %---------------------------------------------------------------------------
    % transpose: C = A' 
    %---------------------------------------------------------------------------

    function C = transpose (A)
    %TRANSPOSE C = A.', array transpose of a GraphBLAS matrix.
    %
    % See also gb.gbtranspose, ctranspose.
    A
    C = gb.gbtranspose (A) ;
    end

    %---------------------------------------------------------------------------
    % horzcat: C = [A1, A2, ..., An]
    %---------------------------------------------------------------------------

    function C = horzcat (varargin)
    error ('horzcat not yet implemented') ;    % TODO [A , B]
    end

    %---------------------------------------------------------------------------
    % vertcat: C = [A1 ; A2 ; ... ; An]
    %---------------------------------------------------------------------------

    function C = vertcat (varargin)
    error ('vertcat not yet implemented') ;    % TODO [A ; B]
    end

    %---------------------------------------------------------------------------
    % subsref: C = A (I,J)
    %---------------------------------------------------------------------------

    function C = subsref (A, S)
    %SUBSREF C = A(I,J) or C = A(I); extract submatrix of a GraphBLAS matrix
    % C = A(I,J) extracts the A(I,J) submatrix of the GraphBLAS matrix A.
    % With a single index, C = A(I) is equivalent to C = A(I,:).  Linear
    % indexing is not supported.
    %
    % See also subsagn.
    if (~isequal (S.type, '()'))
        error ('index type %s not supported\n', S.type) ;
    end
    ndims = length (S.subs) ;
    if (ndims == 1)
        I = S.subs (1) ;
        C = gb.extract (A, I) ;
    elseif (ndims == 2)
        I = S.subs (1) ;
        J = S.subs (2) ;
        C = gb.extract (A, I, J) ;
    else
        error ('%dD indexing not supported\n', ndims) ;
    end
    end

    %---------------------------------------------------------------------------
    % subsasgn: C (I,J) = A
    %---------------------------------------------------------------------------

    function Cout = subsasgn (Cin, S, A)
    %SUBSAGN C(I,J) = A or C(I) = A; assign submatrix into a GraphBLAS matrix
    % C(I,J) = A assigns A into the C(I,J) submatrix of the GraphBLAS matrix C.
    % With a single index, C(I) = A is equivalent to C(I,:) = A.  Linear
    % indexing is not supported.
    %
    % See also subsref.
    if (~isequal (S.type, '()'))
        error ('index type %s not supported\n', S.type) ;
    end
    ndims = length (S.subs) ;
    if (ndims == 1)
        I = S.subs (1) ;
        Cout = gb.assign (Cin, A, I) ;
    elseif (ndims == 2)
        I = S.subs (1) ;
        J = S.subs (2) ;
        Cout = gb.assign (Cin, A, I, J) ;
    else
        error ('%dD indexing not supported\n', ndims) ;
    end
    end

    %---------------------------------------------------------------------------
    % subsindex: C = B (A)
    %---------------------------------------------------------------------------

    function C = subsindex (A, B)
    error ('subsindex not yet implemented') ;    % TODO C = B(A)
    % C = gb.assign using A as the mask?
    end

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
    %               S = sparse (gb.method (...)) :
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
    % matrix.  However, gb.unopinfo does not have a default type and thus
    % one must be provided, either in the op as gb.unopinfo ('abs.double'), or
    % in the second argument, gb.unopinfo ('abs', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 6 different unary
    % operators, each of which may be used with any of the 11 types, for a
    % total of 6*11 = 66 valid unary operators.  Unary operators are defined by
    % a string of the form 'op.type', or just 'op'.  In the latter case, the
    % type defaults to the the type of the matrix inputs to the GraphBLAS
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
    %   % invalid unary operator (generates an error; this is a binary op):
    %   gb.unopinfo ('+.double') ;
    %
    % gb.unopinfo generates an error for an invalid op, so user code can test
    % the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.binopinfo, gb.semiringinfo, gb.descriptorinfo.
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
    % matrices.  However, gb.binopinfo does not have a default type and thus
    % one must be provided, either in the op as gb.binopinfo ('+.double'), or
    % in the second argument, gb.binopinfo ('+', 'double').
    %
    % The MATLAB interface to GraphBLAS provides for 25 different binary
    % operators, each of which may be used with any of the 11 types, for a
    % total of 25*11 = 275 valid binary operators.  Binary operators are
    % defined by a string of the form 'op.type', or just 'op'.  In the latter
    % case, the type defaults to the the type of the matrix inputs to the
    % GraphBLAS operation.
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
    %   % invalid binary operator (generates an error; this is a unary op):
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
    % its output type (logical) does not match its inputs (double), and since
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
    % TODO G = gb.format (G, 'by row') ; to change the format of one matrix G
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
    elseif (nargin == 0)
        % f = gb.format ; get the global format
        f = gbformat ;
    else
        error ('usage: f = gb.format, gb.format (f), or gb.format (G)') ;
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
    % For a MATLAB full matrix F, gb.nvals (F) and numel (F) are the same.
    %
    % See also nnz, numel.

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
    %       1:   { Ilist }  1D list of row indices, like C(I,J) in MATLAB.
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
    %       int64, or uint64).  This defines I = start:inc:fini in MATLAB
    %       notation.  The start and fini are 1-based if double, 0-based if
    %       int64 or uint64.
    %
    %       The J argument is identical, except that it is a list of column
    %       indices of C.  If only one cell array is provided, J = {  } is
    %       implied, refering to all n columns of C, like C(I,:) in MATLAB
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
    %       1:   { Ilist }  1D list of row indices, like a(I,J) in MATLAB.
    %                   If I is double, then it contains 1-based indices, in
    %                   the range 1 to m if a is m-by-n, so that A(1,1) refers
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
    %       int64, or uint64).  This defines I = start:inc:fini in MATLAB
    %       notation.  The start and fini are 1-based if double, 0-based if
    %       int64 or uint64.
    %
    %       The J argument is identical, except that it is a list of column
    %       indices of A.  If only one cell array is provided, J = {  } is
    %       implied, refering to all n columns of A, like A(I,:) in MATLAB
    %       notation.  1D indexing of a matrix A, as in C = A(I), is not
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
    % M: an optional mask matrix.
    %
    % Example:
    %
    %   A = sprand (5, 4, 0.5)
    %   I = [2 1 5]
    %   J = [3 3 1 2]
    %   Cout = gb.extract (A, {I}, {J})
    %   C2 = A (I,J)
    %   C2 - sparse (Cout)
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

    %---------------------------------------------------------------------------
    % get_type: type of a GraphBLAS or MATLAB matrix
    %---------------------------------------------------------------------------

    function t = get_type (A)
    if (isa (A, 'gb'))
        t = gbtype (A.opaque) ;
    else
        t = gbtype (A) ;
    end
    end

    %---------------------------------------------------------------------------
    % expand_scalar: expand a scalar to a matrix
    %---------------------------------------------------------------------------

    function C = expand_scalar (scalar, S)
    % The scalar is expanded to the pattern of S, as in C = scalar*spones(S).
    % C has the same type as the scalar.  The numerical values of S are
    % ignored; only the pattern of S is used.
    C = gb.gbkron (['1st.' get_type(scalar)], scalar, S) ;
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
    A0 = gb.eadd ('1st', A, expand_scalar (false, B)) ;
    B0 = gb.eadd ('1st', B, expand_scalar (false, A)) ;
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

