classdef gb
%GB GraphBLAS sparse matrices for MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Visit http://graphblas.org for more
% details and resources.  See also the SuiteSparse:GraphBLAS User Guide in this
% package.
%
% The 'gb' class is a MATLAB object that represents a GraphBLAS sparse matrix.
% The gb function creates a GraphBLAS sparse matrix from a MATLAB matrix.
% Other methods also generate gb matrices.  For example G = gb.assign (C, M, A)
% constructs a GraphBLAS matrix G, with the result of the MATLAB computation
% C(M)=A(M); in MATLAB this is referred to as logical indexing.  The matrices
% C, M, and A can be either MATLAB matrices or GraphBLAS sparse matrices, in
% any combination.
%
% The gb constructor:  convert a MATLAB sparse matrix to a GraphBLAS matrix
%
%   G = gb (...)            construct a gb matrix
%
% Methods for the gb class: these operate on GraphBLAS matrices only.
%
%   S = sparse (G)          convert a gb matrix G to a MATLAB sparse matrix S
%   F = full (G)            convert a gb matrix G to a MATLAB dense matrix F
%   G = double (X)          typecast a gb matrix G to double
%   G = single (X)          typecast a gb matrix G to single
%   G = complex (X)         typecast a gb matrix G to complex (TODO)
%   G = logical (X)         typecast a gb matrix G to logical
%   G = int8 (X)            typecast a gb matrix G to int8
%   G = int16 (X)           typecast a gb matrix G to int16
%   G = int32 (X)           typecast a gb matrix G to int32
%   G = int64 (X)           typecast a gb matrix G to int64
%   G = uint8 (X)           typecast a gb matrix G to uint8
%   G = uint16 (X)          typecast a gb matrix G to uint16
%   G = uint32 (X)          typecast a gb matrix G to uint32
%   G = uint64 (X)          typecast a gb matrix G to uint64
%   s = type (G)            get the type of a gb matrix G ('double', ...)
%   disp (G, level)         display a gb matrix G
%   display (G)             display a gb matrix G
%   result = numel (G)      m*n for an m-by-n gb matrix G
%   result = nvals (G)      number of entries in a gb matrix G
%   result = nnz (G)        number of entries in a gb matrix G
%   [m n] = size (G)        size of a gb matrix G
%   s = ismatrix (G)        true for any gb matrix G
%   s = isvector (G)        true if m or n is one, for an m-by-n gb matrix G
%   s = isscalar (G)        true if G is a 1-by-1 gb matrix
%   L = tril (G,k)          lower triangular part of gb matrix G
%   U = triu (G,k)          upper triangular part of gb matrix G
%   G = setsemiring (G,s)   set the default semiring of G to s
%   s = getsemiring (G)     get the default semiring
%
% Static methods: these can be used on input matrices of any kind: GraphBLAS
%        sparse matrices, MATLAB sparse matrices, or MATLAB dense matrices,
%        in any combination.  The output matrix Cout is a GraphBLAS matrix,
%        by default, but can be optionally returned as a MATLAB sparse matrix.
%        The static methods divide into two categories: those that perform
%        basic functions, and the GraphBLAS operations that use the mask/accum.
%
%   GraphBLAS basic functions:
%        
%       gb.clear                    clear internal GraphBLAS workspace
%       gb.descriptorinfo (d)       list properties of a descriptor
%       gb.binopinfo (s, type)      list properties of a binary operator
%       gb.unopinfo (s, type)       list properties of a unary operator
%       gb.semiringinfo (s, type)   list properties of a semiring
%       t = gb.threads (t)          set/get # of threads to use in GraphBLAS
%       G = gb.empty (m, n)         return an empty GraphBLAS matrix
%
%       G = gb.build (I, J, X, m, n, dup, type, desc)
%                           build a GraphBLAS matrix from a list of entries
%       [I,J,X] = find (A, onebased)
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
%       Cout = gb.extract (Cin, M, accum, A, I, J, desc)
%                           extract submatrix, like C=A(I,J) in MATLAB
%       Cout = gb.apply (Cin, M, accum, op, A, desc)
%                           apply a unary operator
%       Cout = gb.reduce (Cin, M, accum, op, A, desc)
%                           reduce a matrix to a vector or scalar
%       Cout = gb.kron (Cin, M, accum, op, A, B, desc)
%                           Kronecker product
%       Cout = gb.transpose (Cin, M, accum, A, desc)
%                           transpose a matrix
%       Cout = gb.eadd (Cin, M, accum, op, A, B, desc)
%                           element-wise addition
%       Cout = gb.emult (Cin, M, accum, op, A, B, desc)
%                           element-wise multiplication
%
%       GraphBLAS operations (with Cout and Cin arguments) take the following
%       form:
%
%       C<#M,replace> = accum (C, operation (A,B))
%
%       C is both an input and output matrix.  In this MATLAB interface to
%       GraphBLAS, it is split into Cin (the value of C on input) and Cout (the
%       value of C on output).  M is the optional mask matrix, and #M is either
%       M or !M depending on whether or not the mask is complemented via the
%       desc.mask option.  The replace option is determined by desc.out; if
%       present, C is cleared after it is used in the accum operation but
%       before the final assignment.  A and/or B may optionally be transposed
%       via the descriptor fields desc.in0 and desc.in1, respectively.  See
%       'help gb.descriptorinfo' for more details.
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
% Usage for gb, the GraphBLAS sparse matrix constructor:
%
%   G = gb ;              empty 1-by-1 GraphBLAS double matrix
%   G = gb (X) ;          GraphBLAS copy of a matrix X, same type
%   G = gb (type) ;       empty 1-by-1 GraphBLAS matrix of the given type
%   G = gb (X, type) ;    GraphBLAS typecasted copy of matrix X
%   G = gb (m, n) ;       empty m-by-n GraphBLAS double matrix
%   G = gb (m, n, type) ; empty m-by-n GraphBLAS matrix of the given type
%
% G = gb (...) creates a new GraphBLAS sparse matrix A of the specified type.
%
% In its C-interface, SuiteSparse:GraphBLAS stores its matrices in CSR
% format, by row, since that format tends to be fastest for graph
% algorithms, but it can also use the CSC format (by column).  MATLAB
% sparse matrices are only in CSC format, and for better compatibility with
% MATLAB sparse matrices, the default format for the MATLAB interface for
% SuiteSparse:GraphBLAS is CSC.  This has performance implications, and
% algorithms should be designed accordingly.
%
% TODO allow GraphBLAS matrices to be in CSR or CSC format.
%
% The usage A = gb (m, n, type) is analgous to X = sparse (m, n), which
% creates an empty MATLAB sparse matrix X.  The type parameter is a string,
% which defaults to 'double' if not present.
%
% For the usage A = gb (X, type), X is either a MATLAB sparse matrix or a
% GraphBLAS sparse matrix object.  A is created as a GraphBLAS sparse matrix
% object that contains a copy of X, typecasted to the given type if the type
% string does not match the type of X.  If the type string is not present it
% defaults to 'double'.
% 
% Most of the valid type strings correspond to MATLAB class of the same
% name (see 'help class'), with the addition of the 'complex' type:
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
%               class name, but instead a property of a MATLAB sparse
%               double matrix.  In GraphBLAS, 'complex' is treated as a
%               type.  TODO: complex not yet implemented
%
% To free a GraphBLAS sparse matrix X, simply use 'clear X'.
%
% See also sparse.

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

    % TODO:
    %   plus        a+b
    %   minus       a-b
    %   uminus      -a
    %   uplus       +a
    %   times       a.*b
    %   mtimes      a*b
    %   rdivide     a./b
    %   ldivide     a.\b
    %   mrdivide    a/b
    %   mldivide    a\b
    %   power       a.^b
    %   mpower      a^b
    %   lt          a<b
    %   gt          a>b
    %   le          a<=b
    %   ge          a>=b
    %   ne          a~=b
    %   eq          a==b
    %   and         a&b
    %   or          a|b
    %   not         ~a
    %   ctranspose  a'
    %   transpose   a.'     note the name collision with gb.transpose
    %   horzcat     [a b]
    %   vertcat     [a ; b]
    %   subsref     a(i,j)
    %   subsasgn    a(i,j)=b
    %   subsindex   b(a)
    %   end
    %   permute
    %   reshape

    % isequal

    % diag? spdiags?

    % do not do these:
    %   colon       a:d:b, a:b
    %   char
    %   loadobj
    %   saveobj

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
        % gb (gbmxm (args)), and the typecasting methods, S = double (G), etc.
        % The output of gb is a GraphBLAS object.
        G.opaque = varargin {1} ;
    else
        %   G = gb ;              empty 1-by-1 GraphBLAS double matrix
        %   G = gb (X) ;          gb copy of a matrix X, same type
        %   G = gb (type) ;       empty 1-by-1 gb matrix of the given type
        %   G = gb (X, type) ;    gb typecasted copy of a matrix X
        %   G = gb (m, n) ;       empty m-by-n gb double matrix
        %   G = gb (m, n, type) ; empty m-by-n gb matrix of the given type
        if (isa (varargin {1}, 'gb'))
            % extract the contents of the gb object as its opaque struct so
            % the gbnew function can access it.
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

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
    % full: convert a GraphBLAS sparse matrix into a MATLAB dense matrix
    %---------------------------------------------------------------------------

    function S = full (G, identity)
    %FULL convert a GraphBLAS sparse matrix into a MATLAB dense matrix.
    % S = full (G) converts the GraphBLAS matrix G into a MATLAB sparse matrix
    % S.  It assumes the identity value is zero.  S = full (G,id) allows the
    % identity value to be specified.  No typecasting is done.
    if (nargin < 2)
        S = gbfull (G.opaque) ;
    else
        S = gbfull (G.opaque, identity) ;
    end
    end

    %---------------------------------------------------------------------------
    % double, single, etc: typecast a GraphBLAS sparse matrix to double, etc
    %---------------------------------------------------------------------------

    function G = double (X)
    %DOUBLE typecast a GraphBLAS sparse matrix to double.
    % G = double (X) typecasts the GraphBLAS sparse matrix X to double.
    G = gb (X, 'double') ;
    end

    function G = single (X)
    %SINGLE typecast a GraphBLAS sparse matrix to double.
    % G = double (X) typecasts the GraphBLAS sparse matrix X to single.
    G = gb (X, 'single') ;
    end

    function G = complex (X)
    %COMPLEX typecast a GraphBLAS sparse matrix to complex.
    % G = complex (X) typecasts the GraphBLAS sparse matrix X to complex.
    error ('complex type not yet supported') ;
    end

    function G = logical (X)
    %LOGICAL typecast a GraphBLAS sparse matrix to logical.
    % G = logical (X) typecasts the GraphBLAS sparse matrix X to logical.
    G = gb (X, 'logical') ;
    end

    function G = int8 (X)
    %INT8 typecast a GraphBLAS sparse matrix to int8.
    % G = int8 (X) typecasts the GraphBLAS sparse matrix X to int8.
    G = gb (X, 'int8') ;
    end

    function G = int16 (X)
    %INT16 typecast a GraphBLAS sparse matrix to int16.
    % G = int16 (X) typecasts the GraphBLAS sparse matrix X to int16.
    G = gb (X, 'int16') ;
    end

    function G = int32 (X)
    %INT32 typecast a GraphBLAS sparse matrix to int32.
    % G = int32 (X) typecasts the GraphBLAS sparse matrix X to int32.
    G = gb (X, 'int32') ;
    end

    function G = int64 (X)
    %INT64 typecast a GraphBLAS sparse matrix to int64.
    % G = int64 (X) typecasts the GraphBLAS sparse matrix X to int64.
    G = gb (X, 'int64') ;
    end

    function G = uint8 (X)
    %UINT8 typecast a GraphBLAS sparse matrix to uint8.
    % G = uint8 (X) typecasts the GraphBLAS sparse matrix X to uint8.
    G = gb (X, 'uint8') ;
    end

    function G = uint16 (X)
    %UINT16 typecast a GraphBLAS sparse matrix to uint16.
    % G = uint16 (X) typecasts the GraphBLAS sparse matrix X to uint16.
    G = gb (X, 'uint16') ;
    end

    function G = uint32 (X)
    %UINT32 typecast a GraphBLAS sparse matrix to uint32.
    % G = uint32 (X) typecasts the GraphBLAS sparse matrix X to uint32.
    G = gb (X, 'uint32') ;
    end

    function G = uint64 (X)
    %UINT64 typecast a GraphBLAS sparse matrix to uint64.
    % G = uint64 (X) typecasts the GraphBLAS sparse matrix X to uint64.
    G = gb (X, 'uint64') ;
    end

    %---------------------------------------------------------------------------
    % type: get the type of GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function s = type (G)
    %TYPE get the type of a GraphBLAS matrix.
    % s = type (G) returns the type of G as a string ('double', 'single',
    % 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    % 'logical', or 'complex'.  See also class.
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
        fprintf ('   default semiring: %s\n', G.semiring) ;
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
    fprintf ('\n   default semiring: %s\n', G.semiring) ;
    gbdisp (G.opaque, 2) ;
    end

    %---------------------------------------------------------------------------
    % numel: number of elements in a GraphBLAS matrix, m * n
    %---------------------------------------------------------------------------

    function result = numel (G)
    %NUMEL the maximum number of entries a GraphBLAS matrix can hold.
    % numel (G) is m*n for the m-by-n GraphBLAS matrix G.
    result = prod (size (G)) ;
    end

    %---------------------------------------------------------------------------
    % nvals: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function result = nvals (G)
    %NVALS the number of entries in a GraphBLAS matrix.
    % nvals (G) is the number of explicit entries in a GraphBLAS matrix.  Note
    % that the entries can have any value, including zero.  MATLAB drops
    % zero-valued entries from its sparse matrices.  This cannot be done in
    % GraphBLAS because of the different semirings that may be used.  In a
    % shortest-path problem, for example, and edge with weight zero is very
    % different from no edge at all.
    %
    % See also nnz, nzmax.

    result = gbnvals (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % nnz: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function result = nnz (G)
    %NNZ the number of entries in a GraphBLAS matrix.
    % nnz (G) is the same as nvals (G); some of the entries may actually be
    % explicit zero-valued entries.  See 'help gb.nvals' for more details.
    result = gbnvals (G.opaque) ;
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
    % is*: determine properties of a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function s = ismatrix (G)
    %ISMATRIX always true for any GraphBLAS matrix.
    % ismatrix (G) is always true for any GraphBLAS object G.
    s = true ;
    end

    function s = isvector (G)
    %ISVECTOR determine if the GraphBLAS matrix is a row or column vector.
    % isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) || (n == 1) ;
    end

    function s = isscalar (G)
    %ISSCALAR determine if the GraphBLAS matrix is a scalar.
    % isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) && (n == 1) ;
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
    L = gb (gbselect ('tril', G.opaque, k, struct ('kind', 'object'))) ;
    end

    %---------------------------------------------------------------------------
    % triu: lower triangular part
    %---------------------------------------------------------------------------

    function U = triu (G, k)
    %TRIL upper triangular part of a GraphBLAS matrix.
    % U = triu (G) returns the upper triangular part of G. U = triu (G,k)
    % returns the entries on and above the kth diagonal of X, where k=0 is
    % the main diagonal.
    if (nargin < 2)
        k = 0 ;
    end
    U = gb (gbselect ('triu', G.opaque, k, struct ('kind', 'object'))) ;
    end

end

%===============================================================================
methods (Static) %==============================================================
%===============================================================================

%-------------------------------------------------------------------------------
% Static methods gb.method identical to gbmethod
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
    % This function is optional for the MATLAB interface to GraphBLAS.  Simply
    % terminating the MATLAB session, or typing 'clear all' will do the same
    % thing.  However, if you are finished with GraphBLAS and wish to free its
    % internal resources, but do not wish to free everything else freed by
    % 'clear all', then use this function.  This function also clears any
    % non-default setting of gb.threads.
    %
    % See also: clear, gb.threads

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
    % TODO change 'object' to 'gb'.  Add 'full'
    %   d.kind  'object' or 'sparse'.  The default is d.kind = 'object',
    %            where the GraphBLAS operation returns an object, which is
    %            preferred since GraphBLAS sparse matrices are faster and can
    %            represent many more data types.  However, if you want a
    %            standard MATLAB sparse matrix, use d.kind='sparse'.
    %            For any operation that takes a descriptor, the following uses
    %            are the same, but the latter is faster:
    %            S = gb.operation (..., d) with d.kind='sparse'
    %            S = sparse (gb.operation (...))) with no d or d.kind='object'
    %
    % These are scalar values:
    %
    %   d.nthreads  max # of threads to use; default is omp_get_max_threads.
    %   d.chunk     controls # of threads to use for small problems.
    %
    % This function simply lists the contents of a GraphBLAS descriptor and
    % checks if its contents are valid.
    %
    % Refer to the SuiteSparse:GraphBLAS User Guide for more details.
    %
    % See also gb, gb.unopinfo, gb.binopinfo, gb.semiringinfo.

    if (nargin == 0)
        gbdescriptorinfo ;
    else
        gbdescriptorinfo (d) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.binopinfo: list the details of a GraphBLAS binary operator
    %---------------------------------------------------------------------------

    function binopinfo (s, type)
    %GB.BINOP list the details of a GraphBLAS binary operator, for illustration
    %
    % Usage
    %
    %   gb.binopinfo (op)
    %   gb.binopinfo (op, type)
    %
    % For the first usage, the op must be a string of the form 'op.type', where
    % 'op'.  The second usage allows the type to be omitted from the first
    % argument, as just 'op'.  This is valid for all GraphBLAS operations,
    % since the type defaults to the type of the input matrices.  However, this
    % function does not have a default type and thus one must be provided,
    % either in the op as gb.binopinfo ('+.double'), or in the second argument,
    % gb.binopinfo ('+', 'double').
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
    %   gb.binopinfo ('+.*.double') ;
    %   gb.binopinfo ('min.1st.int32') ;
    %
    %   % invalid binary operator (generates an error); this is a unary op:
    %   gb.binopinfo ('abs.double') ;
    %
    % gb.binopinfo generates an error for an invalid op, so user code can test
    % the validity of an op with the MATLAB try/catch mechanism.
    %
    % See also gb, gb.unopinfo, gb.semiringinfo, gb.descriptorinfo.

    if (nargin < 2)
        gbbinopinfo (s) ;
    else
        gbbinopinfo (s, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.semiringinfo: list the details of a GraphBLAS semiring
    %---------------------------------------------------------------------------

    function semiringinfo (s, type)
    %GB.SEMIRING list the details of a GraphBLAS semiring, for illustration only
    %
    % Usage
    %
    %   gb.semiringinfo (semiring)
    %   gb.semiringinfo (semiring, type)
    %
    % For the first usage, the semiring must be a string of the form
    % 'add.mult.type', where 'add' and 'mult' are binary operators.  The second
    % usage allows the type to be omitted from the first argument, as just
    % 'add.mult'.  This is valid for all GraphBLAS operations, since the type
    % defaults to the type of the input matrices.  However, this function does
    % not have a default type and thus one must be provided, either in the
    % semiring as gb.semiringinfo ('+.*.double'), or in the second argument,
    % gb.semiringinfo ('+.*', 'double').
    %
    % The add operator must be a valid monoid, typically the operators plus,
    % times, min, max, or, and, ne, xor.  The binary operator z=f(x,y) of a
    % monoid must be associate and commutative, with an identity value id such
    % that f(x,id) = f(id,x) = x.  Furthermore, the types of x, y, and z must
    % all be the same.  Thus, the '<.double' is not a valid operator for a
    % monoid, since its output type (logical) does not match its inputs
    % (double).  Thus, <.*.double is not a valid semiring.
    %
    % However, many of the binary operators are equivalent.  xor(x,y) is the
    % same as minus(x,y), for example, and thus 'minus.&.logical' is the same
    % semiring as as 'xor.&.logical', and both strings refer to the same valid
    % semiring.
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

    if (nargin < 2)
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
    % threads.  GraphBLAS may use fewer threads, if the problem is small.  The
    % setting persists for the current MATLAB session, or until 'clear all' is
    % used, at which point the setting reverts to the default number of
    % threads.
    %
    % MATLAB can detect the number of physical and logical cores via an
    % undocumented builtin function: ncores = feature('numcores') ; or via
    % maxNumCompThreads.
    %
    % Example:
    %
    %   feature ('numcores') ;          % print info about cores
    %   ncores = feature ('numcores') ; % get # of logical cores MATLAB uses
    %   ncores = maxNumCompThreads ;    % same as feature ('numcores')
    %   gb.threads (2*ncores) ;         % GraphBLAS will use <= 2*ncores threads
    %
    % TODO add chunk?
    %
    % See also feature, maxNumCompThreads.

    nthreads = gbthreads (varargin {:}) ;
    end

%-------------------------------------------------------------------------------
% Static methods that return a GraphBLAS matrix object or a MATLAB sparse matrix
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

    [args Cout_is_object] = get_args (varargin {:}) ;
    if (Cout_is_object)
        Cout = gb (gbmxm (args {:})) ;
    else
        Cout = gbmxm (args {:}) ;
    end
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
    % dup is a string that defines a binary function; see 'help gb.binopinfo' for a
    % list of available binary operators.  The dup operator need not be
    % associative.  If two entries in [I J X] have the same row and column
    % index, the dup operator is applied to assemble them into a single entry.
    % Suppose (i,j,x1), (i,j,x2), and (i,j,x3) appear in that order in [I J X],
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
    % uint64 arrays, in which case they are treated as 0-based.  Entries in I
    % are the range 0 to m-1, and J are in the range 0 to n-1.  If I, J, and X
    % are double, the following examples construct the same MATLAB sparse
    % matrix S:
    %
    %   S = sparse (I, J, X) ;
    %   S = sparse (gb.build (I, J, X)) ;
    %   S = sparse (gb.build (uint64(I)-1, uint64(J)-1, X)) ;
    %
    % Using uint64 integers for I and J is faster and uses less memory.  I and
    % J need not be in any particular order, but gb.build is fastest if I and J
    % are provided in column-major order.
    %
    % See also sparse (with 3 or more input arguments).

    [args Cout_is_object] = get_args (varargin {:}) ;
    if (Cout_is_object)
        G = gb (gbbuild (args {:})) ;
    else
        G = gbbuild (args {:}) ;
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

    [args Cout_is_object] = get_args (varargin {:}) ;
    if (Cout_is_object)
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
    %
    % C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')
    %
    % where A can be a matrix or a scalar.
    %
    % Usage:
    %
    %   Cout = gb.assign (Cin, M, accum, A, I, J, desc)
    %
    % Cin and A are required parameters.  All others are optional.
    %
    % desc: The descriptor determines if ~M or M is used (d.mask = 'default' or
    %       'complement'), if replace option is used (d.out = 'default' or
    %       'replace'), and whether or not A is transposed (d.in0 = 'default'
    %       or 'transpose').  The d.kind is 'object' or 'sparse', to define how
    %       Cout is to be constructed.  
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

    [args Cout_is_object] = get_args (varargin {:}) ;
    if (Cout_is_object)
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
    % TODO
    % See also gb.assign, subsasgn.

    error ('gb.subassign not yet implemented') ;
    end

    function Cout = colassign (varargin)
    %GB.COLASSIGN sparse matrix assignment to a single column
    %   Cout = gb.colassign (Cin, M, accum, u, I, j, desc)
    % TODO
    error ('gb.colassign not yet implemented') ;
    end

    function Cout = rowassign (varargin)
    %GB.ROWASSIGN sparse matrix assignment to a single row
    %   Cout = gb.rowassign (Cin, M, accum, u, i, J, desc)
    % TODO
    error ('gb.rowassign not yet implemented') ;
    end

    %---------------------------------------------------------------------------
    % gb.find: extract all entries from a matrix
    %---------------------------------------------------------------------------

    function [I,J,X] = find (A, onebased)
    %GB.FIND extract a list of entries from a matrix
    %
    % Usage:
    %
    %   [I J X] = gb.find (A)        % extract 1-based indices; I and J double
    %   [I J X] = gb.find (A, 0) ;   % extract 0-based indices; I and J uint64
    %   [I J X] = gb.find (A, 1) ;   % extract 1-based indices; I and J double
    %
    % gb.find extracts all entries from either a MATLAB matrix A or a GraphBLAS
    % matrix A.  If A is a MATLAB sparse matrix, [I J X] = gb.find (A) is
    % identical to [I J X] = find (A).
    %
    % An optional second argument determines the type of I and J.  It defaults
    % to 1, and in this case, I and J are double, and reflect 1-based indices,
    % just like the MATLAB statement [I J X] = find (A).  If zero, then I and J
    % are returned as uint64 arrays, containing 0-based indices.
    %
    % This function corresponds to the GrB_*_extractTuples_* functions in
    % GraphBLAS.
    %
    % See also find.

    error ('gb.find not yet implemented') ;
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

    error ('gb.reduce not yet implemented') ;
    end

    %---------------------------------------------------------------------------
    % gb.kron: Kronecker product
    %---------------------------------------------------------------------------

    function Cout = kron (varargin)
    %GB.KRON sparse Kronecker product
    %
    % Usage:
    %
    %   Cout = gb.kron (Cin, M, accum, op, A, B, desc)
    %
    % TODO

    error ('gb.kron not yet implemented') ;
    end

    %---------------------------------------------------------------------------
    % gb.transpose: transpose a matrix
    %---------------------------------------------------------------------------

    function Cout = transpose (varargin)
    %GB.TRANSPOSE transpose a matrix
    %
    % Usage:
    %
    %   Cout = gb.transpose (Cin, M, accum, A, desc)
    %
    % See also transpose.

    error ('gb.transpose not yet implemented') ;
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
    % TODO

    error ('gb.eadd not yet implemented') ;
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
    % TODO

    error ('gb.emult not yet implemented') ;
    end

    %---------------------------------------------------------------------------
    % gb.apply: appyl a unary operator to entries in a matrix
    %---------------------------------------------------------------------------

    function Cout = apply (varargin)
    %GB.APPLY apply a unary operator to a sparse matrix
    %
    % Usage:
    %
    %   Cout = gb.apply (Cin, M, accum, op, A, desc)
    %
    % TODO

    error ('gb.apply not yet implemented') ;
    end

    function Cout = extract (varargin)
    %GB.EXTRACT extract sparse submatrix
    %
    % Usage:
    %
    %   Cout = gb.extract (Cin, M, accum, A, I, J, desc)
    %
    % TODO
    %
    % See also subsref.

    error ('gb.extract not yet implemented') ;
    end

end
end

%===============================================================================
% local functions ==============================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % get_args: get the arguments, including the descriptor and check d.kind
    %---------------------------------------------------------------------------

    function [args Cout_is_object] = get_args (varargin)
    %GET_ARGS get the arguments, including the descriptor and check d.kind.
    %
    % Any input arguments that are GraphBLAS sparse matrix objects are replaced
    % with the struct arg.opaque so that they can be passed to the underlying
    % mexFunction.
    %
    % Next, the descriptor is modified to change the default d.kind.
    %
    % The default outside gb.m is d.kind = 'sparse', but inside gb.m the
    % default is modified to d.kind = 'object', by adjusting the descriptor.
    % If the descriptor d is not present, then it is created and appended to
    % the argument list, with d.kind = 'object'.  If the descriptor is present
    % and d.kind does not appear, then d.kind = 'object' is set.  Finally,
    % Cout_is_object is set true if d.kind is 'object'.  If d.kind is 'object',
    % then the underlying mexFunction returns a GraphBLAS sparse matrix struct,
    % which is then converted above to a GraphBLAS sparse matrix object, with
    % Cout_is_object true.  See for example G = gb (gbmxm (args {:})) above.

    % get the args and extract any GraphBLAS matrix structs
    args = varargin ;
    for k = 1:length (args)
        if (isa (args {k}, 'gb'))
            args {k} = args {k}.opaque ;
        end
    end

    % find the descriptor
    Cout_is_object = false ;
    if (length (args) > 0)
        % get the last input argument and see if it is a GraphBLAS descriptor
        d = args {end} ;
        if (isstruct (d) && ~isfield (d, 'GraphBLAS'))
            % found the descriptor.  If it does not have d.kind, add it.
            if (~isfield (d, 'kind'))
                args {end}.kind = 'object' ;
                Cout_is_object = true ;
            else
                Cout_is_object = isequal (d.kind, 'object') ;
            end
        else
            % the descriptor is not present; add it
            d = struct ('kind', 'object') ;
            d.kind = 'object' ;
            args {end+1} = d ;
            Cout_is_object = true ;
        end
    end
    end

