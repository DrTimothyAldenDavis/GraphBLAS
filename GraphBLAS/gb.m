classdef gb
%GB a GraphBLAS sparse matrix object
%
% TODO discuss here
%
% See also 'help gb.method', for each method listed below.  For example
% 'help gb.gb' displays help on how GraphBLAS objects are created with
% G = gb (args).  'help gb.sparse' describes A = sparse (G) for a
% GraphBLAS matrix G.
%
% Methods:
%
%   gb          construct a GraphBLAS matrix from a MATLAB sparse matrix
%   sparse      convert a GraphBLAS matrix to a MATLAB sparse matrix
%   disp        dislpay the contents of a GraphBLAS matrix
%   build       construct a GraphBLAS matrix from a list of entries
%   clear       clear internal GraphBLAS workspace and settings
%   descriptor  list the contents of a GraphBLAS descriptor
%   threads     get/set the number of threads to use in GraphBLAS
%   ...
%

properties (SetAccess = private, GetAccess = private, Hidden = true)
    % the object properties are a single struct, containing the opaque content
    % of a GraphBLAS GrB_Matrix
    opaque = [ ] ;
end

%-------------------------------------------------------------------------------
methods
%-------------------------------------------------------------------------------

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
    %   tranpose    a.'
    %   horzcat     [a b]
    %   vertcat     [a ; b]
    %   subsref     a(i,j)
    %   subsasgn    a(i,j)=b
    %   subsindex   b(a)
    %   end
    %   permute
    %   reshape

    % do not do these:
    %   colon       a:d:b, a:b
    %   char
    %   loadobj
    %   saveobj

    %--------------------------------------------------------------------------
    % gb: construct a GraphBLAS sparse matrix object
    %--------------------------------------------------------------------------

    function G = gb (varargin)
    %GB construct a GraphBLAS sparse matrix object.
    % Usage: ...
    % See 'help gbnew' for details. (TODO add the details here...)
    if (nargin == 0)
        G.opaque = gbnew ;
    else
        if (isa (varargin {1}, 'gb'))
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

    %--------------------------------------------------------------------------
    % gb.sparse: convert a GraphBLAS sparse matrix into a MATLAB sparse matrix
    %--------------------------------------------------------------------------

    function S = sparse (G)
    %SPARSE convert a GraphBLAS matrix into a MATLAB sparse matrix
    % See 'help gbsparse' for more details. TODO
    S = gbsparse (G.opaque) ;
    end

    %--------------------------------------------------------------------------
    % gb.double, etc: typecast a GraphBLAS sparse matrix to double, etc
    %--------------------------------------------------------------------------

    function A = double (G)
    %DOUBLE typecast a GraphBLAS sparse matrix to double
    A = gb (G, 'double') ;
    end

    function A = single (G)
    %SINGLE typecast a GraphBLAS sparse matrix to single
    A = gb (G, 'single') ;
    end

    function A = complex (G)
    %COMPLEX typecast a GraphBLAS sparse matrix to complex
    error ('complex type not yet supported') ;
    end

    function A = logical (G)
    %LOGICAL typecast a GraphBLAS sparse matrix to logical
    A = gb (G, 'logical') ;
    end

    function A = int8 (G)
    %INT8 typecast a GraphBLAS sparse matrix to int8
    A = gb (G, 'int8') ;
    end

    function A = int16 (G)
    %INT16 typecast a GraphBLAS sparse matrix to int16
    A = gb (G, 'int16') ;
    end

    function A = int32 (G)
    %INT32 typecast a GraphBLAS sparse matrix to int32
    A = gb (G, 'int32') ;
    end

    function A = int64 (G)
    %INT64 typecast a GraphBLAS sparse matrix to int64
    A = gb (G, 'int64') ;
    end

    function A = uint8 (G)
    %UINT8 typecast a GraphBLAS sparse matrix to uint8
    A = gb (G, 'uint8') ;
    end

    function A = uint16 (G)
    %UINT16 typecast a GraphBLAS sparse matrix to uint16
    A = gb (G, 'uint16') ;
    end

    function A = uint32 (G)
    %UINT32 typecast a GraphBLAS sparse matrix to uint32
    A = gb (G, 'uint32') ;
    end

    function A = uint64 (G)
    %UINT64 typecast a GraphBLAS sparse matrix to uint64
    A = gb (G, 'uint64') ;
    end

    %--------------------------------------------------------------------------
    % gb.type: get the type of GraphBLAS sparse matrix
    %--------------------------------------------------------------------------

    function s = type (G)
    %TYPE get the type of a GraphBLAS matrix
    s = gbtype (G.opaque) ;
    end

    %--------------------------------------------------------------------------
    % gb.disp: display the contents of a GraphBLAS matrix
    %--------------------------------------------------------------------------

    function disp (G, level)
    %DISP display the contents of a GraphBLAS object.
    % Usage: G.disp (G, level)
    % See 'help gbdisp' for more details. (TODO more here)
    if (nargin < 2)
        level = 3 ;
    end
    if (level > 0)
        name = inputname (1) ;
        if (~isempty (name))
            fprintf ('\n%s =\n', name) ;
        end
    end
    gbdisp (G.opaque, level) ;
    end

    %--------------------------------------------------------------------------
    % gb.display: display the contents of a GraphBLAS matrix.
    %--------------------------------------------------------------------------

    function display (G)
    %DISPLAY display the contents of a GraphBLAS object.
    name = inputname (1) ;
    if (~isempty (name))
        fprintf ('\n%s =\n', name) ;
    end
    gbdisp (G.opaque, 3) ;
    end

    %--------------------------------------------------------------------------
    % gb.numel: number of entries in a GraphBLAS matrix
    %--------------------------------------------------------------------------

    function nvals = numel (G)
    nvals = gbnvals (G.opaque) ;
    end

    function nvals2 = nvals (G)
    nvals2 = gbnvals (G.opaque) ;
    end

    %--------------------------------------------------------------------------
    % gb.nnz: number of entries in a GraphBLAS matrix
    %--------------------------------------------------------------------------

    function nvals = nnz (G)
    nvals = gbnvals (G.opaque) ;
    end

    %--------------------------------------------------------------------------
    % gb.size: number of rows and columns in a GraphBLAS matrix
    %--------------------------------------------------------------------------

    function [arg1, n] = size (G)
    if (nargout <= 1)
        arg1 = gbsize (G.opaque) ;
    else
        [arg1, n] = gbsize (G.opaque) ;
    end
    end

    %--------------------------------------------------------------------------
    % gb.is*: determine properties of a GraphBLAS matrix
    %--------------------------------------------------------------------------

    function s = ismatrix (G)
    s = true ;
    end

    function s = isvector (G)
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) || (n == 1) ;
    end

    function s = isscalar (G)
    [m, n] = gbsize (G.opaque) ;
    s = (m == 1) && (n == 1) ;
    end

end

%-------------------------------------------------------------------------------
methods (Static)
%-------------------------------------------------------------------------------

    %--------------------------------------------------------------------------
    % gb.empty: construct an empty GraphBLAS matrix
    %--------------------------------------------------------------------------

    function G = empty (arg1, arg2)
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

    %--------------------------------------------------------------------------
    % gb.build: build a GraphBLAS sparse matrix from a list of entries
    %--------------------------------------------------------------------------

    function G = build (I, J, X, varargin)
    %BUILD build a GraphBLAS sparse matrix from a list of entries
    % Usage: G = build (I, J, X, m, n, dup, type)
    % m and n default to the largest ... TODO
    G = gb (gbbuild (I, J, X, varargin {:})) ;
    end

    %--------------------------------------------------------------------------
    % gb.clear: clear all internal GraphBLAS workspace and settings
    %--------------------------------------------------------------------------

    function clear
    %CLEAR clear all internal GraphBLAS workspace
    gbclear ;
    end

    %--------------------------------------------------------------------------
    % gb.descriptor: list the contents of a GraphBLAS descriptor
    %--------------------------------------------------------------------------

    function descriptor (d)
    %DESCRIPTOR list the contents of a GraphBLAS descriptor
    if (nargin == 0)
        gbdescriptor ;
    else
        gbdescriptor (d) ;
    end
    end

    %--------------------------------------------------------------------------
    % gb.binop: list the details of a GraphBLAS binary operator
    %--------------------------------------------------------------------------

    function binop (s, type)
    if (nargin < 2)
        gbbinop (s) ;
    else
        gbbinop (s, type) ;
    end
    end

    %--------------------------------------------------------------------------
    % gb.semiring: list the details of a GraphBLAS semiring
    %--------------------------------------------------------------------------

    function semiring (s, type)
    if (nargin < 2)
        gbsemiring (s) ;
    else
        gbsemiring (s, type) ;
    end
    end

    %--------------------------------------------------------------------------
    % gb.threads: get/set the # of threads to use in GraphBLAS
    %--------------------------------------------------------------------------

    function nthreads = threads (varargin)
    %THREADS set/get the # of threads used in GraphBLAS
    nthreads = gbthreads (varargin {:}) ;
    end

    %--------------------------------------------------------------------------
    % gb.mxm: sparse matrix-matrix multiply
    %--------------------------------------------------------------------------

    % TODO put gbmxm in ./private?

    function G = mxm (varargin)
    %MXM: sparse matrix-matrix multiply in GraphBLAS
    % Usage:
    %   Cout = gbmxm (semiring, A, B)
    %   Cout = gbmxm (semiring, A, B, desc)
    %   Cout = gbmxm (Cin, accum, semiring, A, B)
    %   Cout = gbmxm (Cin, accum, semiring, A, B, desc)
    %   Cout = gbmxm (Cin, Mask, semiring, A, B)
    %   Cout = gbmxm (Cin, Mask, semiring, A, B, desc)
    %   Cout = gbmxm (Cin, Mask, accum, semiring, A, B)
    %   Cout = gbmxm (Cin, Mask, accum, semiring, A, B, desc)
    %
    % See 'help gbmxm' for more details.
    for k = 1:nargin
        if (isa (varargin {k}, 'gb'))
            varargin {k} = varargin {k}.opaque ;
        end
    end
    % TODO allow the descriptor to specify that G = gbsparse (...) instead?
    G = gb (gbmxm (varargin {:})) ;
    end

end
end

