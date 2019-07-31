classdef gb
%GB a GraphBLAS sparse matrix object
%
% TODO discuss here
%
% Usage:
%
%   G = gb ;               empty 1-by-1 GraphBLAS double matrix
%   G = gb (X) ;           GraphBLAS struct of a MATLAB sparse X, same type
%   G = gb (type) ;        empty 1-by-1 GraphBLAS matrix of the given type
%   G = gb (X, type) ;     GraphBLAS typecasted copy of a MATLAB sparse X
%   G = gb (m, n) ;        empty m-by-n GraphBLAS double matrix
%   G = gb (m, n, type) ;  empty m-by-n GraphBLAS matrix of the given type
%
% The input X may be a MATLAB sparse matrix or a GraphBLAS sparse
% matrix object. TODO allow X to be a MATLAB dense matrix.
%
% See 'help gbnew' for details. (TODO put the details here instead ...)
%
% See also 'help gb.method', for each method listed below.  For example 'help
% gb.sparse' describes A = sparse (G) for a GraphBLAS matrix G.
%
% Methods:
%
%   gb          construct a GraphBLAS sparse matrix object
%   sparse      convert a GraphBLAS matrix to a MATLAB sparse matrix
%   disp        dislpay the contents of a GraphBLAS matrix
%   ...
%
% Static methods:
%
%   gb.build       construct a GraphBLAS matrix from a list of entries
%   gb.clear       clear internal GraphBLAS workspace and settings
%   gb.threads     get/set the number of threads to use in GraphBLAS
%   gb.descriptor  list the contents of a GraphBLAS descriptor
%   ... TODO

% TODO: do I make gbmxm, gbbuild, etc all private?

properties (SetAccess = private, GetAccess = private, Hidden = true)
    % the object properties are a single struct, containing the opaque content
    % of a GraphBLAS GrB_Matrix
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

    %---------------------------------------------------------------------------
    % gb: construct a GraphBLAS sparse matrix object
    %---------------------------------------------------------------------------

    function G = gb (varargin)
    %GB construct a GraphBLAS sparse matrix object

    % X may also be GraphBLAS struct as returned by another gb* function,
    % but this usage is not meant for the end-user, but only used here.
    % See for example mxm below, which uses G = gb (gbmxm (args)),
    % and the typecasting methods, S = double (G), etc.

    if (nargin == 0)
        % return an empty GraphBLAS object
        G.opaque = gbnew ;
    elseif (nargin == 1 && ...
        (isstruct (varargin {1}) && isfield (varargin {1}, 'GraphBLAS')))
        % input is a GraphBLAS struct; return it as an object
        G.opaque = varargin {1} ;
    else
        if (isa (varargin {1}, 'gb'))
            varargin {1} = varargin {1}.opaque ;
        end
        G.opaque = gbnew (varargin {:}) ;
    end
    end

    %---------------------------------------------------------------------------
    % sparse: convert a GraphBLAS sparse matrix into a MATLAB sparse matrix
    %---------------------------------------------------------------------------

    function S = sparse (G)
    %SPARSE convert a GraphBLAS matrix into a MATLAB sparse matrix
    % Usage:
    %   S = sparse (G)
    % See 'help gbsparse' for more details.
    S = gbsparse (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % double, single, etc: typecast a GraphBLAS sparse matrix to double, etc
    %---------------------------------------------------------------------------

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

    %---------------------------------------------------------------------------
    % type: get the type of GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function s = type (G)
    %TYPE get the type of a GraphBLAS matrix
    s = gbtype (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % disp: display the contents of a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function disp (G, level)
    %DISP display the contents of a GraphBLAS object.
    % Usage:
    %     gb.disp (G, level)
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

    %---------------------------------------------------------------------------
    % display: display the contents of a GraphBLAS matrix.
    %---------------------------------------------------------------------------

    function display (G)
    %DISPLAY display the contents of a GraphBLAS object.
    name = inputname (1) ;
    if (~isempty (name))
        fprintf ('\n%s =\n', name) ;
    end
    gbdisp (G.opaque, 3) ;
    end

    %---------------------------------------------------------------------------
    % numel: number of elements in a GraphBLAS matrix, m * n
    %---------------------------------------------------------------------------

    function result = numel (G)
    result = prod (size (G)) ;
    end

    %---------------------------------------------------------------------------
    % nvals: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function result = nvals (G)
    result = gbnvals (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % nnz: number of entries in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function nvals = nnz (G)
    nvals = gbnvals (G.opaque) ;
    end

    %---------------------------------------------------------------------------
    % size: number of rows and columns in a GraphBLAS matrix
    %---------------------------------------------------------------------------

    function [m n] = size (G)
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
    %CLEAR clear all internal GraphBLAS workspace
    % Usage: gb.clear
    % See 'help gbclear' for more details.
    gbclear ;
    end

    %---------------------------------------------------------------------------
    % gb.descriptor: list the contents of a GraphBLAS descriptor
    %---------------------------------------------------------------------------

    function descriptor (d)
    %DESCRIPTOR list the contents of a GraphBLAS descriptor
    % Usage:
    %   gb.descriptor
    %   gb.descriptor (d)
    % See 'help gbdescriptor' for more details.
    if (nargin == 0)
        gbdescriptor ;
    else
        gbdescriptor (d) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.binop: list the details of a GraphBLAS binary operator
    %---------------------------------------------------------------------------

    function binop (s, type)
    %BINOP list the details of a GraphBLAS binary operator.
    % Usage:
    %   gb.binop (s)
    %   gb.binop (s, type)
    % See 'help gbbinop' for more details.
    if (nargin < 2)
        gbbinop (s) ;
    else
        gbbinop (s, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.semiring: list the details of a GraphBLAS semiring
    %---------------------------------------------------------------------------

    function semiring (s, type)
    %SEMIRING list the details of a GraphBLAS semiring.
    % Usage:
    %   gb.semiring (s)
    %   gb.semiring (s, type)
    % See 'help gbsemiring' for more details.
    if (nargin < 2)
        gbsemiring (s) ;
    else
        gbsemiring (s, type) ;
    end
    end

    %---------------------------------------------------------------------------
    % gb.threads: get/set the # of threads to use in GraphBLAS
    %---------------------------------------------------------------------------

    function nthreads = threads (varargin)
    %THREADS set/get the # of threads to be used in GraphBLAS
    % Usage:
    %   gb.nthreads (t)
    %   t = gb.nthreads ;
    % See 'help gbnthreads' for more details.
    nthreads = gbthreads (varargin {:}) ;
    end

%-------------------------------------------------------------------------------
% Static methods that return a GraphBLAS matrix object or a MATLAB sparse matrix
%-------------------------------------------------------------------------------

    %---------------------------------------------------------------------------
    % gb.empty: construct an empty GraphBLAS sparse matrix
    %---------------------------------------------------------------------------

    function G = empty (arg1, arg2)
    %EMPTY construct an empty GraphBLAS sparse matrix
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

    function G = mxm (varargin)
    %MXM: sparse matrix-matrix multiply in GraphBLAS
    % Usage:
    %
    %   Cout = gb.mxm (semiring, A, B)
    %   Cout = gb.mxm (semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, accum, semiring, A, B)
    %   Cout = gb.mxm (Cin, accum, semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, Mask, semiring, A, B)
    %   Cout = gb.mxm (Cin, Mask, semiring, A, B, desc)
    %
    %   Cout = gb.mxm (Cin, Mask, accum, semiring, A, B)
    %   Cout = gb.mxm (Cin, Mask, accum, semiring, A, B, desc)
    %
    % If desc.kind = 'object' then Cout is returned as GraphBLAS sparse matrix
    % object, which is the default for gb.mxm.  If desc.kind = 'sparse' then
    % Cout is returned as a MATLAB sparse matrix.
    %
    % See 'help gbmxm' for more details.

        [args G_is_object] = get_args (varargin {:}) ;
        if (G_is_object)
            G = gb (gbmxm (args {:})) ;
        else
            G = gbmxm (args {:}) ;
        end
    end

    %---------------------------------------------------------------------------
    % gb.build: build a GraphBLAS sparse matrix from a list of entries
    %---------------------------------------------------------------------------

    function G = build (varargin)
    %BUILD build a GraphBLAS sparse matrix object from a list of entries
    % Usage:
    %
    %   G = gb.build (I, J, X, m, n, dup, type, desc)
    %
    % If desc.kind = 'object', G is returned as a GraphBLAS sparse matrix,
    % which is the default for gb.build.  If desc.kind = 'sparse' then G is
    % returned as a MATLAB sparse matrix.
    %
    % See 'help gbbuild' for more details.

        [args G_is_object] = get_args (varargin {:}) ;
        if (G_is_object)
            G = gb (gbbuild (args {:})) ;
        else
            G = gbbuild (args {:}) ;
        end
    end

end
end

%===============================================================================
% local functions ==============================================================
%===============================================================================

    %---------------------------------------------------------------------------
    % get_args: get the arguments, including the descriptor and check d.kind
    %---------------------------------------------------------------------------

    function [args G_is_object] = get_args (varargin)
    %GET_ARGS get the arguments, including the descriptor and check d.kind.
    %
    % Any input arguments that are GraphBLAS sparse matrix objects are
    % replaced with the struct arg.opaque so that they can be passed to the
    % underlying mexFunction.
    %
    % Next, the descriptor is modified to change the default d.kind.
    %
    % The default outside gb.m is d.kind = 'sparse', but inside gb.m the
    % default is modified to d.kind = 'object', by adjusting the descriptor.
    % If the descriptor d is not present, then it is created and appended to
    % the argument list, with d.kind = 'object'.  If the descriptor is present
    % and d.kind does not appear, then d.kind = 'object' is set.  Finally,
    % G_is_object is set true if d.kind is 'object'.  If d.kind is 'object',
    % then the underlying mexFunction returns a GraphBLAS sparse matrix struct,
    % which is then converted above to a GraphBLAS sparse matrix object, with
    % G_is_object true.  See for example G = gb (gbmxm (args {:})) above.

    % get the args and extract any GraphBLAS matrix structs
    args = varargin ;
    for k = 1:length (args)
        if (isa (args {k}, 'gb'))
            args {k} = args {k}.opaque ;
        end
    end

    % find the descriptor
    G_is_object = false ;
    if (length (args) > 0)
        % get the last input argument and see if it is a GraphBLAS descriptor
        d = args {end} ;
        if (isstruct (d) && ~isfield (d, 'GraphBLAS'))
            % found the descriptor.  If it does not have d.kind, add it.
            if (~isfield (d, 'kind'))
                args {end}.kind = 'object' ;
                G_is_object = true ;
            else
                G_is_object = isequal (d.kind, 'object') ;
            end
        else
            % the descriptor is not present; add it
            d = struct ('kind', 'object') ;
            d.kind = 'object' ;
            args {end+1} = d ;
            G_is_object = true ;
        end
    end
    end

