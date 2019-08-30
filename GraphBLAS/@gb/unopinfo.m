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

