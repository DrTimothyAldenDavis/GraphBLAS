function [opname optype ztype xtype ytype] = GB_spec_operator (op,optype_default)
%GB_SPEC_OPERATOR get the contents of an operator
%
% On input, op can be a struct with a string op.opname that gives the operator
% name, and a string op.optype with the operator class.  Alternatively, op can
% be a string with the operator name, in which case the operator class is given
% by optype_default.
%
% The optype determines the class in the inputs x and y for z=op(x,y); the
% class of the output is ztype, and it is either the same as x and y, or
% logical for 'eq', 'ne', 'gt', 'lt', 'ge', 'le'.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isempty (op))
    % No operator has been defined; return an empty operator.  GB_spec_accum
    % uses this condition just like the (accum == NULL) condition in the C
    % version of GraphBLAS.  It means C<Mask>=T is to be done instead of
    % C<Mask>=accum(C,T).
    opname = '' ;
    optype = '' ;
elseif (isstruct (op))
    % op is a struct with opname and optype
    opname = op.opname ;
    optype = op.optype ;
else
    % op is a string, use the default optype unless the op is just logical
    opname = op ;
    optype = optype_default ;
end

% xtype is always the optype
xtype = optype ;

% ytype is usually the optype, except for bitshift
ytype = optype ;

% ztype is usually the optype, except for the cases below
ztype = optype ;

% get the x,y,z types of the operator
switch opname

    % binary ops
    case { 'eq', 'ne', 'gt', 'lt', 'ge', 'le', 'isinf', 'isnan', 'isfinite' }
        ztype = 'logical' ;
    case { 'bitshift' }
        ytype = 'int8' ;
    case { 'cmplx' }
        if (isequal (optype, 'single'))
            ztype = 'single complex' ;
        else
            ztype = 'double complex' ;
        end

    % unary ops
    case { 'abs', 'creal', 'cimag', 'carg', 'real', 'imag', 'angle' }
        if (isequal (optype, 'single complex'))
            ztype = 'single' ;
        elseif (isequal (optype, 'double complex'))
            ztype = 'double' ;
        end
    case { 'isinf', 'isnan', 'isfinite' }
        ztype = 'logical' ;

    otherwise
        % for all other operators, the ztype is the same as optype

end


