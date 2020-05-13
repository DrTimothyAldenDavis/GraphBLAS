function [opname optype zclass] = GB_spec_operator (op, optype_default)
%GB_SPEC_OPERATOR get the contents of an operator
%
% On input, op can be a struct with a string op.opname that gives the operator
% name, and a string op.optype with the operator class.  Alternatively, op can
% be a string with the operator name, in which case the operator class is given
% by optype_default.
%
% The optype determines the class in the inputs x and y for z=op(x,y); the
% class of the output is zclass, and it is either the same as x and y, or
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

% get the class of the output of the operator
switch opname
    case 'eq'
        zclass = 'logical' ;
    case 'ne'
        zclass = 'logical' ;
    case 'gt'
        zclass = 'logical' ;
    case 'lt'
        zclass = 'logical' ;
    case 'ge'
        zclass = 'logical' ;
    case 'le'
        zclass = 'logical' ;
    otherwise
        % for all other operators, the zclass is the same as optype
        zclass = optype ;
end


