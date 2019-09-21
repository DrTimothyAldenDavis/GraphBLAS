function gbtest53
%GBTEST53 test gb.monoidinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

types10 = {
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    } ;

ops = { '+', '*', 'min', 'max' } ;
logical_ops = { '|', '&', 'xor', 'eq' } ;

for k1 = 1:4
    op = ops {k1} ;
    fprintf ('\nop [ %s ]=============================================\n', op) ;
    for k2 = 1:10
        type = types10 {k2} ;
        gb.monoidinfo ([op '.' type]) ;
        gb.monoidinfo (op, type) ;
    end
end

for k1 = 1:4
    op = logical_ops {k1} ;
    fprintf ('\nop [ %s ]=============================================\n', op) ;
    gb.monoidinfo ([op '.logical']) ;
    gb.monoidinfo (op, 'logical') ;
end

fprintf ('\n\n') ;
gb.monoidinfo

fprintf ('gbtest53: all tests passed\n') ;

