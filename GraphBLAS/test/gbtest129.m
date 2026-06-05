function gbtest129
%GBTEST129 test GrB.jit

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('gbtest129: testing GrB.jit\n') ;

[status1, path1] = GrB.jit ;
fprintf ('JIT: %s at %s\n', status1, path1) ;
GrB.jit ('off', '/tmp') ;
[status2, path2] = GrB.jit ;
assert (isequal (status2, 'off')) ;
assert (isequal (path2, '/tmp')) ;

[status3, path3] = GrB.jit (status1, path1) ;
assert (isequal (status1, status3)) ;
assert (isequal (path1, path3)) ;

try
    GrB.jit (0,0)
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (isequal (msg, 'status must be a string')) ;

try
    GrB.jit ('on',0)
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (isequal (msg, 'path must be a string')) ;

fprintf ('\ngbtest129: all tests passed\n') ;

