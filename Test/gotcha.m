
clear all
make
debug_on
nthreads_set (4,1)
gb

%   testca  % mxm, mxv, vxm
%    testcc  % transpose

debug_off ; nthreads_set (2, 4096) ; test19b
debug_off ; nthreads_set (4, 1)    ; test19b
debug_on ;

% test18
test75
test06
test19b
test20

test19

