
clear all
make
for k = [1 2 4 8 20]
    nthreads_set (k) ;
    debug_off ;
    testall 
    debug_on
    testall 
end
