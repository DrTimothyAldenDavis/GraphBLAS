
clear all
make
for k = [2 1] %  8 20]
    nthreads_set (k) ;
    debug_on
    testall 
    debug_off ;
    testall 
end
