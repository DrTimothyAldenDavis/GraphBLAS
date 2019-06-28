
clear all
make
for k = [2 1] %  8 20]
    nthreads_set (k) ;
    debug_off ;
    testall 
    debug_on
    testall 
end
