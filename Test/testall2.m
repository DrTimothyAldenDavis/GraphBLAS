
clear all
for k = 1:2
    nthreads_set (k) ;
    testall 
end
debug_on
for k = 1:2
    nthreads_set (k) ;
    testall 
end
