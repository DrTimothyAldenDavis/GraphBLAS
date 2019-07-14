
clear all
make
for k = [2 1] %  8 20]

    nthreads_set (k,1) ;

    if (k == 1)
        debug_on
        testall 
    end

    debug_off ;
    testall 
end
