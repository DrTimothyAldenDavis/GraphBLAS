function C = gb_mpower (A, b)
% C = A^b where b > 0 is an integer

if (b == 1)
    C = A ;
else
    T = gb_mpower (A, floor (b/2)) ;
    C = gbmxm (T, '+.*', T) ;
    clear T ;
    if (mod (b, 2) == 1)
        C = gbmxm (C, '+.*', A) ;
    end
end

