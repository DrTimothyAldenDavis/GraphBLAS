clear all
load gunk
%        save gunk A A2 type
A = A(1,1)
A2 = A2(1,1)
A-A2
fprintf ('A : %40.30g\n', A) ;
fprintf ('                          %40o\n', A) ;
        C1 = bitcmp (A, type) ;
        C2 = bitcmp (A2, type) ;
        fprintf ('C1: %30.30g  %30o\n', C1, C1)
        fprintf ('C2: %30.30g  %30o\n', C2, C2) ;
        C1-C2
        assert (isequal (C1, C2)) 

        