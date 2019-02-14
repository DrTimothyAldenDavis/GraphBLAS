




%           A = GB_mex_random (k, m, m*5, 1, 1) ;
%           B = GB_mex_random (k, n, n*5, 1, 2) ;
%           C = GB_mex_AdotB (A, B) ;
%           C2 = A'*B  ;
%           err = norm (C-C2,1) ;
%           maxerr = max (maxerr, err) ;
% save gunk A B C C2 err

load gunk
C3 = GB_mex_AdotB (A, B) ;
norm (C-C3)
