function gbtest35
%GBTEST35 test reshape

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default')

for m = 1:6
    for n = 1:10
        A = rand (m, n) ;
        G = gb (A) ;
        mn = m*n ;
        f = factor (mn) ;
        for k = 1:length (f)
            S = nchoosek (f, k) ;
            for i = 1:size(S,1)
                m2 = prod (S (i,:)) ;
                n2 = mn / m2 ;
                C1 = reshape (A, m2, n2) ;
                C2 = reshape (G, m2, n2) ;
                assert (gbtest_eq (C1, C2)) ;
            end
        end
    end
end

fprintf ('gbtest35: all tests passed\n') ;
