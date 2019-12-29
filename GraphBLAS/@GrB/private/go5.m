

try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

GrB.threads (4) ;

% n = 6 ;
% nz = 100 ;

desc = struct ;
% desc.kind = 'sparse' ;


Prob = ssget ('LAW/indochina-2004') ;
A = Prob.A  ;
clear Prob ;
n = size (A,1) ;
B = sparse (rand (n,1)) ;

for nth = [1 2 4 8]
    fprintf ('\n============================threads is %d nnz(B) is %d\n', nth, nnz(B)) ;
    htest (A, B, nth) ;
end

B = sprandn (n, 1, 0.1) ;
B1 = B ;
for nth = [1 2 4 8]
    fprintf ('\n============================threads is %d nnz(B) is %d\n', nth, nnz(B)) ;
    htest (A, B, nth) ;
end


B = sprandn (n, 1, 0.01) ;
for nth = [1 2 4 8]
    fprintf ('\n============================threads is %d nnz(B) is %d\n', nth, nnz(B)) ;
    htest (A, B, nth) ;
end

B = sprandn (n, 1, 0.001) ;
for nth = [1 2 4 8]
    fprintf ('\n============================threads is %d nnz(B) is %d\n', nth, nnz(B)) ;
    htest (A, B, nth) ;
end

A (2e9,1) = 1 ;
m = size (A,1) ;
fprintf ('\n::::: giving A %d rows ::::::::::::::::::::::::::::::::: \n',m ) ;

B = B1 ;
for nth = [1 2 4 8]
    fprintf ('\n============================threads is %d nnz(B) is %d\n', nth, nnz(B)) ;
    htest (A, B, nth) ;
end

