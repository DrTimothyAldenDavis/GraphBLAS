
try
    GrB.finalize
catch
end
clear all
GrB.init

GrB.threads (4) ;
GrB.chunk (2) ;

load saveit

% z1 = GrB.mxm ('|.&.logical', q, A)
% z2 = spones (double (q) * double (A))

% q = GrB (q, 'double')
% A = GrB (A, 'double')
% z3 = q*A

% q2 = double (q)
% A2 = double (A)

% z5 = GrB.mxm ('+.*', q2, A2)
% z4 = q2 * A2

q = double (q)'
A = double (A)'

z5 = GrB.mxm ('+.*', A, q)
z4 = A * q
