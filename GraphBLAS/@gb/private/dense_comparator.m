function C = dense_comparator (op, A, B)
% The pattern of C is a full matrix.  A and B must first be expanded to to
% a full matrix with explicit zeros.  For example, with A <= B for two
% matrices A and B:
%
%     in A        in B        A(i,j) <= B (i,j)    true or false
%     not in A    in B        0 <= B(i,j)          true or false
%     in A        not in B    A(i,j) <= 0          true or false
%     not in A    not in B    0 <= 0               true, in C

C = gb.select ('nonzero', gb.emult (op, full (A), full (B))) ;

