while (1)
    GrB.burble (1)
A = GrB (ones (3))
% C = A*A
x = GrB ([1 2 3]')
% C

C1 = GrB.mxm (A, '+.*.double', x)

'max:2nd'
C2 = GrB.mxm (A, 'max.second.double', x)
% C = GrB.mxm (A, '|.&.logical', x)
end
