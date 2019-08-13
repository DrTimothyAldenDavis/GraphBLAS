
n = 10
e = ones(n,1);
A = spdiags([e -2*e e], -1:1, n, n)
e = gb (e) 
B = myspdiags([e -2*e e], -1:1, n, n)

nothing = sparse (A-B)
