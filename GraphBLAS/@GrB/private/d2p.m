
load dot2_results

Bnz = Anz ;
a = Anz (:) ;
b = Bnz (:) ;
k = K (:) ;
n = N (:) ;

dot_work = ((a + b) ./ n ) .* (k .* k) ;

tsax = Tsax (:) ;
tdot = Tdot (:) ;

clf
subplot (2,3,1) ; loglog (dot_work, tdot, 'o') ;

saxpy_work = (b .* a) ./ k ;

y = saxpy_work ;
subplot (2,3,2) ; loglog (y, tsax, 'o') ;
subplot (2,3,3) ; loglog (dot_work ./ y, tdot ./ tsax, 'o') ;



c = Csize (:) ;

x = (a + b) ./ c ;

w = y ./ x ;
subplot (2,3,4) ; loglog (w , tdot, 'o') ;

% i = find (a > 1e6 & a < 1e7) ;
% i = 1:length (a) ;
% a = a(i) ;
% b = b (i) ;
% tsax = tsax (i) ;
% tdot = tdot (i) ;

subplot (2,3,5) ; loglog (n ./ c , tsax ./ tdot , 'o') ;

z =  (a./k) .* (b./k) ;
subplot (2,3,5) ; loglog (z , tsax ./tdot , 'o') ;

z = y .* (a ./ (n .*k)) ;

dotspace = c ;
saxspace = a + n ;
z = saxspace ./dotspace ;

i = (x > 10*1024) ;
subplot (2,3,6) ; loglog (x(i), tsax(i)./tdot(i) , 'ro') ;
hold on         ; loglog (x(~i), tsax(~i)./tdot(~i) , 'bo') ;

d = anz ./ n ;
subplot (2,3,5) ; loglog (x ./ d, tsax./tdot , 'ko') ;
