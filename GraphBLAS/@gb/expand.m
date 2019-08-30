function C = expand (scalar, S)
%GB.EXPAND expand a scalar to a matrix
% The scalar is expanded to the pattern of S, as in C = scalar*spones(S).
% C has the same type as the scalar.  The numerical values of S are
% ignored; only the pattern of S is used.

C = gb.gbkron (['1st.' gb.type(scalar)], scalar, S) ;

