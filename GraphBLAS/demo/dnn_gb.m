function Y = dnn_gb (W, bias, Y0)
%DNN_GB Sparse deep neural network in GraphBLAS
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.
%
% Compare with dnn_matlab.m.
%
% Usage:
%
%   Y = dnn_gb (W, bias, Y0)
%
% The matrices can be stored by row or by column, but gb.format ('by row')
% is significantly faster.
%
% See also dnn_matlab, dnn_mat2gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

t = zeros (4) ;

Y = Y0 ;
for i=1:length(W)

    % Propagate through layer, apply bias, and threshold negative values.
    Y = gb.select ('>0', gb.mxm ('+.+', Y * W {i}, bias {i})) ;
     
    %{
    tt = tic ;
    Y = Y * W {i} ;
    t (1) = t (1) + toc (tt) ;

    tt = tic ;
    Y = gb.mxm ('+.+', Y, bias {i}) ;
    t (2) = t (2) + toc (tt) ;

    tt = tic ;
    Y = gb.select ('>0', Y) ;
    t (3) = t (3) + toc (tt) ;

    % Threshold maximum values.
    tt = tic ;
    %}
    M = Y > 32 ;
    if (nnz (M) > 0)
        Y (M) = 32 ;
    end
    %{
    t (4) = t (4) + toc (tt) ;
    %}

end

%{
fprintf ('Y*W : %g sec\n', t (1)) ;
fprintf ('Y+B : %g sec\n', t (2)) ;
fprintf ('ReLU: %g sec\n', t (3)) ;
fprintf ('Ymax: %g sec\n', t (4)) ;
%}
