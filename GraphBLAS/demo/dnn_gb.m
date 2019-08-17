function Y = dnn_gb (W, bias, Y0)
%DNN_GB Sparse deep neural network in GraphBLAS
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

%-------------------------------------------------------------------------------
% convert to GraphBLAS
%-------------------------------------------------------------------------------

% This is mostly optional since gb.* methods can take inputs that are either gb
% objects or MATLAB sparse matrices.  The problem is converted to single
% precision since it gives the same result.  This is a bit faster, but MATLAB
% doesn't have sparse single precision matrices, so a conversion from one to
% the other needs to be made.

% The bias{i} matrix differs, and this needs to be modified here (or in
% dnn_matlab.m).  For dnn_matlab.m, bias{i} is a 1-by-n row vector.  For the
% GraphBLAS semiring, it is an n-by-n diagonal matrix.  When comparing dnn_gb.m
% and dnn_matlab.m, this code should not be considered extra work, since the
% problem could be generated in GraphBLAS format to begin with.  In that case,
% dnn_matlab.m would include this conversion code, to convert the problem from
% GraphBLAS format to MATLAB sparse matrices.

t = tic ;
n = size (Y0, 2) ;
Y0 = gb (Y0, 'single') ;
for i=1:length(W)
    W {i} = gb (W {i}, 'single') ;
    bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
end
t = toc (t) ;
fprintf ('setup time: %g sec\n', t) ;

%-------------------------------------------------------------------------------
% solve the dnn problem: compare with dnn_matlab.m
%-------------------------------------------------------------------------------

Y = Y0 ;
for i=1:length(W)
    t = tic ;

    % The original GraphChallenge did not have the threshold of 32, so that
    % original problem statement can be solved in an inner loop of single line
    % of MATLAB+GraphBLAS; the one-liner Sparse Deep Neural Network:

    % Propagate through layer, apply bias, and threshold negative values.
    Y = gb.select ('>0', gb.mxm ('+.+', Y * W {i}, bias {i})) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, 32) ;
    if (nnz (M) > 0)
        Y = gb.assign (Y, M, 32) ;
    end

    % Alternatively, the above threshold can be done as follows, but it could
    % be a bit slower if M is empty.  This would solve the entire problem in an
    % inner loop just 2 lines of code:

    % Y = gb.assign (Y, gb.select ('>thunk', Y, 32), 32) ;

    t = toc (t) ;
    fprintf ('layer: %4d, nnz (Y) %8d, time %g sec\n', i, nnz (Y), t) ;
end

