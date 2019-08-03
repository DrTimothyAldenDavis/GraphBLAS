function Y = dnn_gb_overload (W, bias, Y0)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.  Returns result as a GraphBLAS matrix

% THIS ONE DOESN'T WORK YET ... IT'S A DRAFT

layers = length (W) ;
n = size (Y0, 2) ;

% convert to GraphBLAS (required if using operator overloading)

    fprintf ('convert to gb single\n') ;
    n = size (Y0, 2) ;
    Y0 = gb (Y0, 'single') ;
    for i=1:length(W)
        W {i} = gb (W {i}, 'single') ;
        % build a gb GrB_FP32 matrix from tuples, using '+' as the dup operator
        bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
        % change the default semiring
        bias{i}.semiring = '+.+' ;
    end

Y = Y0 ; % Initialize feature vectors.
for i=1:layers

    % Propagate through layer, Apply bias to non-zero entries, and Threshold
    % negative values.  This doesn't work yet.  Note the parentheses, since
    % the 2 semirings have to be done in the right order.
    Y = gb.select ('>=0', (Y * W {i}) * bias {i}) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, 32) ;
    if (nnz (M) > 0)
        % I think I can do this with MATLAB object overloading too:
        Y (M) = 32 ;
    end

end
