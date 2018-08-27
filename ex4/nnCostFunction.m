function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

    %My comment: The X matrix is as follows: In each row, there is a
    %training examples(i.e a 20x20 image in this case)
    %Theta1 and Theta2 are the weights that we multiply with respect to
    %each layer of the network(i.e Theta1 is multiplied with a1 to compute
    %z2 and etc. The reason that i take the transpose of input matrix is
    %that the sizes of the matrices only fit like that with the formulation
    %given in the slides.
    tempX = X';
    a1 = [ones(1,size(tempX,2));tempX];
    
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [ones(1,size(a2,2));a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    %Forward propagation is ended, and a3 is calculated
    
    %In here, I calculate the first error margin, by substracting the real
    %outputs from our guessed outputs, and I'm doing this with
    %vectorization and without any major loops.
    multiClassLabel = ones(size(a3));
    for i=1:m
        if(y(i) == 10)
           currentLabel = 1;
        else
            currentLabel = y(i) +1;
        end
        currentLabel = y(i);
        currentRow = zeros(size(a3,1),1);
        currentRow(currentLabel) = 1;
        multiClassLabel(:,i) = currentRow;
    end
    
    delta3 = (a3 - multiClassLabel)';
    delta3 = delta3';
    
    
    %Here, other delta terms are calculated from previous deltas and
    %Thetas. Note that the first column of the error margin function is
    %removed, since bias neurons don't have error margins, they just output
    %1. Hence, we remove them.
    delta2 = Theta2' * delta3;
    delta2(1, :) = [];
    delta2 = delta2 .* sigmoidGradient(z2);
  
    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta2_grad = Theta2_grad/m;
    
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta1_grad = Theta1_grad /m;
    
        
    
    %Here, the cost function is calculated with vectorization.
    %Multiclasslabel is just a matrix which has been fit to work with
    %multiple classes. For example, if y(i) is 1, this needs to be
    %represented as [0 1 0 0 0 0 0 0 0 0](10 is represented as 0), and for
    %calculating delta3, I already created a matrix like that, so I'm using
    %the power of vectorization to calculate the terms of cost funtion
    %without any for loops.
    firstTerm = sum(sum(-multiClassLabel.*log(a3)));
    secondTerm = sum(sum((1-multiClassLabel) .*log(1-a3)));
    
    J = (firstTerm - secondTerm) / m;
    %This part is for regulization of the cost funtion. What we are doing
    %in here is that we are first removing the weights belonging to bias
    %terms, since they should not be regularized. This implies removing the
    %first column of each theta matrix, since these represent the weights
    %of the bias terms with each node on the next layer.
    regularizedTheta1 = Theta1(1:end,2:end) .^ 2;
    regularizedTheta2 = Theta2(1:end,2:end) .^ 2;
    
    J = J + (lambda/(2*m))*(sum(sum(regularizedTheta1))+sum(sum(regularizedTheta2)));
    biasGrad1 = Theta1_grad(1:end,1);
    biasGrad2 = Theta2_grad(1:end,1);
    
    
    Theta1_grad = Theta1_grad(1:end,2:end) + (lambda/m) * Theta1(1:end,2:end);
    Theta2_grad = Theta2_grad(1:end,2:end) + (lambda/m) * Theta2(1:end,2:end);
    
    Theta1_grad = [biasGrad1 Theta1_grad];
    Theta2_grad = [biasGrad2 Theta2_grad];

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
