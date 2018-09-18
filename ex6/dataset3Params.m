function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% =========================================================================
  
  valueList = [0.01,0.03,0.1,0.3,1,3,10,30,100];
  bestCValue = 0;
  bestSigmaValue = 0;
  bestErrorValue = Inf;
  listSize = size(valueList,2);
  totalTrainCount = listSize ^ 2;
  currentTrainCount = 0;
  for i=1:listSize
    currentCValue = valueList(i);
    for j=1:listSize
      currentSigmaValue = valueList(j);
      model= svmTrain(X, y, currentCValue, @(X, y) gaussianKernel(X, y, currentSigmaValue));
      predictions = svmPredict(model,Xval);
      error = mean(double(predictions ~= yval));
      currentTrainCount = currentTrainCount +1;
      fprintf("Training: %d/%d\n",currentTrainCount,totalTrainCount);
      if(error < bestErrorValue)
        fprintf("Old Error: %f,New error: %f, Old C: %f, New C: %f,\n Old sigma: %f, New sigma: %f",bestErrorValue, error,
        bestCValue,currentCValue,bestSigmaValue,currentSigmaValue);
        bestCValue = currentCValue;
        bestSigmaValue = currentSigmaValue;
        bestErrorValue = error;
      endif
    endfor
    
  endfor
  
  C = bestCValue;
  sigma = bestSigmaValue;



end
