function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

grad = zeros(size(theta));

thetaWithZeroFirstElem = [0; theta(2:end)];
hypothesis = X * theta;
J = (1.0 / (2 * m)) * sum((hypothesis - y) .^ 2); % regular cost function term
J += (lambda / (2 * m)) * sum(thetaWithZeroFirstElem .^ 2); % add regularization term

grad = (1.0 / m) * X' * (hypothesis - y) + ...
        (lambda / m) * thetaWithZeroFirstElem;
grad = grad(:);

end
