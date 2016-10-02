function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

hypothesis = X * theta;
J = (1.0 / (2 * m)) * sum((hypothesis - y) .^ 2); % regular cost function term
J += (lambda / (2 * m)) * sum(theta(2:end) .^ 2); % add regularization term

% TODO: implement grad

grad = grad(:);

end
