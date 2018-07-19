function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


k = X * theta;
ll = k - y;
lk = sum(ll.^2);
kw = (1/(2*m)) * lk;

d1 = (lambda/(2*m));
d2 = theta(2:end,:);
d3 = sum(d2.^2);
d4 = d1 * d3;
J = kw + d4;


%theta = theta - (alpha* (1/m)) * (X' * (X*theta - y))
                                  
a8 = (X*theta - y);
a9 = X' * a8;
ks = a9/m ;
grad(1) = ks(1);
grad(2:end,:) = ks(2:end,:) + (lambda / m) * theta(2:end,:);







% =========================================================================

grad = grad(:);

end
