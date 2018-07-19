function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%k = ((log(X*theta) .* (-y)) - (log(1- X*theta) .* ( 1 - y)));

a1 = X*theta;
a2 = log(sigmoid(a1));
a3 = -y .* a2;
a4 =log(1-sigmoid(a1));
a5 = 1-y;
a6 = a5 .* a4;
a7 = sum(a3 - a6);
J = (1/m)*a7;
a8 = sigmoid(a1)-y;
a9 = X' * a8;
grad = a9/m ;





% =============================================================

end
