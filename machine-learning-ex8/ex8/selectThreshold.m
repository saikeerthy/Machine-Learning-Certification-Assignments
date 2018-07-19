function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
%sum1 = 0;
%sum2 = 0;
%sum3 = 0;
predictions = (pval < epsilon);
%m = size(yval,1);

tp = sum((yval == 1) & (predictions == 1));
fp = sum((yval == 0) & (predictions == 1));
fn = sum((yval == 1) & (predictions == 0));
%for i = 1:m
%tp = (pval(i,:) < epsilon) & (yval(i,:) == 1) + sum1;
%fp = (pval(i,:) < epsilon) & (yval(i,:) == 0) + sum2;
%fn = (pval(i,:) > epsilon) & (yval(i,:) == 1) + sum3;
%end
prec = (tp / (tp + fp)) ;

rec = (tp / (tp + fn ));

F1 = ((2*prec*rec)/ (prec + rec ));








    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
