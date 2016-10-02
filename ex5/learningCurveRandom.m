function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)



boundaryExamples = min(size(X, 1), size(Xval, 1));
error_train_sum = zeros(boundaryExamples, 1);
error_val_sum = zeros(boundaryExamples, 1);

iterations = 25;
for iteration = 1:iterations
    for i = 1:boundaryExamples
        randomIndicesTrain = randperm(size(X, 1));
        randomIndicesTrain = randomIndicesTrain(1:i);
        X_rand_train = X(randomIndicesTrain, :);

        randomIndicesVal = randperm(size(Xval, 1));
        randomIndicesVal = randomIndicesVal(1:i);
        X_rand_val = Xval(randomIndicesVal, :);

        theta = trainLinearReg(X_rand_train, y(randomIndicesTrain), lambda);
        error_train_sum(i) += linearRegCostFunction(X_rand_train, y(randomIndicesTrain), theta, 0);
        error_val_sum(i) += linearRegCostFunction(X_rand_val, yval(randomIndicesVal), theta, 0);
    end
end

error_train = error_train_sum / iterations;
error_val = error_val_sum / iterations;

error_train = error_train(:);
error_val = error_val(:);

end
