data = readtable('C:\Users\THINKPAD\Desktop\merged_sepsis_data.csv');

data = data(randperm(size(data, 1)), :);

trainRatio = 0.8;
trainSize = round(size(data, 1) * trainRatio);
trainData = data(1:trainSize, :);
testData = data(trainSize+1:end, :);

trainFeatures = table2array(trainData(:, 1:9));
trainTarget = table2array(trainData(:, 10));

testFeatures = table2array(testData(:, 1:9));
testTarget = table2array(testData(:, 10));

opt = genfisOptions("SubtractiveClustering");
fis = genfis(trainFeatures, trainTarget,opt);

trainDataMatrix = [trainFeatures, trainTarget];

trainedFIS = anfis(trainDataMatrix, fis, 10);

testOutput = evalfis(testFeatures, trainedFIS);
mse = mean((testOutput - testTarget).^2);

testOutput(testOutput < 0.5) = 0;
testOutput(testOutput >= 0.5) = 1;

numInstances = 10;
indices = randperm(numel(testTarget), numInstances);

selectedTestFeatures = testFeatures(indices, :);
selectedTestTarget = testTarget(indices);
selectedTestOutput = testOutput(indices);
tableData = [selectedTestTarget, selectedTestOutput];
disp('Actual / Prediction Table:');
disp(tableData);

[x,mf] = plotmf(fis,'input',1);
subplot(3,1,1)
plot(x,mf)
xlabel('Membership Functions for Input 1')
[x,mf] = plotmf(fis,'input',2);
subplot(3,1,2)
plot(x,mf)
xlabel('Membership Functions for Input 2')

confusionMatrix = [sum(testTarget & testOutput), sum(~testTarget & testOutput);
                   sum(testTarget & ~testOutput), sum(~testTarget & ~testOutput)];

TN = confusionMatrix(1,1);
FP = confusionMatrix(1,2);
FN = confusionMatrix(2,1);
TP = confusionMatrix(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);
accuracy = sum(testOutput == testTarget) / numel(testTarget);

fprintf('Confusion Matrix:\n');
fprintf('-----------------\n');
fprintf('True Negative (TN): %d\n', TN);
fprintf('False Positive (FP): %d\n', FP);
fprintf('False Negative (FN): %d\n', FN);
fprintf('True Positive (TP): %d\n', TP);

fprintf('\n');
fprintf('Accuracy: %.2f\n', precision);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-score: %.2f\n', f1_score);