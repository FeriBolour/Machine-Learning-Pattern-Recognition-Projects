%% Generating Dataset
clear 
close all

m=[zeros(5,1) ones(5,1)];
S(:,:,1)=[0.8 0.2 0.1 0.05 0.01;
 0.2 0.7 0.1 0.03 0.02;
 0.1 0.1 0.8 0.02 0.01;
 0.05 0.03 0.02 0.9 0.01;
 0.01 0.02 0.01 0.01 0.8];
S(:,:,2)=[0.9 0.1 0.05 0.02 0.01;
 0.1 0.8 0.1 0.02 0.02;
 0.05 0.1 0.7 0.02 0.01;
 0.02 0.02 0.02 0.6 0.02;
 0.01 0.02 0.01 0.02 0.7];
P=[1/2 1/2]';
% Generating Training Set 1(100 Samples)
rng(0)
X_100 = [mvnrnd(m(:,1),S(:,:,1),50) ; mvnrnd(m(:,2),S(:,:,2),50)];
% Generating Training Set 2 (1000 Samples)
X_1000 = [mvnrnd(m(:,1),S(:,:,1),500) ; mvnrnd(m(:,2),S(:,:,2),500)];
% Generating Test Set
rng(100)
X_test = [mvnrnd(m(:,1),S(:,:,1),5000) ; mvnrnd(m(:,2),S(:,:,2),5000)];
Y_test = ones(length(X_test),1);  Y_test(5001:end) = 2;
%% Naive Bayes Classifier
% Calculating Test Error with Training Set 1
NaiveBayes_TestError_100 = NaiveBayes(X_100,X_test,Y_test);
fprintf("\nThe test error with 100 datapoints for training set ");
fprintf("using Naive Bayes is %.2f%%", NaiveBayes_TestError_100*100);
% Calculating Test Error with Training Set 2
NaiveBayes_TestError_1000 = NaiveBayes(X_1000,X_test,Y_test);
fprintf("\nThe test error with 1000 datapoints for training set ");
fprintf("using Naive Bayes is %.2f%%", NaiveBayes_TestError_1000*100);
%% Bayes Classifier with MLE
% Calculating Test Error with Training Set 1
BayesMLE_TestError_100 = BayesMLE(X_100,X_test,Y_test);
fprintf("\nThe test error with 100 datapoints for training set ");
fprintf("using Bayes with MLE is %.2f%%", BayesMLE_TestError_100*100);
% Calculating Test Error with Training Set 2
BayesMLE_TestError_1000 = BayesMLE(X_1000,X_test,Y_test);
fprintf("\nThe test error with 1000 datapoints for training set ");
fprintf("using Bayes with MLE is %.2f%%", BayesMLE_TestError_1000*100);
%% Bayes Classifier with true parameter values
% Calculating Test Error with Training Set 1
Bayes_TestError_100 = Bayes(X_test,Y_test,m',S,P);
fprintf("\nThe test error with 100 datapoints for training set ");
fprintf("using Bayes without MLE is %.2f%%", Bayes_TestError_100*100);
% Calculating Test Error with Training Set 2
Bayes_TestError_1000 = Bayes(X_test,Y_test,m',S,P);
fprintf("\nThe test error with 1000 datapoints for training set ");
fprintf("using Bayes without MLE is %.2f%%", Bayes_TestError_1000*100);
%% NaiveBayes: Naïve Bayes classifier
function NaiveBayes_TestError = NaiveBayes(X,X_test,Y_test)
% Calculate the mean and stdev for each class
Stats = cell(1,2);
Stats{1} = [mean(X(1:length(X)/2,:)); std(X(1:length(X)/2,:))];
Stats{2} = [mean(X(length(X)/2+1:end,:)); std(X(length(X)/2+1:end,:))];
% Calculating the Prior Probabilities of each class (in this case we know
% they are each 50%
priorC1 = length(X(1:length(X)/2,:))/length(X);
priorC2 = length(X(length(X)/2+1:end,:))/length(X);
predictions = zeros(length(X_test),1);
for i =1:length(X_test)
    predictions(i) = NaiveBayesPredict(X_test(i,:),priorC1,priorC2,Stats);
end
NaiveBayes_TestError = CalError(Y_test,predictions);
end
%% NaiveBayesPredict: Function that predicits classes based on Naive Bayes 
function output = NaiveBayesPredict(input,priorC1,priorC2,Stats)
prob = [priorC1 priorC2];
for i = 1:length(Stats)
    for j = 1:length(Stats{i})
        % pdf = return the probability density function of the input value
        prob(i) = prob(i)*pdf('Normal',input(j),Stats{i}(1,j),Stats{i}(2,j));
    end
end
output = find(prob == max(prob));
end
%% BayesMLE : Bayes classifier that uses MLE for parameter estimation
function BayesMLE_TestError = BayesMLE(X,X_test,Y_test)
% Calculate the mean and stdev for each class
Stats = cell(1,2);
Stats{1} = [mean(X(1:length(X)/2,:)); std(X(1:length(X)/2,:))];
Stats{2} = [mean(X(length(X)/2+1:end,:)); std(X(length(X)/2+1:end,:))];
% Calculating the Prior Probabilities of each class (in this case we know
% they are each 50%
priorC1 = length(X(1:length(X)/2,:))/length(X);
priorC2 = length(X(length(X)/2+1:end,:))/length(X);
predictions = zeros(length(X_test),1);
for i =1:length(X_test)
    predictions(i) = BayesPredict(X_test(i,:),priorC1,priorC2,Stats);
end
BayesMLE_TestError = CalError(Y_test,predictions);
end
%% Bayes: Bayes classifier that uses the true parameter values
function Bayes_TestError = Bayes(X_test,Y_test,mu,cov,Priors)
% Calculate the mean and stdev for each class
Stats = cell(1,2);
Stats{1} = [mu(1,:); cov(:,:,1)];
Stats{2} = [mu(2,:); cov(:,:,2)];
% Calculating the Prior Probabilities of each class (in this case we know
% they are each 50%
priorC1 = Priors(1);
priorC2 = Priors(2);
predictions = zeros(length(X_test),1);
for i =1:length(X_test)
    predictions(i) = BayesPredict(X_test(i,:),priorC1,priorC2,Stats);
end
Bayes_TestError = CalError(Y_test,predictions);
end
%% BayesMLEPredict: Function that predicits classes based on Bayes Classifier with MLE
function output = BayesPredict(input,priorC1,priorC2,Stats)
prob = [priorC1 priorC2];
for i = 1:length(Stats)
        prob(i) = prob(i)*mvnpdf(input,Stats{i}(1,:),Stats{i}(2:end,:));
end
output = find(prob == max(prob));
end
%% CalError: Testing error calculation
function Test_Error = CalError(Y_test,predictions)
misClassifications = 0;
for i =1:length(predictions)
    if predictions(i) ~= Y_test(i)
        misClassifications = misClassifications + 1;
    end
end
Test_Error = (misClassifications/length(predictions));
end
