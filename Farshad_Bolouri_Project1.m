 %% Farshad Bolouri - R11630884 - Machine Learning - Project 1
%% Loading the Dataset
clear 
close all
load('carbig')
Dataset = [Weight Horsepower];
Dataset(find(isnan(Dataset(:,2)) == 1), :) = [];  %removing the 'NaN's 
%% Closed Form
X = Dataset(:,1); 
T = Dataset(:,2);
hold on
plot(X,T,'X','color','r');
xlabel('Weight')
ylabel('Horsepower');
title('Matlab''s "carbig" Dataset');

X = [X ones(length(X),1)];  
W = inv((X')*X)*X'*T;
Y = X*W;

plot(X(:,1),Y,'b','LineWidth',2);
legend('Dataset','Closed Form')
hold off
%% Gradient Descent
X = Dataset(:,1);
X_norm = (X-mean(X))/std(X); %Normalizing input Data with Z-Score method
T = Dataset(:,2);

figure;
hold on 
plot(X,T,'X','color','r');
xlabel('Weight')
ylabel('Horsepower');
title('Matlab''s "carbig" Dataset');

X = [X ones(length(X),1)];
X_norm = [X_norm ones(length(X),1)];
W = [1,1];
difference = 1;
previousW = 0;

while difference > 1e-10  
    W =W';
    previousW = W; 
    
    W = W' - 0.001*(2*W'*(X_norm'*X_norm) - 2*T'*X_norm); %Gradient Descent
    
    difference = abs(norm(W)-norm(previousW));
 
end

Y = X_norm*W';
plot(X(:,1),Y,'g','LineWidth',2);
legend('Dataset','Gradient Descent')
hold off

