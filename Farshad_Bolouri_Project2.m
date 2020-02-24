%% Farshad Bolouri - R11630884 - Machine Learning - Project 2 
clear all
rng(31);    %Used this seed after looping around the first 150 seeds

%First Dataset using N-Train = 10
N =10;
[X_train,T_train,X_test,T_test] = GenerateDataset(N);
Main(X_train,T_train,X_test,T_test,N);

%Second Dataset using N-Train = 100 
N =100;
[X_train,T_train,X_test,T_test] = GenerateDataset(N);
Main(X_train,T_train,X_test,T_test,N);

%% Main Function: This function Predicts your Y's, Calculates the error and plots the E-RMS
function y = Main(X_train,T_train,X_test,T_test,N)
%Calculate predictions for training and testing set 
[Y_train, Y_test] = Predict(X_train,T_train,X_test);

[Train_Error, Test_Error] = CalError(Y_train,Y_test,T_train,T_test...
    ,X_train,X_test);
%Now Calculate E-RMS for Train-Error and Test-Error
E_train = sqrt(Train_Error/length(X_train));
E_test = sqrt(Test_Error/length(X_train));
 
plotE_rms(E_train,E_test,N);
end

%% Generate Dataset
function [X_train,T_train,X_test,T_test] = GenerateDataset(N)
    X_train= rand(N,1);  
    T_train = sin(2*pi*X_train) + mvnrnd(0,0.3,N);
    
    X_test= rand(100,1);
    T_test = sin(2*pi*X_test) + mvnrnd(0,0.3,100);
        
end

%% Predict: This function Calculates your predictions for training and testing set 
function [Y_train, Y_test] = Predict(X_train,T_train,X_test)
%Calculate Design Matrix (phi) and Feature Vectors(W)
[phi_train,phi_test] = phiCal(X_train,X_test); 
W = WCal(phi_train,T_train);
%intiialize your Y_train and Y_test; 10 Columns for M = 0:9
Y_train = ones(length(X_train),10);
Y_test = ones(length(X_test),10);
%Calculate Y_train
for i = 1:10
    Y_train(:,i) = phi_train(:,[1:i])*cell2mat(W(i));
end 
%Calculate Y_test
for i = 1:10
    Y_test(:,i) = phi_test(:,[1:i])*cell2mat(W(i));
end

end
%% phiCal: This function calculates Design Matrix (phi)
function [phi_train,phi_test] = phiCal(X_train,X_test)
phi_train = ones(length(X_train),10);
phi_test = ones(length(X_test),10);

for j =0:9
        phi_train(:,j+1) = X_train.^j;
        phi_test(:,j+1) = X_test.^j;
end
end
%% WCal: This function calculates Feature Vector (W)
function W = WCal(phi_train,T_train)
W = cell(1,10);
for i =1:10
    W(i) = mat2cell(inv(((phi_train(:,[1:i]))')*phi_train(:,[1:i]))*...
        phi_train(:,[1:i])'*T_train,i);
end
end

%% CalError: Training and Testing error calculation
function [Train_Error, Test_Error] = CalError(Y_train,Y_test,T_train,T_test,X_train,X_test)
Train_Error = zeros(1,10);
Test_Error = zeros(1,10);
 for j =1:10
    for i=1:length(X_train)
        Train_Error(j) = Train_Error(j)+(Y_train(i,j)-T_train(i))^2;
    end
    for i=1:length(X_test)
        Test_Error(j) = Test_Error(j)+(Y_test(i,j)-T_test(i))^2;
    end
 end
end
%% Plotting E-RMS
function y = plotE_rms(E_train,E_test,N)
figure
M = 0:9;
hold on
plot(M,E_train,'b-o','LineWidth',2);
xlabel('M');ylabel('E-RMS');
plot(M,E_test,'r-o','LineWidth',2);
str = sprintf('N-Train = %d',N);
title(str)
legend('Training','Test');
hold off
end

