%% Farshad Bolouri - R11630884 - Machine Learning - Project 3 
clear
close all

rng(41);

[D ,X_test ,T_test] = GenerateDataset(25,100,1000);
lambda = 0.08:4.82/30:4.9;
[phi, phi_test] = phiCal(D,100,25,X_test);
[W_average, Y, W, Y_test] = Predict(D,phi,lambda,25,phi_test);

Test_Error = CalError(Y_test,T_test,X_test);

[BiasSq, Var, BiasSq_Var] = BiasVarCal(W_average,W,D,25,phi,Y);

plt(BiasSq,Var,BiasSq_Var, Test_Error, lambda);

%% Generate Dataset
function [D ,X_test ,T_test] = GenerateDataset(N,L,testN)
    D = cell(1,L);
    
    for i =1:L
        X_train= rand(N,1);
        epsilon = 0.3*randn(N,1);
        D(i) = mat2cell([X_train,(sin(2*pi*X_train)...
            + epsilon)],N,2);
    end
    X_test = rand(testN,1);
    epsilon2 = 0.3*randn(testN,1);
    T_test = sin(2*pi*X_test) + epsilon2;
end

%% phiCal: This function calculates Design Matrix (phi)
function [phi ,phi_test] = phiCal(D,L,N,X_test)

phi= cell(1,L);

Mu = 0:1/14:1;

phi_train = ones(length(cell2mat(D(1))),length(Mu));
phi_test = ones(length(X_test),length(Mu));

for i =1:L
    X = cell2mat(D(i));
    X = X(:,1);
    
    for j =2:15
        phi_train(:,j) = exp(-(X-Mu(j-1)).^2/(2*(0.1^2)));
    end
    phi(i) = mat2cell(phi_train,N,length(Mu));
end

for j = 2:15
    phi_test(:,j) = exp(-(X_test-Mu(j-1)).^2/(2*(0.1^2)));
end

end
%% Predict function:
%this function does the predictions for each lambda and then averages them
function [W_average,Y, W, Y_test] = Predict(D, phi,lambda,N,phi_test)
    W_average = zeros(size(phi_test,2),length(lambda));
    Y = cell(1,length(lambda));
    W = cell(1,length(lambda));
    for i = 1: length(lambda)
        W(i) = mat2cell(WCal(D,phi,lambda(i),size(phi_test,2))...
            ,size(phi_test,2),length(D));
        Y(i) = mat2cell(YCal(W{i},phi,N,D),N,length(D));
        WCell = W{i};
        W_average(:,i) = mean(WCell,2);
        Y_test(:,i) = phi_test*W_average(:,i);
    end
    
end
%% WCal: This function calculates Feature Vector (W)
function W = WCal(D,phi,lambda,Mu)
W = ones(Mu,length(D));
for i =1:length(D)
    T = cell2mat(D(i));
    T = T(:,2);
    X = cell2mat(phi(:,i));
    W(:,i) = inv((X')*X+lambda*eye(Mu))*X'*T;
end
end
%% YCal: This function calculates Y
function Y = YCal(W,phi,N,D)
Y = ones(N,length(D));
    for i = 1:length(D)
        Y(:,i) = phi{i}*W(:,i);
    end
end
%% BiasVarCal: This function Calculates Bias and Variance
function [BiasSq Var BiasSq_Var] = BiasVarCal(W_average,W,D,N,phi,Y)
BiasSq = zeros(length(W_average),1);
Var = zeros(length(W_average),1);

   X = D{1};
   X = X(:,1);
   
   for j=1:length(W_average)
       for i =1:N
           Y_average =  phi{1}*W_average;
           BiasSq(j) = BiasSq(j) + (Y_average(i,j)-sin(2*pi*X(i)))^2;
           for k =1:length(D)
               W_lambda = W{j};
               F = phi{1}*W_lambda;
               Var(j) = Var(j) + (F(i,k)-Y_average(i,j))^2;
           end
       end
   end
   BiasSq = BiasSq/(N);
   Var = Var/(N*length(D));
   BiasSq_Var = BiasSq + Var;
  
end
%% CalError: Testing error calculation
function Test_Error = CalError(Y_test,T_test,X_test)
Test_Error = zeros(1,size(Y_test,2));
 for j =1:size(Y_test,2)
    for i=1:length(X_test)
        Test_Error(j) = Test_Error(j)+(Y_test(i,j)-T_test(i))^2;
    end

 end
 Test_Error = Test_Error/length(X_test);
end
%% plt: Plots the figure includig Bias Squared, Variance, and their addition
function y = plt(BiasSq,Var,BiasSq_Var, Test_Error,lambda)
figure
hold on
plot(log(lambda), BiasSq,'b' ,'LineWidth',2);
plot(log(lambda), Var,'r','LineWidth',2);
plot(log(lambda), BiasSq_Var,'Color','magenta','LineWidth',2);
plot(log(lambda), Test_Error,'Color','black','LineWidth',2);
xlim([-3 2]);
ylim([0 0.15]);
xticks(-3:2);
yticks(0:0.03:0.15);
xlabel('ln{\lambda}');
legend('(bias)^{2}','Variance','(bias)^{2} + Variance',...
    'test error','Location','northwest')
end

