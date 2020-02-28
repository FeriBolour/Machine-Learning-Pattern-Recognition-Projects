%% Farshad Bolouri - R11630884 - Machine Learning - Project 3 
clear
close all

rng(41);

D = GenerateDataset(25,100);
lambda = 0.08:4.415/35:4.9;
phi = phiCal(D,100,25);
[W_average, Y, W] = Predict(D,phi,lambda,25);
%  for i=1:length(Y)
%      pltLine(D,Y{1},Y_average(:,1));
%  end
% hold on
% plot(0:1/24:1,Y_average(:,1),'Color','black','LineWidth',1.5);

[BiasSq Var BiasSq_Var] = BiasVarCal(W_average,W,D,25,phi,Y);

plt(BiasSq,Var,BiasSq_Var,lambda);

%% Generate Dataset
function D = GenerateDataset(N,L)
    D = cell(1,L);
    
    for i =1:L
        X_train= rand(N,1);
        D(i) = mat2cell([X_train,(sin(2*pi*X_train)...
            + mvnrnd(0,0.3,N))],N,2);
    end
end

%% phiCal: This function calculates Design Matrix (phi)
function phi = phiCal(D,L,N)
phi_train = ones(length(cell2mat(D(1))),N);
phi= cell(1,L);

Mu = 0:1/13:1;

for i =1:L
    X = cell2mat(D(i));
    X = X(:,1);
    for j =2:14
        phi_train(:,j) = exp(-(X-Mu(j-1)).^2/(2*(0.1^2)));
    end
    phi(i) = mat2cell(phi_train,N,N);
end
end
%% Predict function:
%this function does the predictions for each lambda and then averages them
function [W_average,Y, W] = Predict(D, phi,lambda,N)
    W_average = zeros(N,length(lambda));
    Y = cell(1,length(lambda));
    W = cell(1,length(lambda));
    for i = 1: length(lambda)
        W(i) = mat2cell(WCal(D,phi,lambda(i),N),N,length(D));
        Y(i) = mat2cell(YCal(W{i},phi,N,D),N,length(D));
        WCell = W{i};
        W_average(:,i) = mean(WCell,2);
    end
    
end
%% WCal: This function calculates Feature Vector (W)
function W = WCal(D,phi,lambda,N)
W = ones(N,length(D));
for i =1:length(D)
    T = cell2mat(D(i));
    T = T(:,2);
    X = cell2mat(phi(:,i));
    W(:,i) = inv((X')*X+lambda*eye(length(X)))*X'*T;
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
%    
%   D_t = mat2cell(rand(N,1),N,1);
%   phi = phiCal(D_t,1,N);
%   phi = phi{1};
%   X = D{1};
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
   Var = Var/(N*length(D)*j/7);
   BiasSq_Var = BiasSq + Var;
  
end

%% 
function plt = pltLine(D,Y,Y_average)
figure
hold on

for i = 1:length(D)
    x = D{i};
    y = Y(:,i);
    sorted=sortrows([(x(:,1)) y]);
    sorted_x = sorted(:,1);
    sorted_y = sorted(:,2);
    fitx=linspace(0,4,100);
    fity = interp1(sorted_x,sorted_y,fitx,'spline');
    line(fitx,fity,'Color','r','LineWidth',0.5);
end
    %figure
    %Y_average = mean(Y,2);
%     sorted=sortrows([(x(:,1)) Y_average]);
%     sorted_x = sorted(:,1);
%     sorted_y = sorted(:,2);
%     fitx=linspace(0,4,100);
%     fity = interp1(sorted_x,sorted_y,fitx,'spline');
%    line(fitx,fity,'Color','black','LineWidth',1.5);
    %plot(Y_average,'Color','black','LineWidth',1.5);
    xlim([0 1])
    ylim([-2 2])
end
%% plt: Plots the figure includig Bias Squared, Variance, and their addition
function y = plt(BiasSq,Var,BiasSq_Var,lambda)
figure
hold on
plot(log(lambda), BiasSq ,'LineWidth',2);
plot(log(lambda), Var,'LineWidth',2);
plot(log(lambda), BiasSq_Var,'Color','magenta','LineWidth',2);
xlim([-3 2]);
ylim([0 0.15]);
xticks(-3:2);
yticks(0:0.03:0.15);
legend('(bias)^{2}','Variance','(bias)^{2} + Variance','Location','northwest')
end

