%% Farshad Bolouri - R11630884 - Machine Learning - Project 3 
clear
close all
rng();    

D = GenerateDataset(25,100);
lambda = 0.085:0.1:4.5;
phi = phiCal(D,100,25);
[Y_average, Y, W] = Predict(D,phi,lambda,25);

% Y = ones(25,100);
% for i = 1:100
%     Y(:,i) = (cell2mat(phi(1,i)))*W(:,i);
% end
% XT = cell2mat(D(4));
% hold on
% plot(XT(:,1),XT(:,2),'X');
% plot(XT(:,1),Y(:,4),'o');
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

%phi_test = ones(length(X_test),10);
Mu = 0:1/25:1;
for i =1:100
    X = cell2mat(D(i));
    X = X(:,1);
    for j =2:25
        phi_train(:,j) = exp(-(X-Mu(j)).^2/(2*(0.1^2)));
%        phi_test(:,j+1) = X_test.^j;
    end
    phi(i) = mat2cell(phi_train,N,N);
end
end
%% Predict function:
%this function does the predictions for each lambda and then averages them
function [Y_average, Y, W] = Predict(D, phi,lambda,N)
    Y_average = zeros(N,length(lambda));
    Y = cell(1,length(lambda));
    W = cell(1,length(lambda));
    for i = 1: length(lambda)
        W(i) = mat2cell(WCal(D,phi,lambda(i),N),N,length(D));
        Y(i) = mat2cell(YCal(W{i},phi,N,D),N,length(D));
        YCell = Y{i};
        Y_average(:,i) = mean(YCell,2);
%         for j = 1:length(D)
%             YCell = Y{i};
%             Y_average(:,i) =  Y_average(:,i)+ YCell(:,j);
%         end
        %Y_average(:,i) = Y_average(:,i)/length(D);
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
