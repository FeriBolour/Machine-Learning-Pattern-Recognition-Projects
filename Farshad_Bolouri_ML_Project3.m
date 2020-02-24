%% Farshad Bolouri - R11630884 - Machine Learning - Project 3 
clear
close all
rng();    

D = GenerateDataset(25,100);
lambda = 0.085:0.01:4.5;
phi = phiCal(D,100);
W = WCal(D,phi);
Y = ones(25,100);
for i = 1:100
    Y(:,i) = (cell2mat(phi(1,i)))*W(:,i);
end
XT = cell2mat(D(4));
hold on
plot(XT(:,1),XT(:,2),'X');
plot(XT(:,1),Y(:,4),'o');
%% Generate Dataset
function D = GenerateDataset(N,L)
    D = cell(1,L);
       
    for i =1:L
        X_train= rand(N,1);
        D(i) = mat2cell([X_train,(sin(2*pi*X_train)...
            + mvnrnd(0,0.3,N))],25,2);
    end
end

%% phiCal: This function calculates Design Matrix (phi)
function phi = phiCal(D,L)
phi_train = ones(length(cell2mat(D(1))),25);
phi= cell(1,L);

%phi_test = ones(length(X_test),10);
Mu = randn(1,25);
for i =1:100
    X = cell2mat(D(i));
    X = X(:,1);
    for j =2:25
        phi_train(:,j) = exp(-(X-Mu(j-1)).^2/(2*(0.1^2)));
%        phi_test(:,j+1) = X_test.^j;
    end
    phi(i) = mat2cell(phi_train,25,25);
end
end
%% WCal: This function calculates Feature Vector (W)
function W = WCal(D,phi)
W = ones(25,100);
for i =1:100
    T = cell2mat(D(i));
    T = T(:,2);
    X = cell2mat(phi(:,i));
    W(:,i) = inv((X')*X+1*eye(length(X)))*X'*T;
end
end