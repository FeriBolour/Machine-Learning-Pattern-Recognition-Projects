%% Farshad Bolouri - R11630884 - Pattern Recognition - Project 2 - Part 3
clear 
close all
time = zeros(8,46);
i =1;

for N =100:20:1000
    rng(100);
    class1=mvnrnd([1 3],[1 0; 0 1],N/2);
    class2=mvnrnd([4 1],[2 0; 0 2],N/2);
    X = [class1 ; class2];
    Y = ones(length(X),1);
    Y(1:(length(X)/2)) = -1;
    %% linearly nonseparable soft margin SVM
    %SVM with C = 0.1
    tic;
    [W1, W0_1] = SVM(X,Y,0.1);
    time(1,i) = toc;
    %SVM with C = 100
    tic;
    [W2, W0_2]= SVM(X,Y,100);
    time(2,i) = toc;
    %% Linear SVM using fitcsvm
    tic;
    SVM1 = fitcsvm(X,Y,'KernelFunction','linear','BoxConstraint',0.1);
    time(3,i) = toc;
    tic;
    SVM2 = fitcsvm(X,Y,'KernelFunction','linear','BoxConstraint',100);
    time(4,i) = toc;
    %% linearly nonseparable kernel SVM
    tic;
    [G3, W0_3] = kernelSVM(1.75,X,Y,10);
    time(5,i) = toc;
    %SVM with C = 100
    tic;
    [G4, W0_4] = kernelSVM(1.75,X,Y,100);
    time(6,i) = toc;
    %% kernelSVM using fitcsvm
    tic;
    KernelSVM1 = fitcsvm(X,Y,'KernelFunction','rbf','BoxConstraint',10);
    time(7,i) = toc;
    tic;
    KernelSVM2 = fitcsvm(X,Y,'KernelFunction','rbf','BoxConstraint',100);
    time(8,i) = toc;
    
    i = i+1;
end
N = 100:20:1000;
str = ["LinearSVM, C = 0.1","LinearSVM, C = 100","kernelSVM, C = 10",...
    "kernelSVM, C = 100"];
j =1;
for i =[1,2,5,6]
    figure
    hold on
    plot(N,time(i,:),'r','LineWidth',2);
    plot(N,time(i+2,:),'b','LineWidth',2);
    legend('Implemented','fitcsvm');
    xlabel('Number of Samples');
    ylabel('Time');
    title(str(j));
    hold off
    j=j+1;
end
%% SVM: This function uses quadprog to calculcate SVM's and plots them
function [W, W0] = SVM(X,Y,C)
N = length(X);
H = (Y*Y').*(X*X');
f= -ones(1,N);
A = [-1*eye(N) ; eye(N)];
b = [zeros(1,N) C*ones(1,N)];
Aeq = Y';
beq = 0;
lambda = quadprog(H,f,A,b,Aeq,beq);
S = find(lambda > 1e-4);
W = X'*(lambda.*Y);
W0 = Y(S) - X(S,:)*W;
end
%% kernelSVM: This function uses quadprog to calculcate kernelSVM's and plots them
function [F, W0]= kernelSVM(sigma,X,Y,C)

N = length(X);
K = ones(N);

for i = 1:N
    for j =1:N
        K(i,j) = exp(-(norm(X(i,:)-X(j,:)))^2/(2*(sigma^2)));
    end
end

H = (Y*Y').*K;
f= -ones(1,N);
A = [-1*eye(N) ; eye(N)];
b = [zeros(1,N) C*ones(1,N)];
Aeq = Y';
beq = 0;
lambda = quadprog(H,f,A,b,Aeq,beq);
S = find(lambda > 1e-4);
W0 = zeros(length(S),1);
K = zeros(length(S),N);
for k = 1:length(S)
    for j = 1:N
        K(k,j) = exp(-(norm(X(S(k),:)-X(j,:)))^2/(2*(sigma^2)));
    end
end
G = ones(length(S));
for i =1:length(S)
    for j =1:N
        G(i) = G(i) + lambda(j)*Y(j)*K(i,j);
    end
    W0(i) = Y(S(i)) - G(i);
end
d = mean(W0);

F = zeros(size(Y));
K = 0;
for i = 1:N
    for j =1:N
        K = K + lambda(j)*Y(j)*exp(-(norm(X(i,:)-X(j,:)))^2/(2*(sigma^2)));
    end
    F(i) = K + d;
    K = 0;
end

end
