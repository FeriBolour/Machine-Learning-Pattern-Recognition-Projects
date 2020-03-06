%% Farshad Bolouri - R11630884 - Pattern Recognition - Project 2
clear 
close all
rng(100);
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
X = [class1 ; class2];
Y = ones(length(X),1);
Y(1:60) = -1;

%% linearly nonseparable soft margin SVM 
%SVM with C = 0.1
misClassified_C1 = SVM(X,Y,0.1,class1,class2,1);
%SVM with C = 100
misClassified_C2 = SVM(X,Y,100,class1,class2, 0);
%% linearly nonseparable kernel SVM 
misClassified_C3 = kernelSVM(1.75,X,Y,10,class1,class2);
%SVM with C = 100
%misClassified_C4 = SVM(X,Y,100,class1,class2, 0);
%% SVM: This function uses quadprog to calculcate SVM's and plots them
function misClassified = SVM(X,Y,C,class1,class2,flag)
figure
hold on;
plot(class1(:,1),class1(:,2),'ro','LineWidth',3,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','r')
plot(class2(:,1),class2(:,2),'gs','LineWidth',2,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','g')


L = size(X,2);
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
d=0; tol = 0;
if flag == 1
    d = W0(2)
    tol = 0.0001;
else
    d =mean(W0);
    tol = 0.01;
end
f = @(x1,x2)  d + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6]);
h.Color = 'cyan';
h.LineWidth = 2;

f = @(x1,x2)  (d-1) + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6],'--');
h.Color = 'cyan';
h.LineWidth = 1;

f = @(x1,x2)  (d+1) + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6],'--');
h.Color = 'cyan';
h.LineWidth = 1;


str = sprintf("C=%.1f; Sup.Vec.=%d",C,length(W0));
title(str);
misClassified = 0;

for  i = 1:N
    if abs(Y(i)*(X(i,:)*W + d) - 1) < tol
        plot(X(i,1),X(i,2),'bX','LineWidth',3,'MarkerSize',12);
    elseif Y(i)*(X(i,:)*W + d) < 1 
        plot(X(i,1),X(i,2),'bO','LineWidth',2,'MarkerSize',2 ...
        ,'MarkerFaceColor','b');
        misClassified = misClassified + 1;
    end
end

hold off
end
%% phiCal: This function calculates Design Matrix (phi)
function misClassified= kernelSVM(sigma,X,Y,C,class1,class2)
figure
hold on;
plot(class1(:,1),class1(:,2),'ro','LineWidth',3,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','r')
plot(class2(:,1),class2(:,2),'gs','LineWidth',2,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','g')

N = length(X);
misClassified = 0;
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
K=0;
G = zeros(N,1);
for i = 1:N
    for j = 1:N
      K = K + (lambda(j)*Y(j)*...
           exp(-(norm(X(j,:)-X(i,:)))^2/(2*(sigma^2))));   
    end
    G(i) = K + d;
    K = 0;
end

plot(X(:,1),G,'X','Color','Cyan')
end