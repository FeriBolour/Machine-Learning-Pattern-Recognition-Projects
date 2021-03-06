%% Farshad Bolouri - R11630884 - Pattern Recognition - Project 2 - Part 1 & 2
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
misClassified_C4 = kernelSVM(1.75,X,Y,100,class1,class2);
%% SVM: This function uses quadprog to calculcate SVM's and plots them
function misClassified = SVM(X,Y,C,class1,class2,flag)
figure
hold on;
plot(class1(:,1),class1(:,2),'ro','LineWidth',2.5,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','r','DisplayName','class1')
plot(class2(:,1),class2(:,2),'gs','LineWidth',1.5,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','g','DisplayName','class2')


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

if flag == 1
    d = W0(2);
    tol = 0.0001;
else
    d =mean(W0);
    tol = 0.01;
end
f = @(x1,x2)  d + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6]);
h.Color = 'cyan';
h.LineWidth = 2;
h.DisplayName = 'Decision Boundary';

f = @(x1,x2)  (d-1) + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6],'--');
h.Color = 'cyan';
h.LineWidth = 1;
h.DisplayName = 'Margin Boundary';

f = @(x1,x2)  (d+1) + W(1)*x1 + W(2)*x2;
h = fimplicit(f,[-2 8 -2 6],'--');
h.Color = 'cyan';
h.LineWidth = 1;
h.DisplayName = 'Margin Boundary';

str = sprintf("C=%.1f; Sup. Vec.=%d",C,length(W0));
title(str);
misClassified = 0;
index1 = []; index2 = [];
for  i = 1:N
    if abs(Y(i)*(X(i,:)*W + d) - 1) < tol
        index1 =[index1 i];
    elseif Y(i)*(X(i,:)*W + d) < 1
        index2 = [index2 i];
        misClassified = misClassified + 1;
    end
end

plot(X(index1,1),X(index1,2),'bX','LineWidth',3,'MarkerSize',12,...
                'DisplayName','On Margin');
plot(X(index2,1),X(index2,2),'bO','LineWidth',2,'MarkerSize',2 ...
                ,'MarkerFaceColor','b','DisplayName','Inside or Misclass');
legend
hold off
end
%% kernelSVM: This function uses quadprog to calculcate kernelSVM's and plots them
function misClassified= kernelSVM(sigma,X,Y,C,class1,class2)
figure
hold on;
plot(class1(:,1),class1(:,2),'ro','LineWidth',2.5,...
    'MarkerSize',12,...
    'MarkerEdgeColor',[0.5,0.5,0.5],...
    'MarkerFaceColor','r')
plot(class2(:,1),class2(:,2),'gs','LineWidth',1.5,...
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
d = W0(2);
K=0;
[X_new,Y_new] = meshgrid(-2:0.1:6,-2:0.1:6); 
G = zeros(length(reshape(X_new.',1,[])),1);

for i = 1:length(reshape(X_new.',1,[]))
    for j = 1:N
      X_reshape  = reshape(X_new.',1,[]);
      Y_reshape  = reshape(Y_new.',1,[]);
      K = K + (lambda(j)*Y(j)*...
          exp(-(norm(X(j,:)-...
          transpose([X_reshape(i);Y_reshape(i)])))^2/(2*(sigma^2))));   
    end
    G(i) = K + d;
    K = 0;
end
G = reshape(G,size(X_new));

contour(X_new,Y_new,transpose(G),[-1 10],'Color','Cyan',...
    'LineWidth',2)
contour(X_new,Y_new,transpose(G),[-2 -1 0],'--','Color','Cyan',...
    'LineWidth',1)

str = sprintf("C=%d; Sup. Vec.=%d",C,length(W0));
title(str);
F = zeros(size(Y));
for i = 1:N
    for j =1:N
        K = K + lambda(j)*Y(j)*exp(-(norm(X(i,:)-X(j,:)))^2/(2*(sigma^2)));
    end
    F(i) = K + d;
    K = 0;
end
index1 = []; index2 = [];
for i =1 : length(S)
    if (abs(F(S(i)) +2) < 0.0001) || (abs(F(S(i))) < 0.0001)
        index1 = [index1 , S(i)];
    else
        index2 = [index2 , S(i)];
        misClassified = misClassified + 1;
    end
end
plot(X(index1,1),X(index1,2),'bX','LineWidth',3,'MarkerSize',12);
plot(X(index2,1),X(index2,2),'bO','LineWidth',2,'MarkerSize',2 ...
        ,'MarkerFaceColor','b');
legend('class1','class2','Decision Boundary','Margin Boundary',...
    'On Margin','Inside or Misclass','Location','best');

end
