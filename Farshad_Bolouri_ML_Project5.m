%% Farshad Bolouri - R11630884 - Machine Learning - Project 5
close all
clear
%% Part a - XOR Classification Problem
X = [-1 1 -1 1;-1 1 1 -1]; %DxN
T = [1 1 0 0;0 0 1 1];
Title = 'XOR Classification Problem';
[W1,W2,b1,b2] = NeuralNetwork(X,T,2,0.5,3000,10,Title,'Classification');
figure
x1range = -1.5:.075:1.5;
x2range = -1.5:.075:1.5;
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:)  xx2(:)]';
Predictions = predictNN(XGrid,W1,W2,b1,b2,'Classification');
decisionSurface = reshape(Predictions(1,:)',length(xx1),length(xx1));
decisionSurface = rescale(decisionSurface,-1,1);
surf(xx1, xx2 ,decisionSurface,'FaceAlpha',0.55);
hold on
title('XOR Surface Decision')
set(gca,'xlim',[-2 2],'ylim',[-2 2]);
points = rescale(T,-1,1);
plot3(X(1,1:2),X(2,1:2),points(1,1:2),'bo','MarkerSize',10,...
    'MarkerFaceColor','blue');
plot3(X(1,3:4),X(2,3:4),points(1,3:4),'ro','MarkerSize',10,...
    'MarkerFaceColor','red');
xlabel('x1'); ylabel('x2'); zlabel('d(x)');
%% Part b - Regression Problem - 3 tanh hidden units units
rng(100)
X=2*rand(1,50)-1;
T=sin(2*pi*X)+0.3*randn(1,50);
Title = 'Regression with 3 tanh units';
[W1,W2,b1,b2] = NeuralNetwork(X,T,3,0.01,3000,10,Title,'Regression');

figure
plot(X,T,'o')
hold on
title('Regression with 3 tanh units')
X = -1:0.01:1;
Predictions = predictNN(X,W1,W2,b1,b2,'Regression');
plot(X,Predictions,'r');
%% Part b - Regression Problem - 20 tanh hidden units units
rng(100)
X=2*rand(1,50)-1;
T=sin(2*pi*X)+0.3*randn(1,50);
Title = 'Regression with 20 tanh units';
[W1,W2,b1,b2] = NeuralNetwork(X,T,20,0.0039,3000,10,Title,'Regression');

figure
plot(X,T,'o')
hold on
title('Regression with 20 tanh units')
X = -1:0.01:1;
Predictions = predictNN(X,W1,W2,b1,b2,'Regression');
plot(X,Predictions,'r');
%% one hidden layer NeuralNetwork - n = number of units
function [W1,W2,b1,b2] = NeuralNetwork(X,T,n,rho,epochs,seed,Title,problemType)
if strcmp(problemType,'Classification')
    rng(seed)
end
[D,N] = size(X);
W1 = randn(n,D);
W2 = randn(size(T,1),n);

b1 = zeros(n,1);
b2 = zeros(size(T,1),1);

if strcmp(problemType,'Classification')
    h = @Sigmoid;
    H = @Softmax;
    h_prime = @sigDer;
    error = @CrossError;
else
    h = @(X) tanh(X);
    H = @(X) X;
    h_prime = @(X) sech(X).^2;
    %h_prime = @tanh_der;
    error = @(Y,T,N) mean((Y - T).^2);
end

figure
title(Title);
set(gca,'YScale','log');
ylabel('Error')
xlabel('Epochs')
plt = animatedline('Color','r','LineWidth',1.5);
a = tic;
for i = 1:epochs
    B1 = repmat(b1,1,N);
    B2 = repmat(b2,1,N);
    % Feed Forward
    A1 = W1*X + B1;
    Z1 = h(A1);
    A2 = W2*Z1 + B2;
    Z2 = H(A2);
    
    % BackProp
    Delta2 = (Z2 - T);
    Delta1 = (W2'*Delta2).*h_prime(A1);
    
    W2 = W2 - rho*(Delta2*Z1');
    W1 = W1 - rho*(Delta1*X');
    b2 = b2 - rho*sum(Delta2,2);
    b1 = b1 - rho*sum(Delta1,2);
    
    %Error
    addpoints(plt,i,error(Z2,T,N));
    b = toc(a);
    if b > (1/1000)
        drawnow % update screen every 1/30 seconds
        a = tic; % reset timer after updating
    end
     
end
drawnow
end
%% PredictNN function
function Predictions = predictNN(X,W1,W2,b1,b2,problemType)

if strcmp(problemType,'Classification')
    h = @Sigmoid;
    H = @Softmax;
else
    h = @(X) tanh(X);
    H = @(X) X;
end

B1 = repmat(b1,1,length(X));
B2 = repmat(b2,1,length(X));
A1 = W1*X + B1;
Z1 = h(A1);
A2 = W2*Z1 + B2;
Predictions = H(A2);
end
%% Logistic Sigmoid function
function Y = Sigmoid(X)
Y = 1./(1+exp(-X));
end
%% Softmax function
function Y = Softmax(X)
Y = exp(X)./sum(exp(X));
end
%% Sigmoid Derivative
function Y = sigDer(X)
h = 1./(1+exp(-X));
Y = h.*(1-h);
end
%% CrossEntropy Error
function error = CrossError(Y,T,N)
[~, argmax] = max(T);
logP = [];
for i = 1:N
    logP = [logP -log(Y(argmax(i),i))];
end
error = sum(logP)/N;
end
