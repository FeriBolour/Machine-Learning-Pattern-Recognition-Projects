%% Farshad Bolouri - R11630884 - Pattern Recognition - Project 1
clear
close all
load('fisheriris');
SepL = meas(:,1);
SepW = meas(:,2);
PetL = meas(:,3);
PetW = meas(:,4);
Class = ones(150,1);
%Class label: Setosa = 1, Versicolor = 2, Virginica = 3
W = cell(1,9);
missClassifiedClosed = [];
%% Data analisys
%Minimum of Measurements
minSepL=min(SepL); minSepW=min(SepW); minPetL=min(PetL); minPetW=min(PetW);
%Maximum of Measurements
maxSepL=max(SepL); maxSepW=max(SepW); maxPetL=max(PetL); maxPetW=max(PetW);
%Mean of Measurements
meanSepL=mean(SepL); meanSepW=mean(SepW); meanPetL=mean(PetL); 
meanPetW=mean(PetW);
%Variance of Measurements
varSepL=var(SepL); varSepW=var(SepW); varPetL=var(PetL); varPetW=var(PetW);
%Within-Class Variance
SW = zeros(1,4);
for i = 1:4
   SW(i) = (1/3)*(var(meas([1:50],i)) + var(meas([51:100],i)) +...
       var(meas([101:150],i)));
end
%Between-Class Variance
SB = zeros(1,4);
for i = 1:4
   SB(i) = (1/3)*((mean(meas([1:50],i))-mean(meas(:,i)))^2+...
                  (mean(meas([51:100],i))-mean(meas(:,i)))^2+...
                  (mean(meas([101:150],i))-mean(meas(:,i)))^2);
end
%% Computing and Displaying Correlation Coefficients
Class(1:50) = 1;Class(51:100) = 2;Class(101:150) = 3;
hold on
CC = corrcoef([meas,Class]);
imagesc(rot90(CC));
colormap('jet');
colorbar();
xlim([0,6]);
ylim([0,6]);
grid()
xticks([1,2,3,4,5])
xticklabels({'SepL','SepW','PetL','PetW','Class'});
yticks([1,2,3,4,5])
yticklabels({'Class','PetW','PetL','SepW','SepL'});
hold off
%% Features vs the Class Label
figure
subplot(2,2,1)
plot(SepL,Class,'rx');
title('SepL Vs Class');
xlim([0 8]);

subplot(2,2,2)
plot(SepW,Class,'rx');
title('SepW Vs Class');
xlim([0 8]);

subplot(2,2,3)
plot(PetL,Class,'rx');
title('PetL Vs Class');
xlim([0 8]);

subplot(2,2,4)
plot(PetW,Class,'rx');
title('PetW Vs Class');
xlim([0 8]);
%% Perceptron for All Features : Setosa Vs. Versi+Virgi
missClassified = zeros(1,4);
epochs = zeros(1,4);
Class(1:50) = -1; %Setosa class
Class(51:150) = 1;  %Versi+Virigi class
[S_vs_VV_All_Perce,W(1),epochs(1),missClassified(i)] = Perceptron(...
    [meas ones(length(meas),1)],Class,[1; 1; 1; 1; 1]);

%% Least Squares for All Features : Setosa Vs. Versi+Virgi
[S_vs_VV_All_LS,W(2)] = LS_predict(meas,Class,0);
missClassifiedClosed(1) = missClassification(51,S_vs_VV_All_LS,2);
%% Perceptron for features 3 and 4 : Setosa Vs. Versi+Virgi
figure
hold on
plot(PetL(1:50),PetW(1:50),'X','color','r');
plot(PetL(51:150),PetW(51:150),'o','color','b');
xlabel('PetL');
ylabel('PetW');

Class(1:50) = -1; %Setosa class
Class(51:150) = 1;  %Versi+Virigi class
[S_vs_VV_34_Perce,W(3),epochs(2),missClassified(2)] = Perceptron(...
    [PetL PetW ones(length(meas),1)],Class,[1; 1; 1]);

legend('Setosa','Versi+Virgi','Perceptron d(x)','Location','best');
title('Setosa Vs. Versi+Virgi: Perceptron');
hold off
%% Least Squares for features 3 and 4 : Setosa Vs. Versi+Virgi
figure
hold on
plot(PetL(1:50),PetW(1:50),'X','color','r');
plot(PetL(51:150),PetW(51:150),'o','color','b');
xlabel('PetL');
ylabel('PetW');

[S_vs_VV_34,W(4)] = LS_predict([PetL PetW],Class,2);
missClassifiedClosed(2) = missClassification(51,S_vs_VV_34,2);
    
legend('Setosa','Versi+Virgi','closed form d(x)','Location','best');
title('Setosa Vs. Versi+Virgi: Closed Form');
hold off
%% Perceptron for All Features : Virgi Vs. Versi+Setosa
Class(1:100) = -1; %Versi+Setosa class
Class(101:150) = 1;  %Virgi class
[S_vs_VV_All_Perce,W(5),epochs(3),missClassified(3)] = Perceptron(...
    [meas ones(length(meas),1)],Class,[1; 1; 1; 1; 1]);
%% Least Squares for All Features : Virgi Vs. Versi+Setosa
[V_vs_VS_All,W(6)] = LS_predict(meas,Class,0);
missClassifiedClosed(3) = missClassification(101,V_vs_VS_All,2);
%% Perceptron for features 3 and 4 : Virgi Vs. Versi+Setosa
figure
hold on
plot(PetL(1:100),PetW(1:100),'X','color','r');
plot(PetL(101:150),PetW(101:150),'o','color','b');
xlabel('PetL');
ylabel('PetW');

[S_vs_VV_34_Perce,W(7),epochs(4),missClassified(4)] = Perceptron(...
    [PetL PetW ones(length(meas),1)],Class,[1; 1; 1]);

legend('Versi+Setosa','Virgi','Perceptron d(x)','Location','best');
title('Virgi Vs. Versi+Setosa: Perceptron');
hold off
%% Least Squares for features 3 and 4 : Virgi Vs. Versi+Setosa
figure
hold on
plot(PetL(1:100),PetW(1:100),'X','color','r');
plot(PetL(101:150),PetW(101:150),'o','color','b');
xlabel('PetL');
ylabel('PetW');

[V_vs_VS_34,W(8)] = LS_predict([PetL PetW],Class,2);
missClassifiedClosed(4) = missClassification(101,V_vs_VS_34,2);

legend('Versi+Setosa','Virgi','closed form d(x)','Location','best');
title('Virgi Vs. Versi+Setosa: Closed Form');
hold off
%% Least Squares for features 3 and 4 : Virgi Vs. Vers Vs. Setosa
figure
hold on
plot(PetL(1:50),PetW(1:50),'X','color','r');
plot(PetL(51:100),PetW(51:100),'s','color',[0.5 0 1]);
plot(PetL(101:150),PetW(101:150),'o','color','b');
xlabel('PetL');
ylabel('PetW');

Class = -(ones(150,3));    %Class values = -1, 1
%Class = zeros(150,3);       %Class values = 0, 1                                
Class([1:50],1) = 1; Class([51:100],2) =1; Class([101:150],3) =1;

[S_vs_V_vs_V,W(9)] = LS_predict([PetL PetW],Class,3);
missClassifiedClosed(5) = missClassification([51 101],S_vs_V_vs_V,3);

legend('Versi','Setosa','Virgi','closed form d(x)1'...
    ,'closed form d(x)2','closed form d(x)3','Location','best');
title('Virgi Vs. Vers Vs. Setosa: Closed Form');
hold off
%% LS_predict: Least Squares Method and Visualization
function [Y,W] = LS_predict(X,T,NumOfClass)
X = [X,ones(length(X),1)];
W = inv((X')*X)*X'*T;
Y = X*W;

if NumOfClass == 2   %Plots d(x) in case TwoD is True 
    f = @(x1,x2) W(3) + W(1)*x1 + W(2)*x2;
    h = fimplicit(f,[1 7 0 2.5]);
    h.Color = 'g';
    h.LineWidth = 2;
    
elseif NumOfClass == 3
    f1 = @(x1,x2) W(3,1) + W(1,1)*x1 + W(2,1)*x2;
    h1 = fimplicit(f1,[1 7 0 2.5]);
    h1.Color = 'g';
    h1.LineWidth = 2;
    
    f2 = @(x1,x2) W(3,2) + W(1,2)*x1 + W(2,2)*x2;
    h2 = fimplicit(f2,[1 7 0 2.5]);
    h2.Color = [1 0.5 0];
    h2.LineWidth = 2;
    
    f3 = @(x1,x2) W(3,3) + W(1,3)*x1 + W(2,3)*x2;
    h3 = fimplicit(f3,[1 7 0 2.5]);
    h3.Color = 'black';
    h3.LineWidth = 2;
end

W = mat2cell(W,length(W));
end
%% missClassification: Calculates the number of missClassifications
function num = missClassification(Separation,Predictions,NumOfClasses)
num = 0;
if NumOfClasses == 2
    for i =1:150
     if i < Separation
         if Predictions(i) >  0
            num = num + 1;
         end
     else
          if Predictions(i) < 0
            num = num + 1;
          end
     end
    end
elseif NumOfClasses == 3
    for i = 1:150
        if i < Separation(1)
            if Predictions(i,1) < 0.5
                num = num + 1;
            end
        elseif i > Separation(2)
            if Predictions(i,3) < 0.5
                num = num + 1;
            end
        else
            if Predictions(i,2) < 0.5
                num = num + 1;
            end
        end
    end
end
end

%% Perceptron: Perceptron Algorithm
function [Y,w,iteration,missClassified]= Perceptron(X,y,W)
max_iter=10000; % Maximum number of iterations
rho = 0.1;
X =X';
difference = 1;
w = W;        % Initilaization of the parameter vector
iteration=0;         % Iteration counter
perviousW= 0;
missClassified=length(X);     % Number of misclassfied vectors

while(missClassified > 0)&&(iteration<max_iter)&&(difference > 1e-10 )
    iteration=iteration+1;
    missClassified=0;
    
    gradi= 0; % Computation of the "gradient" term
    for i=1:length(X)
        if((X(:,i)'*w)*y(i)< 0)
            missClassified=missClassified+1;
            gradi=gradi+rho*(-y(i)*X(:,i));
        end
    end    

    previousW = w;
    w=w-rho*gradi; % Updating the parameter vector
    
     difference = abs(norm(w)-norm(previousW));
end
Y = X'*w;
%Visualizing W in case it was 2 dimensional
if (length(w) == 3) 
f = @(x1,x2) w(3) + w(1)*x1 + w(2)*x2;
    h = fimplicit(f,[1 7 0 2.5]);
    h.Color = 'g';
    h.LineWidth = 2;
    if (iteration == max_iter)
    annotation('textbox', [0.35, 0.15, 0.2, 0.2], 'String',...
        "Perceptron did not converge!",'FontSize', 13);
    end
end

w = mat2cell(w,length(w));
end