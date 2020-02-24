%% Farshad Bolouri - R11630884 - Machine Learning - Project 2
clear all
rng(31);
%% Dataset 1 ,N_train = 10
X_train= rand(10,1);  
T_train = sin(2*pi*X_train) + mvnrnd(0,0.3,10);

X_test= rand(100,1);
T_test = sin(2*pi*X_test) + mvnrnd(0,0.3,100);
X_train = zscore(X_train);

[Y_train, Y_test] = Predict(X_train,T_train,X_test);

[Train_Error, Test_Error] = CalError(Y_train,Y_test,T_train,T_test...
    ,X_train,X_test);
E_train = sqrt(Train_Error/length(X_train));
M = 0:9;
hold on
%ylim([0 100])
plot(M,E_train,'b-o','LineWidth',2);
xlabel('M');ylabel('E-RMS');
E_test = sqrt(Test_Error/length(X_train));
plot(M,E_test,'r-o','LineWidth',2);
title('N-Train = 10')
legend('Training','Test');
hold off
%% Dataset 2 ,N_train = 100
X_train= rand(100,1);  
T_train = sin(2*pi*X_train) + mvnrnd(0,0.3,100);

X_test= rand(100,1);
T_test = sin(2*pi*X_test) + mvnrnd(0,0.3,100);

[Y_train, Y_test] = Predict(X_train,T_train,X_test);



[Train_Error, Test_Error] = CalError(Y_train,Y_test,T_train,T_test...
    ,X_train,X_test);
E_train = sqrt(Train_Error/length(X_train));
M = 0:9;
figure
hold on
plot(M,E_train,'b-o','LineWidth',2);
xlabel('M');ylabel('E-RMS');
E_test = sqrt(Test_Error/length(X_train));
plot(M,E_test,'r-o','LineWidth',2);
title('N-Train = 100')
legend('Training','Test');
hold off



function [Y_train, Y_test] = Predict(X_train,T_train,X_test)
%% M = 0
phi0 = ones(size(X_train));
phi0_test =ones(size(X_test));

W0 = inv((phi0')*phi0)*phi0'*T_train;
Y0 = phi0*W0;
Y0_test = phi0_test*W0;
%% M = 1
phi1 = [ones(size(X_train)) , X_train];
phi1_test =[ones(size(X_test)),X_test];

W1 = inv((phi1')*phi1)*phi1'*T_train;
Y1 = phi1*W1;
Y1_test = phi1_test*W1;
%% M = 2
phi2 = [ones(size(X_train)) , X_train, X_train.^2];
phi2_test =[ones(size(X_test)),X_test, X_test.^2];

W2 = inv((phi2')*phi2)*phi2'*T_train;
Y2 = phi2*W2;
Y2_test = phi2_test*W2;
%% M = 3
phi3 = [ones(size(X_train)) , X_train, X_train.^2, X_train.^3];
phi3_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3];

W3 = inv((phi3')*phi3)*phi3'*T_train;
Y3 = phi3*W3;
Y3_test = phi3_test*W3;
%% M = 4
phi4 = [ones(size(X_train)) , X_train, X_train.^2, X_train.^3, X_train.^4];
phi4_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4];

W4 = inv((phi4')*phi4)*phi4'*T_train;
Y4 = phi4*W4;
Y4_test = phi4_test*W4;
%% M = 5
phi5 =[ones(size(X_train)), X_train, X_train.^2, X_train.^3, X_train.^4,...
    X_train.^5];
phi5_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4,...
    X_test.^5];

W5 = inv((phi5')*phi5)*phi5'*T_train;
Y5 = phi5*W5;
Y5_test = phi5_test*W5;
%% M = 6
phi6 =[ones(size(X_train)), X_train, X_train.^2, X_train.^3, X_train.^4,...
    X_train.^5, X_train.^6];
phi6_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4,...
    X_test.^5, X_test.^6];

W6 = inv((phi6')*phi6)*phi6'*T_train;
Y6 = phi6*W6;
Y6_test = phi6_test*W6;
%% M = 7
phi7 =[ones(size(X_train)), X_train, X_train.^2, X_train.^3, X_train.^4,...
    X_train.^5, X_train.^6, X_train.^7];
phi7_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4,...
    X_test.^5, X_test.^6, X_test.^7];

W7 = inv((phi7')*phi7)*phi7'*T_train;
Y7 = phi7*W7;
Y7_test = phi7_test*W7;
%% M = 8
phi8 =[ones(size(X_train)), X_train, X_train.^2, X_train.^3, X_train.^4,...
    X_train.^5, X_train.^6, X_train.^7, X_train.^8];
phi8_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4,...
    X_test.^5, X_test.^6, X_test.^7, X_test.^8];

W8 = inv((phi8')*phi8)*phi8'*T_train;
Y8 = phi8*W8;
Y8_test = phi8_test*W8;
%% M = 9
phi9 =[ones(size(X_train)), X_train, X_train.^2, X_train.^3, X_train.^4,...
    X_train.^5, X_train.^6, X_train.^7, X_train.^8, X_train.^9];
phi9_test =[ones(size(X_test)),X_test, X_test.^2, X_test.^3, X_test.^4,...
    X_test.^5, X_test.^6, X_test.^7, X_test.^8, X_test.^9];

W9 = inv((phi9')*phi9)*phi9'*T_train;
Y9 = phi9*W9;
Y9_test = phi9_test*W9;

Y_train = [Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9];
Y_test = [Y0_test,Y1_test,Y2_test,Y3_test,Y4_test,Y5_test,Y6_test,...
    Y7_test,Y8_test,Y9_test];
end

%% Training and Testing error calculation
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