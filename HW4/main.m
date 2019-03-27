%% Logistic Regression
%% Load Data
X = importdata('data.txt');
Y = importdata('labels.txt');

X = [X, ones(size(X,1),1)];
%% Set Label 1 and -1
Y(Y < 1) = -1;

%% Select X & Y
test_X = X(2001:4601,:);
test_Y = Y(2001:4601,:);

%% Build Training & Testing
split = [200,500,800,1000,1500,2000];

for i = 1:size(split,2)
    train_X = X(1:split(i),:);
    train_Y = Y(1:split(i),:);
    
    % Train
    weight = logistic_train(train_X, train_Y);
    
    % Test
    logit = -test_X*weight;
    logit(logit > 10) = 10;
    %logit(logit < -10) = -10;
    predict_y = 1.0./(1.0+exp(logit));
    predict_y(predict_y > 0.5) = 1;
    predict_y(predict_y < 1) = -1;
    
    % Accuracy
    accuracy = sum(predict_y == test_Y)/size(test_Y, 1);
    disp(['Logistic Regression ', num2str(split(i)), ' samples | accuracy: ', num2str(accuracy)]);
end

%% Logistic Regression + L1
load('ad_data.mat')
par  = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
figure();
for i = 1:size(par,2)
    [w, c] = logistic_l1_train(X_train, y_train, par(i));
    logit = -X_test*w;
    logit(logit > 10) = 10;
    logit(logit < -10) = -10;
    predict_y = 1.0./(1.0+exp(logit));
    
    [X,Y,T,AUC] = perfcurve(y_test, predict_y, 1);
    subplot(4,3,i);
    plot(X,Y);
    xlabel('False positive rate'); 
    ylabel('True positive rate');
    title(['ROC alpha: ', num2str(par(i))]);
    disp(['num of feature selected: ', num2str(sum(w ~= 0)), ' AUC: ', num2str(AUC)]);
end