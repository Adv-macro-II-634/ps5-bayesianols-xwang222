% I sincerely appreciate Mingyang's generous help.
clear, clc, close all
tic

filename = 'I:\SUNY\2019 fall\SCF\ps5-bayesianols-xwang222\card.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%*s%*s%*s%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%f%f%f%*s%*s%*s%*s%*s%*s%*s%f%f%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
educ = dataArray{:, 1};
black = dataArray{:, 2};
smsa = dataArray{:, 3};
south = dataArray{:, 4};
exper = dataArray{:, 5};
lwage = dataArray{:, 6};


% OLS Regression
Y = lwage; % Dependent variable log(wage)
X = [ones(length(Y),1) educ exper smsa black south]; % Independent variables
[length_X,width_X] = size(X); % Numbers of observations and betas
b = (X'*X)\(X'*Y); % OLS Estimators
u = Y-X*b; % OLS Residuals
s2 = u'*u/(length_X-width_X); %5vd
sigma = sqrt(s2); % Residual standard error
vcv = s2*inv(X'*X); % Variance-covariance matrix of betas
se = sqrt(diag(vcv)); % Standard errors of betas
% How to calculate (and why)? Since the residual variance is one parameter of interest.
var_res = 2/(length_X-width_X)*s2^2; % Variance of residual variance (s2)
se_res = sqrt(var_res); % Standard error of residual variance (s2)

% 2.(a) Metropolis-Hastings Algorithm with flat priors
N = 100000; % Number of trials
theta = [b;s2]'; % Parameters to estimate
var_theta = diag([diag(vcv);var_res]'); % Standard errors of parameters
var_theta_noeduc = var_theta;
var_theta_noeduc(2,:)=[];
var_theta_noeduc(:,2)=[];
mu = zeros(1,length(theta)); % Mean of Gaussian distribution
mu_new=zeros(1,length(theta)-1);
acc = 0; % Number of acceptances
theta_trial = zeros(N,length(theta)); % Proposal matrix
theta_trial(1,:) = theta; % Initialize with estimates
theta_acc = zeros(N,length(theta)); % Acceptance matrix
theta_acc(1,:) = theta; % Initialize with estimates
guess = .11; % Adjust to get an acceptance rate in 20%-25%
var_guess = guess * var_theta; % Adjusted standard errors of parameters
var_guess_noeduc = guess * var_theta_noeduc; % Adjusted standard errors of parameters

for i = 1:(N-1)
    % Draw from Gaussian and set proposal
    theta_trial(i+1,:) = theta_acc(i,:) + mvnrnd(mu,var_guess);
    % Keep variance of residual variance nonnegative
    while theta_trial(i,length(theta)) < 0
        theta_trial(i+1,:) = theta_acc(i,:) + mvnrnd(mu,var_guess);
    end
    ratio = exp(normlike([mu(length(theta));sqrt(theta_acc(i,length(theta)))],...
        Y - X * (theta_acc(i,1:length(theta)-1))') - ...
        normlike([mu(length(theta));sqrt(theta_trial(i+1,length(theta)))],...
        Y - X * (theta_trial(i+1,1:length(theta)-1))'));
    x = rand; % Random number uniformly distributed in (0,1)
    if x < ratio
        theta_acc(i+1,:) = theta_trial(i+1,:);
        acc = acc + 1;
    else
        theta_acc(i+1,:) = theta_acc(i,:);
    end
end
acc_rate = acc/N;

% 2.(b) Metropolis-Hastings Algorithm with given prior of beta_educ
% Prior of beta_educ
mean = 0.06;                            % Mean of prior
alpha = 0.05;                           % Significance level
cv = norminv(1-alpha/2);                % RHS critical value
CI_h = 0.085;                           % Higher bound of confidence interval
sd = 2*((CI_h-mean)/cv);                % se of beta_educ
%p = @(x)(log(normpdf(x,mean,sd)));      % PDF of prior

acc_new = 0; % Number of acceptances
theta_pro_new = zeros(N,length(theta)); % Proposal matrix
theta_pro_new(1,:) = theta; % Initialize with estimates
theta_hat_new = zeros(N,length(theta)); % Acceptance matrix
theta_hat_new(1,:) = theta; % Initialize with estimates

for i = 1:(N-1)
    % Draw from Gaussian and set proposal
    % since I have prior for educ only, I propose the rest of the
    % parameters from flat normal. For the educ, I propose from the
    % distribution defined by the know prior
    theta_pro_new(i+1,[1 3:7]) = theta_hat_new(i,[1 3:7]) + mvnrnd(mu_new,var_guess_noeduc);
    %theta_pro_new(i+1,3:7) = theta_hat_new(i,3:7) + mvnrnd(mu,var_guess);
    theta_pro_new(i+1,2) = theta_hat_new(i,2) + normrnd(mean,sd);
    
    % Keep variance of residual variance nonnegative
    while theta_pro_new(i,length(theta)) < 0
        theta_pro_new(i+1,:) = theta_hat_new(i,:) + mvnrnd(mu,var_guess);
    end
    ratio_new = exp(normlike([mu(length(theta));sqrt(theta_hat_new(i,length(theta)))],...
        Y - X * (theta_hat_new(i,1:length(theta)-1))') - ...
        normlike([mu(length(theta));sqrt(theta_pro_new(i+1,length(theta)))],...
        Y - X * (theta_pro_new(i+1,1:length(theta)-1))'));
    %+ ...
     %   p(theta_pro_new(i+1,2)) - p(theta_hat_new(i,2)));
    x_new = rand; % Random number uniformly distributed in (0,1)
    if x_new < ratio_new
        theta_hat_new(i+1,:) = theta_pro_new(i+1,:);
        acc_new = acc_new + 1;
    else
        theta_hat_new(i+1,:) = theta_hat_new(i,:);
    end
end
acc_rate_new = acc_new/N;

% Plot posterior distributons with flat priors and given prior of beta_educ
% for the plotting part, I should give credit to Mingyang

figure
subplot(1,2,1)
histfit(theta_acc(:,1),250,'kernel')
hold on
line([theta(1) theta(1)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_0 with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,1),250,'kernel')
hold on
line([theta(1) theta(1)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_0 with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,2),250,'kernel')
hold on
line([theta(2) theta(2)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{educ} with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,2),250,'kernel')
hold on
line([theta(2) theta(2)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{educ} with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,3),250,'kernel')
hold on
line([theta(3) theta(3)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{exp} with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,3),250,'kernel')
hold on
line([theta(3) theta(3)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{exp} with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,4),250,'kernel')
hold on
line([theta(4) theta(4)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{SMSA} with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,4),250,'kernel')
hold on
line([theta(4) theta(4)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{SMSA} with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,5),250,'kernel')
hold on
line([theta(5) theta(5)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{black} with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,5),250,'kernel')
hold on
line([theta(5) theta(5)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{black} with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,6),250,'kernel')
hold on
line([theta(6) theta(6)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{south} with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,6),250,'kernel')
hold on
line([theta(6) theta(6)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \beta_{south} with Given Prior of \beta_{educ}')

figure
subplot(1,2,1)
histfit(theta_acc(:,7),250,'kernel')
hold on
line([theta(7) theta(7)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \sigma_{\epsilon}^2 with Flat Priors')

subplot(1,2,2)
histfit(theta_hat_new(:,7),250,'kernel')
hold on
line([theta(7) theta(7)],ylim,'Color','y','LineWidth',2)
title('Posterior Distribution of \sigma_{\epsilon}^2 with Given Prior of \beta_{educ}')

toc