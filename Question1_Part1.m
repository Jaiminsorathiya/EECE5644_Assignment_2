%% Q1 - Part A: Bayes min-P(error) classifier + ROC + decision boundary
% I apply the theoretically optimal classifier using the true pdf.
% If Dval10k is not in my workspace, I regenerate it so this runs standalone.

clearvars -except Dval10k; close all; clc;

% ----- Given problem parameters -----
pi0 = 0.6;  pi1 = 0.4;
mu01 = [-0.9; -1.1];  mu02 = [ 0.8;  0.75];
mu11 = [-1.1;  0.9];  mu12 = [-0.9; -0.75];
C = [0.75 0; 0 1.25];
d = 2;

% ----- Ensure validation data exists -----
if ~exist('Dval10k','var') || ~isstruct(Dval10k) || ~isfield(Dval10k,'X')
    rng(42);
    [Dval10k.X, Dval10k.y] = local_gen(10000, [pi0 pi1], ...
        [mu01.'; mu02.'], [mu11.'; mu12.'], C);
end

X = Dval10k.X;        % N x 2
ytrue = Dval10k.y;    % N x 1 in {0,1}
N = size(X,1);

% ----- Stable log-Gaussian pieces -----
logdetC = log(det(C));
const   = -0.5*(d*log(2*pi) + logdetC);  % same for all components
logw    = log(0.5);                      % equal mixture weights

% Row-wise log N(x|mu,C): const - 0.5 * (x-mu)' C^{-1} (x-mu)
% I avoid inv(C) and use /C, which computes (X-mu.')*inv(C) stably.
logN = @(X,mu) const - 0.5 * sum( (X - mu.').*((X - mu.')/C), 2 );

% Log-sum-exp for two terms (elementwise)
logsumexp2 = @(a,b) max(a,b) + log( exp(min(a,b)-max(a,b)) + 1 );

% ----- Log-likelihoods under each class -----
logp0_c1 = logw + logN(X, mu01);
logp0_c2 = logw + logN(X, mu02);
logp1_c1 = logw + logN(X, mu11);
logp1_c2 = logw + logN(X, mu12);

logp0 = logsumexp2(logp0_c1, logp0_c2);
logp1 = logsumexp2(logp1_c1, logp1_c2);

% ----- Bayes discriminant and decisions (min-P(error)) -----
scores = (log(pi1) + logp1) - (log(pi0) + logp0);   % s(x)
yhat   = double(scores >= 0);                        % threshold = 0

TP = sum(yhat==1 & ytrue==1);
TN = sum(yhat==0 & ytrue==0);
FP = sum(yhat==1 & ytrue==0);
FN = sum(yhat==0 & ytrue==1);
minPerror_est = (FP + FN)/N;

fprintf('Confusion (on Dval10k): TP=%d  TN=%d  FP=%d  FN=%d\n', TP,TN,FP,FN);
fprintf('Estimated min-P(error) on validation = %.4f\n', minPerror_est);

% Bayes operating point for plotting
tpr_bayes = TP / (TP + FN);
fpr_bayes = FP / (FP + TN);

% ----- ROC curve from threshold sweep on the same score -----
[thr, idx] = sort(scores, 'descend');
y_sorted = ytrue(idx);
P = sum(ytrue==1);
N0 = sum(ytrue==0);

TP_cum = cumsum(y_sorted==1);
FP_cum = cumsum(y_sorted==0);

TPR = [0; TP_cum./P; 1];
FPR = [0; FP_cum./N0; 1];

figure; hold on; grid on; box on;
plot(FPR, TPR, 'LineWidth', 1.8, 'DisplayName','ROC');
plot([0 1],[0 1], '--', 'DisplayName','Chance');
plot(fpr_bayes, tpr_bayes, 'p', 'MarkerSize', 12, ...
    'MarkerFaceColor',[0.85 0.1 0.1], 'Color',[0.4 0 0], ...
    'DisplayName','Bayes (thr=0)');
xlabel('$\mathrm{FPR}=P(\hat{L}=1\,|\,L=0)$','Interpreter','latex');
ylabel('$\mathrm{TPR}=P(\hat{L}=1\,|\,L=1)$','Interpreter','latex');
title(sprintf('ROC from Bayes discriminant; min-P(error) $\\approx$ %.4f', minPerror_est), ...
      'Interpreter','latex');
legend('Location','southeast');

% ----- Optional: decision boundary overlaid on validation data -----
do_boundary_plot = true;
if do_boundary_plot
    figure; hold on; grid on; box on;
    scatter(X(ytrue==0,1), X(ytrue==0,2), 8, 'filled', 'MarkerFaceAlpha',0.55, 'DisplayName','L=0');
    scatter(X(ytrue==1,1), X(ytrue==1,2), 8, 'filled', 'MarkerFaceAlpha',0.55, 'DisplayName','L=1');

    pad = 0.5;
    x1min = min(X(:,1))-pad; x1max = max(X(:,1))+pad;
    x2min = min(X(:,2))-pad; x2max = max(X(:,2))+pad;
    [gx, gy] = meshgrid(linspace(x1min,x1max,400), linspace(x2min,x2max,400));
    G = [gx(:), gy(:)];

    % Grid scores from the same Bayes rule
    logp0_g = logsumexp2(logw + logN(G, mu01), logw + logN(G, mu02));
    logp1_g = logsumexp2(logw + logN(G, mu11), logw + logN(G, mu12));
    sgrid   = (log(pi1) + logp1_g) - (log(pi0) + logp0_g);
    sgrid   = reshape(sgrid, size(gx));

    contour(gx, gy, sgrid, [0 0], 'k', 'LineWidth', 2);
    xlabel('x_1'); ylabel('x_2');
    title('Validation data with Bayes decision boundary s(x)=0');
    legend('Location','best');
    axis equal;
end

% ========================== Local generator ===============================
function [X,y] = local_gen(N, piL, mu0, mu1, C)
    % Labels from class prior
    y = double(rand(N,1) >= piL(1));  % 0 with prob piL(1), else 1
    % Component per sample (uniform over {1,2})
    k = randi(2, N, 1);
    X = zeros(N,2);
    i = (y==0 & k==1); if any(i), X(i,:) = mvnrnd(mu0(1,:), C, sum(i)); end
    i = (y==0 & k==2); if any(i), X(i,:) = mvnrnd(mu0(2,:), C, sum(i)); end
    i = (y==1 & k==1); if any(i), X(i,:) = mvnrnd(mu1(1,:), C, sum(i)); end
    i = (y==1 & k==2); if any(i), X(i,:) = mvnrnd(mu1(2,:), C, sum(i)); end
end
