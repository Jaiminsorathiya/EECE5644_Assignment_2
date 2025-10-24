%% Q1 - Part 2: Logistic (linear & quadratic) posterior approximations
% I train six classifiers:
%   Linear and Quadratic logits on Dtrain^50, Dtrain^500, Dtrain^5000
% I evaluate each on Dval^10k and report error & confusion.
% Decision-boundary plots now take Dval10k as an argument (no globals).

clearvars -except Dtrain50 Dtrain500 Dtrain5k Dval10k; close all; clc;

% ----------------------- Ensure datasets exist ----------------------------
pi0 = 0.6; pi1 = 0.4;
mu01 = [-0.9 -1.1]; mu02 = [0.8 0.75];
mu11 = [-1.1 0.9];  mu12 = [-0.9 -0.75];
C = [0.75 0; 0 1.25];

if ~exist('Dtrain50','var') || ~isfield(Dtrain50,'X')
    rng(42);
    [Dtrain50.X,  Dtrain50.y ]  = local_gen(50,   [pi0 pi1], [mu01; mu02], [mu11; mu12], C);
    [Dtrain500.X, Dtrain500.y ] = local_gen(500,  [pi0 pi1], [mu01; mu02], [mu11; mu12], C);
    [Dtrain5k.X,  Dtrain5k.y ]  = local_gen(5000, [pi0 pi1], [mu01; mu02], [mu11; mu12], C);
end
if ~exist('Dval10k','var') || ~isfield(Dval10k,'X')
    rng(43); % different seed is fine
    [Dval10k.X,   Dval10k.y ]   = local_gen(10000,[pi0 pi1], [mu01; mu02], [mu11; mu12], C);
end

% ----------------------- Feature maps -------------------------------------
phi_lin  = @(X) [ones(size(X,1),1), X(:,1), X(:,2)];
phi_quad = @(X) [ones(size(X,1),1), X(:,1), X(:,2), X(:,1).^2, X(:,2).^2, X(:,1).*X(:,2)];

% ----------------------- Training & evaluation helper ---------------------
train_and_eval = @(trainSet, phi, name) do_train_eval(trainSet, phi, name, Dval10k);

% ----------------------- Run the six experiments --------------------------
results = struct();

results.lin50   = train_and_eval(Dtrain50,  phi_lin,  'Linear on Dtrain^{50}');
results.lin500  = train_and_eval(Dtrain500, phi_lin,  'Linear on Dtrain^{500}');
results.lin5k   = train_and_eval(Dtrain5k,  phi_lin,  'Linear on Dtrain^{5000}');

results.quad50  = train_and_eval(Dtrain50,  phi_quad, 'Quadratic on Dtrain^{50}');
results.quad500 = train_and_eval(Dtrain500, phi_quad, 'Quadratic on Dtrain^{500}');
results.quad5k  = train_and_eval(Dtrain5k,  phi_quad, 'Quadratic on Dtrain^{5000}');

% ----------------------- Print a compact summary table --------------------
fprintf('\n================ Summary on Dval^{10k} ================\n');
fprintf('%-28s  Err    TPR    FPR    TP    TN    FP    FN\n', 'Model');
print_row(results.lin50);
print_row(results.lin500);
print_row(results.lin5k);
print_row(results.quad50);
print_row(results.quad500);
print_row(results.quad5k);

% ----------------------- Optional decision-boundary plots -----------------
do_boundary_plots = true;
if do_boundary_plots
    plot_boundary(Dtrain50,  results.lin50,  phi_lin,  'Linear boundary (train 50)',   Dval10k);
    plot_boundary(Dtrain500, results.lin500, phi_lin,  'Linear boundary (train 500)',  Dval10k);
    plot_boundary(Dtrain5k,  results.lin5k,  phi_lin,  'Linear boundary (train 5000)', Dval10k);

    plot_boundary(Dtrain50,  results.quad50,  phi_quad, 'Quadratic boundary (train 50)',   Dval10k);
    plot_boundary(Dtrain500, results.quad500, phi_quad, 'Quadratic boundary (train 500)',  Dval10k);
    plot_boundary(Dtrain5k,  results.quad5k,  phi_quad, 'Quadratic boundary (train 5000)', Dval10k);
end

% ====================== Local functions ===================================
function out = do_train_eval(trainSet, phi, label, Dval10k)
    Xtr = trainSet.X;  ytr = trainSet.y(:);
    Xva = Dval10k.X;   yva = Dval10k.y(:);

    % Design matrices
    Phi_tr = phi(Xtr);
    Phi_va = phi(Xva);

    % Initial theta
    theta0 = zeros(size(Phi_tr,2),1);

    % Negative log-likelihood (with tiny eps for stability)
    epsl = 1e-12;
    sigmoid = @(z) 1./(1+exp(-z));
    nll = @(th) -sum( ytr.*log(max(sigmoid(Phi_tr*th),epsl)) + ...
                     (1-ytr).*log(max(1 - sigmoid(Phi_tr*th),epsl)) );

    % Optimize theta with fminsearch (no toolbox needed)
    opts = optimset('Display','off','MaxFunEvals',2e5,'MaxIter',2e5);
    theta = fminsearch(nll, theta0, opts);

    % Validation predictions (0.5 threshold approximates min-error rule)
    pva  = sigmoid(Phi_va*theta);
    yhat = double(pva >= 0.5);

    TP = sum(yhat==1 & yva==1);
    TN = sum(yhat==0 & yva==0);
    FP = sum(yhat==1 & yva==0);
    FN = sum(yhat==0 & yva==1);
    N  = numel(yva);

    err = (FP+FN)/N;
    tpr = TP/(TP+FN);
    fpr = FP/(FP+TN);

    fprintf('%-28s  err=%.4f  TPR=%.3f  FPR=%.3f  TP=%d  TN=%d  FP=%d  FN=%d\n', ...
            label, err, tpr, fpr, TP, TN, FP, FN);

    out = struct('label',label,'theta',theta,'err',err,'TP',TP,'TN',TN,'FP',FP,'FN',FN, ...
                 'tpr',tpr,'fpr',fpr);
end

function print_row(r)
    fprintf('%-28s  %.4f  %.3f  %.3f  %4d  %4d  %4d  %4d\n', ...
        r.label, r.err, r.tpr, r.fpr, r.TP, r.TN, r.FP, r.FN);
end

function plot_boundary(trainSet, model, phi, ttl, Dval10k)
    % Combine train + validation for plotting extent
    X = [trainSet.X; Dval10k.X];
    y = [trainSet.y; Dval10k.y];

    figure; hold on; grid on; box on;
    scatter(X(y==0,1), X(y==0,2), 8, 'filled', 'MarkerFaceAlpha',0.45, 'DisplayName','L=0');
    scatter(X(y==1,1), X(y==1,2), 8, 'filled', 'MarkerFaceAlpha',0.45, 'DisplayName','L=1');

    pad = 0.5;
    x1min = min(X(:,1))-pad; x1max = max(X(:,1))+pad;
    x2min = min(X(:,2))-pad; x2max = max(X(:,2))+pad;
    [gx, gy] = meshgrid(linspace(x1min,x1max,400), linspace(x2min,x2max,400));
    G = [gx(:), gy(:)];

    Phi_g = phi(G);
    pgrid = 1./(1+exp(-Phi_g*model.theta));
    pgrid = reshape(pgrid, size(gx));

    contour(gx, gy, pgrid, [0.5 0.5], 'k', 'LineWidth', 2);
    title(sprintf('%s  |  %s  |  err=%.4f', ttl, model.label, model.err));
    xlabel('x_1'); ylabel('x_2'); legend('Location','best'); axis equal;
end

function [X,y] = local_gen(N, piL, mu0, mu1, C)
    % Sample labels from prior
    y = double(rand(N,1) >= piL(1));
    % Component choice (uniform over {1,2})
    k = randi(2, N, 1);
    X = zeros(N,2);
    if any(y==0 & k==1), X(y==0 & k==1,:) = mvnrnd(mu0(1,:), C, sum(y==0 & k==1)); end
    if any(y==0 & k==2), X(y==0 & k==2,:) = mvnrnd(mu0(2,:), C, sum(y==0 & k==2)); end
    if any(y==1 & k==1), X(y==1 & k==1,:) = mvnrnd(mu1(1,:), C, sum(y==1 & k==1)); end
    if any(y==1 & k==2), X(y==1 & k==2,:) = mvnrnd(mu1(2,:), C, sum(y==1 & k==2)); end
end
