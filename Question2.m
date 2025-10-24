% ------------------- HW2 Q2: ML vs MAP (Ridge) -------------------
clear; clc; close all;
rng(0);                       % reproducible sampling

% ----- Generate data (given helper) -----
Ntrain = 100;
Nvalidate = 1000;
[xTrain,yTrain,xVal,yVal] = hw2q2(Ntrain, Nvalidate);   % x: 2xN, y: 1xN

% ----- Design matrices with cubic features -----
PhiTrain = cubicFeatures(xTrain);        % Ntrain x 10
PhiVal   = cubicFeatures(xVal);          % Nvalidate x 10
yT = yTrain(:);
yV = yVal(:);

% ----- ML (OLS) -----
w_ml = (PhiTrain' * PhiTrain) \ (PhiTrain' * yT);
pred_ml_tr  = PhiTrain * w_ml;
pred_ml_val = PhiVal   * w_ml;

mse_ml_tr  = mean((yT - pred_ml_tr ).^2);
mse_ml_val = mean((yV - pred_ml_val).^2);

% Optional: estimate sigma^2 from ML residuals (for λ scaling)
sigma2_ml = mean((yT - PhiTrain*w_ml).^2);

% ----- MAP (ridge) for a sweep of γ values -----
% Prior: w ~ N(0, γ I)  =>  MAP = (Φ'Φ + (σ²/γ) I)^{-1} Φ' y
m = 6; n = 6;                                     % sweep 10^{-6} ... 10^{+6}
gammas = logspace(-m, n, 61);                     % 61 log-spaced points
mse_map_tr  = zeros(size(gammas));
mse_map_val = zeros(size(gammas));

I = eye(size(PhiTrain,2));
for k = 1:numel(gammas)
    gamma  = gammas(k);
    lambda = sigma2_ml / gamma;                   % λ = σ²/γ

    w_map = (PhiTrain' * PhiTrain + lambda*I) \ (PhiTrain' * yT);

    pred_tr  = PhiTrain * w_map;
    pred_val = PhiVal   * w_map;

    mse_map_tr(k)  = mean((yT - pred_tr ).^2);
    mse_map_val(k) = mean((yV - pred_val).^2);
end

% ----- Select best γ by validation MSE -----
[best_mse, idx] = min(mse_map_val);
best_gamma = gammas(idx);
fprintf('ML  training MSE: %.6f\n', mse_ml_tr);
fprintf('ML  validation MSE: %.6f\n', mse_ml_val);
fprintf('Best MAP validation MSE: %.6f at gamma = %.3e\n', best_mse, best_gamma);

% ===================== Plots (no TeX/LaTeX) =====================

% 1) Validation error vs gamma with ML baseline
figure('Name','Validation MSE vs gamma');
semilogx(gammas, mse_map_val, 'LineWidth', 2); hold on; grid on;
yline(mse_ml_val, '--', 'ML (no regularization)', 'LineWidth', 1.5);
xline(best_gamma, ':', sprintf('γ* = %.2e', best_gamma), 'LineWidth', 1.0);
xlabel('γ (prior covariance scale)', 'Interpreter','none');
ylabel('Validation MSE', 'Interpreter','none');
title('MAP (ridge) validation error vs γ,  λ = σ²/γ', 'Interpreter','none');
legend({'MAP(γ)','ML baseline'}, 'Location','best', 'Interpreter','none');

% 2) Training vs Validation error vs gamma
figure('Name','Training vs Validation MSE vs gamma');
semilogx(gammas, mse_map_tr,  'LineWidth', 2); hold on; grid on;
semilogx(gammas, mse_map_val, 'LineWidth', 2);
yline(mse_ml_tr,  '--', 'ML train', 'LineWidth', 1.2);
yline(mse_ml_val, '--', 'ML val',   'LineWidth', 1.2);
xline(best_gamma, ':', sprintf('γ* = %.2e', best_gamma), 'LineWidth', 1.0);
xlabel('γ (prior covariance scale)', 'Interpreter','none');
ylabel('MSE', 'Interpreter','none');
title('Training vs Validation MSE across MAP(γ)', 'Interpreter','none');
legend({'MAP train','MAP val','ML train','ML val'}, 'Location','best', 'Interpreter','none');

% ===================== Helper =====================
function Phi = cubicFeatures(X)
% X: 2xN  ->  Phi: N x 10  (all monomials up to total degree 3)
x1 = X(1,:)'; x2 = X(2,:)';
Phi = [ ones(size(x1)) , ...
        x1 , x2 , ...
        x1.^2 , x1.*x2 , x2.^2 , ...
        x1.^3 , (x1.^2).*x2 , x1.*(x2.^2) , x2.^3 ];
end
