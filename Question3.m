%% MAP localization with range sensors — contours & coarse MAP (grid)
% Reproducible simulation and common contour levels across K

clear; clc; close all;
rng(7);                               % reproducibility

% ----- Prior & noise -----
sigma_x = 0.25;
sigma_y = 0.25;
sigma_r = 0.3;                        % range noise std (same for all sensors)

% ----- True position: uniform in unit disk -----
theta  = 2*pi*rand;
rad    = sqrt(rand);                  % sqrt trick -> uniform in disk
x_true = rad*cos(theta);
y_true = rad*sin(theta);

% ----- Grid to evaluate objective on [-2,2]^2 -----
xymin = -2; xymax =  2; ngrid = 301;
xs = linspace(xymin, xymax, ngrid);
ys = linspace(xymin, xymax, ngrid);
[XX,YY] = meshgrid(xs, ys);

Ks = 1:4;
Jcells = cell(numel(Ks),1);
Lcells = cell(numel(Ks),1);
rcells = cell(numel(Ks),1);
mapXY  = zeros(numel(Ks),2);

% ----- Simulate & evaluate objective for each K -----
for k = 1:numel(Ks)
    K = Ks(k);

    % Landmarks evenly spaced on unit circle
    ang = linspace(0, 2*pi, K+1); ang(end) = [];
    L = [cos(ang(:)), sin(ang(:))];           % Kx2, [xi, yi]
    Lcells{k} = L;

    % True ranges
    d_true = hypot(x_true - L(:,1), y_true - L(:,2));

    % Simulated noisy nonnegative ranges
    r = zeros(K,1);
    for i = 1:K
        val = d_true(i) + sigma_r*randn;
        while val < 0
            val = d_true(i) + sigma_r*randn;  % resample if negative
        end
        r(i) = val;
    end
    rcells{k} = r;

    % ---- MAP objective on the grid ----
    % J(x,y) = sum_i (||[x,y]-l_i|| - r_i)^2 / (2*sigma_r^2) + 0.5*(x^2/sx^2 + y^2/sy^2)
    J = 0.5*((XX./sigma_x).^2 + (YY./sigma_y).^2);  % prior term
    for i = 1:K
        di = hypot(XX - L(i,1), YY - L(i,2));
        J  = J + 0.5*((di - r(i)).^2)/(sigma_r^2);
    end
    Jcells{k} = J;

    % Coarse MAP (grid argmin)
    [~, idx] = min(J(:));
    [row, col] = ind2sub(size(J), idx);
    mapXY(k,:) = [xs(col), ys(row)];
end

% ----- Choose common contour levels across all K (trimmed range) -----
allVals = [];
for k = 1:numel(Jcells), allVals = [allVals; Jcells{k}(:)]; end %#ok<AGROW>
allVals = sort(allVals);
N = numel(allVals);
ilo = max(1, round(0.02*N));          % ~2nd percentile (robust to tiny minima)
ihi = min(N, round(0.92*N));          % ~92nd percentile
lo  = allVals(ilo);
hi  = allVals(ihi);
levels = linspace(lo, hi, 14);         % same levels for every plot

% ----- Plot for each K -----
tt = linspace(0, 2*pi, 400);           % unit circle (for context)
xc = cos(tt); yc = sin(tt);

fprintf('True position: (%.3f, %.3f)\n', x_true, y_true);
for k = 1:numel(Ks)
    K   = Ks(k);
    J   = Jcells{k};
    L   = Lcells{k};
    mxy = mapXY(k,:);
    err = hypot(mxy(1)-x_true, mxy(2)-y_true);

    figure('Color','w','Position',[100 100 640 520]);
    [C,h] = contour(XX, YY, J, levels); %#ok<ASGLU>
    clabel(C,h,'FontSize',7); hold on;

    plot(xc, yc, 'k:','LineWidth',1.1);                         % unit circle
    plot(L(:,1), L(:,2), 'o','MarkerSize',6,'LineWidth',1.0);   % landmarks
    plot(x_true, y_true, '+','MarkerSize',10,'LineWidth',1.5);  % true
    plot(mxy(1), mxy(2), 'x','MarkerSize',9,'LineWidth',1.4);   % MAP (grid)

    axis equal; xlim([xymin xymax]); ylim([xymin xymax]);
    xlabel('x'); ylabel('y');
    title(sprintf('MAP objective contours — K = %d', K));
    legend({'Unit circle','Landmarks','True','MAP (grid)'}, 'Location','northwest');
    grid on; box on;

    fprintf('K=%d: MAP(grid) ≈ (%.3f, %.3f); distance to true ≈ %.3f\n', ...
        K, mxy(1), mxy(2), err);
end

%% Notes:
% - Objective being minimized:
%   J(x,y) = sum_i (||[x,y]-l_i|| - r_i)^2 / (2*sigma_r^2) + 0.5*(x^2/sigma_x^2 + y^2/sigma_y^2)
% - Prior stds sigma_x, sigma_y control how strongly the estimate is pulled toward the origin.
% - Contour levels are identical across figures to allow fair visual comparison across K.
