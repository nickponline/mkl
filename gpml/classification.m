function [varargout] = classification(filepath)

disp(filepath);
close all;
setenv("GNUTERM", "x11");

z  = load(filepath);
y  = z.y;
x1 = z.X(:, z.y == 1);
x2 = z.X(:, z.y == -1);

% % Normalization
% [r, c] = size(x2);
% x2 = (x2) - repmat(mean(x2), r, 1) ./ repmat(std(x2), r, 1);
% [r, c] = size(x1);
% x1 = (x1) - repmat(mean(x1), r, 1) ./ repmat(std(x1), r, 1);
% x1(isnan(x1)) = 0; 
% x2(isnan(x2)) = 0; 

spread = max(z.X') - min(z.X');


n1 = length(x1)
n2 = length(x2)
d1 = size(x1, 2)
d2 = size(x2, 2)

%x1 = (x1 - repmat(mean(x1), n1, 1)) ./ repmat(std(x1), n1, 1);
%x2 = (x2 - repmat(mean(x2), n2, 1)) ./ repmat(std(x2), n2, 1);

x1 = x1';
x2 = x2';

S1 = eye(d1); 
S2 = eye(d2);

m1 = ones(d1, 1); 
m2 = ones(d2, 1);

x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
%plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on;
%plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);

% [t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
% t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
% tmm = bsxfun(@minus, t, m1');
% p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
% tmm = bsxfun(@minus, t, m2');
% p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
% set(gca, 'FontSize', 24);
% contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9]);
% [c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
% set(h, 'LineWidth', 2)
% colorbar
% grid
% axis([-4 4 -4 4])
%print -depsc f5.eps

results = [];
for k = 1:20 

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([ones(1, d1) 1]);
likfunc = @likErf;
hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);

data = exp(hyp.cov(1:(length(hyp.cov)-1)));

k;
results = [results ; data];

end 

save(strcat(filepath, ".sav"), "results", "spread")
strcat(filepath, ".sav")

% [a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));

% figure
% set(gca, 'FontSize', 24);
% plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on;
% plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
% contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
% [c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
% set(h, 'LineWidth', 2)
% colorbar
% grid
% axis([-4 4 -4 4])
% %print -depsc f6.eps

end
