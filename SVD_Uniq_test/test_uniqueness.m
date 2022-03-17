% for r=1:1000
rng(25)            % r = 25 is ok for sigma = 1e-3

x = exp(-(1:4)'*(1:4));      % count this as label 1
xpert = x + 1e-3*randn(4);   % this should have label 1 too
y = exp(-2.5*(1:4)'*(1:4));  % different frequency - count this as label -1
ypert = y + 1e-3*randn(4);   % this should have label -1

% Take 3 columns of u as features
u = [];
[u{1},~,~] = svd(x,'econ');      u{1} = u{1}(:,1:3);
[u{2},~,~] = svd(xpert,'econ');  u{2} = u{2}(:,1:3);  
[u{3},~,~] = svd(y,'econ');      u{3} = u{3}(:,1:3);
[u{4},~,~] = svd(ypert,'econ');  u{4} = u{4}(:,1:3);

% Gaussian kernel
K = [];
for j=1:4
    for i=1:4
        K(i,j) = exp(-norm(u{i}-u{j},'fro').^2/0.5^2);
    end
end
K
urow_orig = u{1}(1,1:3) 
urow_pert = u{2}(1,1:3)

% Uniqueness
for i=1:4
    u{i} = u{i} * diag(1./sign(u{i}(1,:)));
end

% unique Gaussian kernel 
K_unique = [];
for j=1:4
    for i=1:4
        K_unique(i,j) = exp(-norm(u{i}-u{j},'fro').^2/0.5^2);
    end
end
K_unique

u_orig = u{1}(1,1:3)
u_pert = u{2}(1,1:3)
% r
% pause;
% end