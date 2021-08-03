function [ data_KTTCP ] = KTTCP( X, data_TTCP, kerfilter);

%   Detailed explanation goes here
%% Input
% X: a N*1 cell array for storing tensor data
% data_TTCP: The TT-CP expansion of the input tensor.
% This code is first part of algorithm in which we take
% identity/random/gaussian/covariance kernel to project over the TT-CP expansion's cores.
%%
N = length(X);
stdev = 1;
dimn=size(X{1});
M = length(dimn);
K =cell(M,1);
invK = cell(M,1);
data_KTTCP = cell(N,1);

%% Four methods for defining common matrices
switch kerfilter

% Random normalized method
    case 'normalize'
for i = 1:M
    K_temp = randn(dimn(i),dimn(i)) * stdev;
    K_temp = K_temp * diag(sqrt(1 ./ (sum(K_temp.^2) + 1e-6)));
    K{i,1} = K_temp * K_temp';
   % invK{i,1} = inv(K{i,1});
         for j=1:N
             data_KTTCP{j,1}{i,1} = K{i,1}\data_TTCP{j,1}{i,1};  
         end
end


% Gaussian kernel method 
    case 'gauss'
for i = 1:M
    K_temp = zeros(randn(dimn(i),dimn(i)));
    for j = 1:N
         K_temp =  K_temp + gauss(data_TTCP{j,1}{i,1}.');
    end
    K{i,1} = K_temp./N;
    %invK{i,1} = inv(K{i,1});
    for j=1:N
        data_KTTCP{j,1}{i,1} = K{i,1}\data_TTCP{j,1}{i,1};  
    end
end


% Covariance method
    case 'covariance'
for i = 1:M
    K_temp = zeros(dimn(i),dimn(i));
    for j = 1:N
         K_temp =  K_temp + cov(data_TTCP{j,1}{i,1}');
    end
    K{i,1} = K_temp./N;
    %invK{i,1} = inv(K{i,1});
    for j=1:N
        data_KTTCP{j,1}{i,1} = K{i,1}\data_TTCP{j,1}{i,1};  
    end
    
end


% Inverse TT approximation
    case 'identity'
for i = 1:M
    K{i,1} = eye(dimn(i));
    for j=1:N
        data_KTTCP{j,1}{i,1} = K{i,1}\data_TTCP{j,1}{i,1};  
    end
end


% random kernel filtering 
    case 'random'
for i = 1:M
    K{i,1} = randn(dims(i),dims(i))*randn(dims(i),dims(i))' * stdev;
%     invK{i,1} = inv(K{i,1});
    for j=1:N
        data_KTTCP{j,1}{i,1} = K{i,1}\data_TTCP{j,1}{i,1};  
    end
end 


end % end for switch function
end

 