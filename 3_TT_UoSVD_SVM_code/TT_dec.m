function [ data_TT , R, pos, core] = TT_dec(X,l,eps)
%% Input
% X: a n*1 cell array for storing third-order tensor data
% Task 1 : Getting compact representation of input data in TT form
n = length(X);
% l is rank truncation 
%% Decompose the input data with TT decomposition
addpath('.\TT-Toolbox-master');
data_TT=cell(n,1);                                                            % Save TT decomposition results
R = cell(n,1);
core = cell(n,1);
posi = cell(1,n);
fprintf('Decomposing the input data with TT decomposition, please wait!\n');
for i=1:n
    tt = tt_tensor(X{i,1},eps); % tt- decomposition of each cell tensor
    tt_round = round(tt,eps,l); % rounding of rank to l
    R{i,1} = tt_round.r;
    posi{i} = tt_round.ps;
    pos = cell2mat(posi).'; % pos is a matrix with position of tt-core(each row is for each tensor) 
    core{i,1} = tt_round.core;
    G = core2cell(tt_round); 
    data_TT{i,1}=cell(3,1);
    data_TT{i,1}{1}=G{1,1};
    data_TT{i,1}{2}=G{2,1};
    data_TT{i,1}{3}=G{3,1};
    
end
clear G
end