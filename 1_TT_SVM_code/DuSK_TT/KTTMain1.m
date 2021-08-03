function [Bestcv,Besttimetr,Besttimete] = KTTMain1(X,label,R,data_TT);
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2012  Lifang He - All rights reserved                       %
% This mustn't be distributed without prior permission of the author        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Input :
%
%          X         the input data cell array, n * 1 --- each array represents a three-order tensor 
%                    n is the number of training examples
%          label     the output labels associated with the input data, n * 1
%          R         the rank of tensor decomposition      
%          Data_CP   the CP factorization result

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          c1, c2 :  the trade-off parameter range [2^c1, 2^(c1+1),..., 2^(c2-1), 2^c2] in SVM model
%          g1, g2 :  the RBF kernel width parameter range [2^g1, 2^(g1+1),..., 2^(g2-1), 2^g2] in SVM model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
% Add Path
% addpath('.\libsvm-mat-3.0-1');

%% Initialize
N=size(X,1);                                                                    % Row is data number
c=ones(1,N);  
a=cumsum(c);
c1=-3;                                                                 
c2=-3;
g1=5;
g2=5;
acc=0;
counttimetr=0;                                                                  % Training time
counttimete=0;                                                                  % Test time                                                       % The rank of tensor decomposition

rand('state',0);

% %% Decompose the input data with CP factorization
% addpath('.\TT-Toolbox-master');
% data_TT=cell(N,1);                                                            % Save CP factorization results
% fprintf('Decomposing the input data with TT factorization, please wait!\n');
% for i=1:N
%     tt = tt_tensor(X{i}, eps);
%      R{i} = tt.r;
%      G = core2cell(tt); 
%     data_TT{i}=cell(3,1);
%     data_TT{i}{1}=G{1};
%     data_TT{i}{2}=G{2};
%     data_TT{i}{3}=G{3};
% end
% clear G

%% Matricization of data_TT to get Kernel form
M =3;
for i = 1:N
  for j = 1:M
    data_TT2{i}{j} = my_matricization(data_TT{i}{j},2);
    end
end
display(data_TT2);

%% Train and test
% Repeat t times with k-fold cross validation 
t=10;
k=5;
for i=1:t                                                                        % Repeat t times
    randseed = round(rand(1)*5489);
    elimin_test=devision(label,k,randseed);                                        % Randomly sample 80% of the whole data as the training set
    b=setdiff(a,elimin_test{k,1});
    Y=X(b,1);
    Y_label=label(b);
    randseed = round(rand(1)*5489);
    Div=devision(Y_label,k,randseed);                                              % k-fold cross validation
    [~, bestc,bestg]= TrainAvgAcu1(Y,Y_label,R,k,data_TT2(b,1),Div,c1,c2,g1,g2);  % Train and select the optimal paremeters 
    for j=1:5                                                                    % Extra repeat 5 times to get more stable result
        randseed = round(rand(1)*5489);
        DivOpti=devision(label,k,randseed);
        [cv, ~,~,timetr,timete]=TrainAvgAcu1(X,label,R,k,data_TT2,DivOpti,bestc,bestc,bestg,bestg);  
        acc=acc+cv;
        counttimetr=counttimetr+timetr;
        counttimete=counttimete+timete;
    end
    fprintf('The accuracy is %g corresponding to the %g th repeat, bestc is %g, bestg is %g\n',acc/(i*j),i, bestc,bestg);
    clear elimin_test b Y Y_label Div bestc bestg
end
