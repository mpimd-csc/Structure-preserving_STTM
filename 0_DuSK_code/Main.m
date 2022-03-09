function [Bestcv,Besttimetr,Besttimete] = Main(X,label,R,Data_CP);
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
n=size(X,1);                                                                    % Row is data number
c=ones(1,n);  
a=cumsum(c);
c1=-8;                                                                 
c2=8;
g1=-8;
g2=8;
acc=0;
counttimetr=0;                                                                  % Training time
counttimete=0;                                                                  % Test time
R=size(Data_CP{1,1},1);                                                       % The rank of tensor decomposition

rand('state',0);

%% Decompose the input data with CP factorization
addpath('.\cp3_alsls');
Data_CP=cell(n,1);                                                            % Save CP factorization results
fprintf('Decomposing the input data with CP factorization, please wait!\n');
for i=1:n
    [A,B,C,~,~]=cp3_alsls(X{i,1},R);
    Data_CP{i,1}=cell(3,1);
    Data_CP{i,1}{1,1}=A;
    Data_CP{i,1}{2,1}=B;
    Data_CP{i,1}{3,1}=C;
end
clear A B C

%% Train and test
% Repeat t times with k-fold cross validation 
t=10;
k=5;
for i=1:t                                                                        % Repeat t times
    randseed = round(rand(1)*5489);
    elimin_test=Divide(label,k,randseed);                                        % Randomly sample 80% of the whole data as the training set
    b=setdiff(a,elimin_test{k,1});
    Y=X(b,1);
    Y_label=label(b);
    randseed = round(rand(1)*5489);
    Div=Divide(Y_label,k,randseed);                                              % k-fold cross validation
    [~, bestc,bestg]= TrainAverAcc(Y,Y_label,R,k,Data_CP(b,1),Div,c1,c2,g1,g2);  % Train and select the optimal paremeters 
    for j=1:5                                                                    % Extra repeat 5 times to get more stable result
        randseed = round(rand(1)*5489);
        DivOpti=Divide(label,k,randseed);
        [cv, ~,~,timetr,timete]=TrainAverAcc(X,label,R,k,Data_CP,DivOpti,bestc,bestc,bestg,bestg);  
        acc=acc+cv;
        counttimetr=counttimetr+timetr;
        counttimete=counttimete+timete;
    end
    fprintf('The accuracy is %g corresponding to the %g th repeat, bestc is %g, bestg is %g\n',acc/(i*j),i, bestc,bestg);
    clear elimin_test b Y Y_label Div bestc bestg
end

Bestcv = acc/(i*j);
Besttimetr=counttimetr/(i*j);
Besttimete=counttimete/(i*j);
end