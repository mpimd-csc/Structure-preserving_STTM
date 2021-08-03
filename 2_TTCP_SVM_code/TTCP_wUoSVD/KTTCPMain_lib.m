function [Bestcv,Besttimetr,Besttimete] = KTTCPMain_lib(X,label,l,data_KTTCP,t)
% deatil of the inputs are as following:

%          X           the input data cell array, n * 1 --- each array represents a three-order tensor 
%                      n is the number of training examples
%          label       the output labels associated with the input data, n * 1
%          l           the rank of tensor decomposition  
%          data_KTTCP: kernel projected TT-CP expansion data     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          c1, c2 :  the trade-off parameter range [2^c1, 2^(c1+1),..., 2^(c2-1), 2^c2] in SVM model
%          g1, g2 :  the RBF kernel width parameter range [2^g1, 2^(g1+1),..., 2^(g2-1), 2^g2] in SVM model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
% Add Path
addpath('.\libsvm-master');
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
rand('state',0);  % randomseed
%% Train and test
% Repeat t times with k-fold cross validation 
%t=1;
k=5;                                       % Randomly sample 80% of the whole data as the training set 
for i=1:t                                                                        % Repeat t times
    randseed = round(rand(1)*5489); 
    elimin_test=Divide(label,k,randseed);                                        % Randomly sample 80% of the whole data as the training set
    b=setdiff(a,elimin_test{k,1});
    Y=X(b,:);
    Y_label=label(b);
    randseed = round(rand(1)*5489);
    Div=Divide(Y_label,k,randseed);                                              % k-fold cross validation
    [~, bestc,bestg]= TrainAvgAcuTTCP_wUoSVD_lib(Y,Y_label,l,k,data_KTTCP(b,1),Div,c1,c2,g1,g2);  % Train and select the optimal paremeters 
    for j=1:5                                                                    % Extra repeat 5 times to get more stable result
        randseed = round(rand(1)*5489);
        DivOpti=Divide(label,k,randseed);
        [cv, ~,~,timetr,timete]=TrainAvgAcuTTCP_wUoSVD_lib(X,label,l,k,data_KTTCP,DivOpti,bestc,bestc,bestg,bestg);  
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