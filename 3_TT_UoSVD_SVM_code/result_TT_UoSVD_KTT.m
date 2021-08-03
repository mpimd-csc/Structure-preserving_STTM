%%%% This is the file which produce KTT results (TT decomposiiton + kernel approximation)%%%%
%%%% along with the uniqueness of SVD constraint in the TT-toolbox %%%%

% The codes are for the paper
% "Efficient Structure-preserving Support Tensor Train Machine" by K.Kour,
% S.Dolgov, M. Stoll and P.Benner

% Code author @Kirandeep Kour

% Step 1. Setup the TT toolbox
% Step 2. run make file in matlab folder in libsvm-master
% Step 3. Add complete folder and subfolders into path
addpath(genpath('../../Toolboxes/TT-Toolbox-master_UoSVD'))
addpath(genpath('../../Toolboxes/libsvm-master/'))

 datatype = 'adni';

switch datatype
    %% HSI Indiana dataset
    case 'hsi_indiana'
% loading Indiana data file
load('HSI_Indiana.mat')
n = size(X,1);

% Computing TT decomposition
eps = 0; 
trunc = 2; % while fixing rank 
dimn = size(X{1});
% Repeat t times with k-fold cross validation 
t = 10; % number of repitition of whole procedure
global l % the rank has been defined as a global variable
for l = 1:10
[data_TT,~] = TT_dec(X,l,eps); %TT factorization of input 

% computing TT matricized of input tensor
[ data_TT2 ] = TTmat( X, data_TT);

% main results including training and testing
[CVofTT_UoSVD_HSI_indiana,trainTIMEofTT_UoSVD_HSI_indiana,testTIMEofTT_UoSVD_HSI_indiana] = KTTMain_lib(X,label,data_TT2,data_TT,t);

end
save('CVofTT_UoSVD_HSI_indiana.mat','CVofTT_UoSVD_HSI_indiana') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTT_UoSVD_HSI_indiana == max(CVofTT_UoSVD_HSI_indiana)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTT_UoSVD_HSI_indiana, '--s',...
    'MarkerIndices',1:1:length(l),...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerFaceColor',[0.5,0.5,0.5],...
    'MarkerEdgeColor','b',...
    'MarkerIndices',[idxmax1],...
    'MarkerFaceColor','red',...
    'MarkerSize', 5)

%%
%% HSI Salines dataset
    case 'hsi_salines'
% loading Salines data file
load('HSI_Salines.mat')
n = size(X,1);

% Computing TT decomposition
eps = 0; 
trunc = 2; % while fixing rank 
dimn = size(X{1});
% Repeat t times with k-fold cross validation 
t = 10; % number of repitition of whole procedure
global l % the rank has been defined as a global variable
for l = 1:10
[data_TT,~] = TT_dec(X,l,eps); %TT factorization of input 

% computing TT matricized of input tensor
[ data_TT2 ] = TTmat( X, data_TT);


% main results including training and testing
[CVofTT_UoSVD_HSI_Salines,trainTIMEofTT_UoSVD_HSI_salines,testTIMEofTT_UoSVD_HSI_salines] = KTTMain_lib(X,label,data_TT2, data_TT,t);

end
save('CVofTT_UoSVD_HSI_Salines.mat','CVofTT_UoSVD_HSI_Salines') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTT_UoSVD_HSI_Salines == max(CVofTT_UoSVD_HSI_Salines)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTT_UoSVD_HSI_Salines, '--s',...
    'MarkerIndices',1:1:length(l),...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerFaceColor',[0.5,0.5,0.5],...
    'MarkerEdgeColor','b',...
    'MarkerIndices',[idxmax1],...
    'MarkerFaceColor','red',...
    'MarkerSize', 5)

%% ADNI dataset
    case 'adni'
% loading ADNi data file
load('ADNI_first.mat')
n = size(X,1);

% Computing TT decomposition
eps = 0; 
trunc = 2; % while fixing rank 
dimn = size(X{1});
% Repeat t times with k-fold cross validation 
t = 10; % number of repitition of whole procedure
global l % the rank has been defined as a global variable
for l = 1:10 
[data_TT,~] = TT_dec(X,l,eps); %TT factorization of input 


% merging index r1 and r2 into r = r1+(r2-1)*R1
R1 =  l;
R2 = l;

 
% computing TT matricized of input tensor
[ data_TT2 ] = TTmat( X, data_TT);

% main results including training and testing
[CVofTT_UoSVD_ADNI(l),trainTIMEofTT_UoSVD_ADNI,testTIMEofTT_UoSVD_ADNI] = KTTMain_lib(X,label,data_TT2,data_TT,t);

end

l = 1:1:10;
save('CVofTT_UoSVD_ADNI.mat','CVofTT_UoSVD_ADNI') % mat file for accuracy output
idxmax1 = find(CVofTT_UoSVD_ADNI == max(CVofTT_UoSVD_ADNI)); % maximum accuracy
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
% plotting accuracy vs TT rank
plot(l,CVofTT_UoSVD_ADNI, '--s',...
    'MarkerIndices',1:1:length(l),...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerFaceColor',[0.5,0.5,0.5],...
    'MarkerEdgeColor','b',...
    'MarkerIndices',[idxmax1],...
    'MarkerFaceColor','red',...
    'MarkerSize', 5)

%%
%%%%%%%%%%%%%%%%%%%%%% DATA2: ADHD %%%%%%%%%%%%%%%%%%%%%%%%%%

%% ADHD dataset
    case 'adhd'
load('ADHD_mainset.mat');
n = size(X,1);

% Computing TT decomposition
eps = 0; 
trunc = 2; % while fixing rank 
dimn = size(X{1});
% Repeat t times with k-fold cross validation 
t = 1; % number of repitition of whole procedure
global l % the rank has been defined as a global variable
for l = 1:10
[data_TT,~] = TT_dec(X,l,eps); %TT factorization of input 


% merging index r1 and r2 into r = r1+(r2-1)*R1
R1 =  l;
R2 = l;

% computing TT matricized of input tensor
[ data_TT2 ] = TTmat( X, data_TT);

% main results including training and testing
[CVofTT_UoSVD_ADHD(l),trainTIMEofTT_UoSVD_ADHD,testTIMEofTT_UoSVD_ADHD] = KTTMain_lib(X,label,l,data_TT2,data_TT,t);

end

l = 1:1:10;
save('CVofTT_UoSVD_ADHD.mat','CVofTT_UoSVD_ADHD') % saving accuracy mat file
idxmax1 = find(CVofTT_UoSVD_ADHD == max(CVofTT_UoSVD_ADHD)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTT_UoSVD_ADHD, '--s',...
    'MarkerIndices',1:1:length(l),...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerFaceColor',[0.5,0.5,0.5],...
    'MarkerEdgeColor','b',...
    'MarkerIndices',[idxmax1],...
    'MarkerFaceColor','red',...
    'MarkerSize', 5)
%%
end % end for switch function




