%%%% This is the main file for producing all the results %%%%
%%%% This is the main file for producing all the results %%%%
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


% merging index r1 and r2 into r = r1+(r2-1)*R1
R1 =  l;
R2 = l;
[TT_CP_data] = ttcptensor_withoutnorm(data_TT,R1,R2,dimn,trunc);

% main results including training and testing
[CVofTTCPwNE_HSI_indiana,trainTIMEofTTCP_HSI_indiana,testTIMEofTTCP_HSI_indiana] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end
save('CVofTTCPwNE_HSI_indiana.mat','CVofTTCPwNE_HSI_indiana') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTTCPwNE_HSI_indiana == max(CVofTTCPwNE_HSI_indiana)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCPwNE_HSI_indiana, '--s',...
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


% merging index r1 and r2 into r = r1+(r2-1)*R1
R1 =  l;
R2 = l;
[TT_CP_data] = ttcptensor_withoutnorm(data_TT,R1,R2,dimn,trunc);

% main results including training and testing
[CVofTTCPwNE_HSI_Salines,trainTIMEofTTCP_HSI_salines,testTIMEofTTCP_HSI_salines] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end
save('CVofTTCPwNE_HSI_Salines.mat','CVofTTCPwNE_HSI_Salines') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTTCPwNE_HSI_Salines == max(CVofTTCPwNE_HSI_Salines)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCPwNE_HSI_Salines, '--s',...
    'MarkerIndices',1:1:length(l),...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerFaceColor',[0.5,0.5,0.5],...
    'MarkerEdgeColor','b',...
    'MarkerIndices',[idxmax1],...
    'MarkerFaceColor','red',...
    'MarkerSize', 5)

%%%%%%%%%%%%%%%%%%%%%% DATA1: ADNI %%%%%%%%%%%%%%%%%%%%%%%%%%

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
[TT_CP_data] = ttcptensor_withoutnorm(data_TT,R1,R2,dimn,trunc);


% main results including training and testing
[CVofTTCPwNE_UoSVD_ADNI(l),trainTIMEofTTCPwNE_UoSVD_ADNI,testTIMEofTTCPwNE_UoSVD_ADNI] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end

l = 1:1:10;
save('CVofTTCPwNE_UoSVD_ADNI.mat','CVofTTCPwNE_UoSVD_ADNI') % mat file for accuracy output
idxmax1 = find(CVofTTCPwNE_UoSVD_ADNI == max(CVofTTCPwNE_UoSVD_ADNI)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCPwNE_UoSVD_ADNI, '--s',...
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
[TT_CP_data] = ttcptensor_withoutnorm(data_TT,R1,R2,dimn,trunc);


% main results including training and testing
[CVofTTCPwNE_UoSVD_ADHD(l),trainTIMEofTTCPwNE_UoSVD_ADHD,testTIMEofTTCPwNE_UoSVD_ADHD] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end

l = 1:1:10;
save('CVofTTCPwNE_UoSVD_ADHD.mat','CVofTTCPwNE_UoSVD_ADHD') % saving accuracy mat file
idxmax1 = find(CVofTTCPwNE_UoSVD_ADHD == max(CVofTTCPwNE_UoSVD_ADHD)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCPwNE_UoSVD_ADHD, '--s',...
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




