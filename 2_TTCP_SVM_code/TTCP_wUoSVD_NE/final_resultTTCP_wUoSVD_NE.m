%%%% This is the main file for producing all the results %%%%
% The codes are for the paper
% "Efficient Structure-preserving Support Tensor Train Machine" by K.Kour,
% S.Dolgov, M. Stoll and P.Benner

% Code author @Kirandeep Kour

% Step 1. Setup the TT toolbox
% Step 2. run make file in matlab folder in libsvm-master
% Step 3. Add complete folder and subfolders into path
addpath(genpath('../../Toolboxes/TT-Toolbox-master'))
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
[TT_CP_data] = ttcptensor(data_TT,R1,R2,dimn,trunc);

% main results including training and testing
[CVofTTCP_wUoSVD_NE_HSI_indiana,trainTIMEofTTCP_HSI_indiana,testTIMEofTTCP_HSI_indiana] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end
save('CVofTTCP_wUoSVD_NE_HSI_indiana.mat','CVofTTCP_wUoSVD_NE_HSI_indiana') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTTCP_wUoSVD_NE_HSI_indiana == max(CVofTTCP_wUoSVD_NE_HSI_indiana)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCP_wUoSVD_NE_HSI_indiana, '--s',...
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
[TT_CP_data] = ttcptensor(data_TT,R1,R2,dimn,trunc);

% main results including training and testing
[CVofTTCP_wUoSVD_NE_HSI_Salines,trainTIMEofTTCP_HSI_salines,testTIMEofTTCP_HSI_salines] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end
save('CVofTTCP_wUoSVD_NE_HSI_Salines.mat','CVofTTCP_wUoSVD_NE_HSI_Salines') % mat file for accuracy output

l = 1:1:10;
idxmax1 = find(CVofTTCP_wUoSVD_NE_HSI_Salines == max(CVofTTCP_wUoSVD_NE_HSI_Salines)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCP_wUoSVD_NE_HSI_Salines, '--s',...
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
[TT_CP_data] = ttcptensor(data_TT,R1,R2,dimn,trunc);


% main results including training and testing
[CVofTTCP_wUoSVD_NE_ADNI(l),trainTIMEofTTCP_ADNI,testTIMEofTTCP_ADNI] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end

l = 1:1:10;
save('CVofTTCP_wUoSVD_NE_ADNI.mat','CVofTTCP_wUoSVD_NE_ADNI') % mat file for accuracy output
idxmax1 = find(CVofTTCP_wUoSVD_NE_ADNI == max(CVofTTCP_wUoSVD_NE_ADNI)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCP_wUoSVD_NE_ADNI, '--s',...
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
[TT_CP_data] = ttcptensor(data_TT,R1,R2,dimn,trunc);


% main results including training and testing
[CVofTTCP_wUoSVD_NE_ADHD(l),trainTIMEofTTCP_ADHD,testTIMEofTTCP_ADHD] = KTTCPMain_lib(X,label,l,TT_CP_data,t);

end

l = 1:1:10;
save('CVofTTCP_wUoSVD_NE_ADHD.mat','CVofTTCP_wUoSVD_NE_ADHD') % saving accuracy mat file
idxmax1 = find(CVofTTCP_wUoSVD_NE_ADHD == max(CVofTTCP_wUoSVD_NE_ADHD)); % maximum accuracy
% plotting accuracy vs TT rank
plot(l,CVofTTCP_wUoSVD_NE_ADHD, '--s',...
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




