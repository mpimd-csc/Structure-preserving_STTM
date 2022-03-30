% Comparison to "Kernelized Support Tensor Train Machine"
% The codes are available and taken from https://github.com/ZhaoYibo61/KSTTM
clear all;
clc;

addpath('KSTTM-master_comparison')


% some parameters in VBMF  d{i} = VBMF(Y, cacb, sigma2);
cacb = 100;
sigma2 = 0.1; % should be less than 1 in this case, the larger, the less 
% singualr values. Play with it to get a good performance.

% model parameters 
class_num=2; % class number
weight=1; % weight on the first and second modes of the tensor data,
% which determines the importance of those two modes compared with the
% third tensor mode, no need to change the value if no obvious clue.
flag='p';% 'a' for K-STTM-Sum, 'p' for K-STTM-Prod

dataset = 'ADNI';
switch dataset
    case 'ADNI'

% load ADNI data
load ADNI_first.mat

% stacking all tensors together 
% writing data in standard form for running experiment
tensor_data = cat(4, X{:},[]);
tensor_data = reshape(permute(tensor_data, [4,1,2,3]),33, 61,73,61 );
neg = find(label<0);
pos = find(label>0);
tensor_data_neg = tensor_data(neg, :,:,:);
tensor_data_pos = tensor_data(pos,:,:,:);


label(neg) = 2;
label_pos = label(pos);
label_neg = label(neg);



tensor_data = cat(1,tensor_data_pos, tensor_data_neg);
label = cat(1, label_pos, label_neg);

tensor_data=reshape(tensor_data,33,[]);
train_X=tensor_data([1:10,16:28],:);
test_X=tensor_data([11:15,29:33],:);
train_labels=label([1:10,16:28],:);
test_labels=[-ones(5,1);ones(5,1)];

I1 = size(tensor_data_neg,2);
I2 = size(tensor_data_neg,3);
I3 = size(tensor_data_neg,4);

A=tabulate(train_labels);
train_samples{1}=train_X(1:A(1,2),:);
for i=2:class_num
    train_samples{i}=train_X(1+sum(A(1:i-1,2)):sum(A(1:i,2)),:);
end

% image classification
for c1=1:class_num-1
    for c2=c1+1:class_num
      
        % training and valid data and label preparation
        samplenum=9;
        validnum=3;
        traindata=[train_samples{c1}(1:samplenum,:);train_samples{c2}(1:samplenum,:)];
        trainingL=[-ones(size(train_samples{c1}(1:samplenum,:),1),1);ones(size(train_samples{c2}(1:samplenum,:),1),1)];
        validdata=[train_samples{c1}(end-validnum+1:end,:);train_samples{c2}(end-validnum+1:end,:)]; 
        validL=[-ones(size(train_samples{c1}(end-validnum+1:end,:),1),1);ones(size(train_samples{c2}(end-validnum+1:end,:),1),1)];
        X=traindata;
        Y=trainingL;
        N=size(X,1);
        X=reshape(X,[N I1 I2 I3]);
        X=permute(X,[1 4 2 3]);
        
        d=3;
        X=permute(X,[2 3 4 1]);
        [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
        TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
        for t=2:d-1
            [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
            TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
        end
        TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
        U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
        U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
        traindata_temp = reshape(X,[I1*I3,N*I2]);
        TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),I2,N]);
        X=TT;
        
        validdata=reshape(validdata,[validnum*2 I1 I2 I3]);
        validdata=permute(validdata,[4 2 3 1]);
        validdata_temp = reshape(validdata,[I1*I3,validnum*2*I2]);
        TT_valid{3}=reshape(pinv(U_common)*validdata_temp,[size(TT{2},3),I2,validnum*2]);
        TT_valid{1}=TT{1};
        TT_valid{2}=TT{2};
        validdata=TT_valid;
       
        % training and validation start
        sigmarange =[-8:1:8];
        Crange = [-8:1:8];
        for sigma_i = 1:size(sigmarange,2)
            sigma=2^sigmarange(sigma_i);
            tic
            [ K] = kernel_mat( X, N,d,sigma,weight,flag);
            fprintf('kernel matrix costs time is   ')
            toc
            for C_i = 1:size(Crange,2)
                sigma=2^sigmarange(sigma_i);
                C=2^Crange(C_i);
                [ alpha, b] = svm_solver( K, Y, C, N);
                % for validation
                tic
                Ypred = predict(validdata, alpha, b, X, Y, sigma,d,weight,flag);
                fprintf('validation costs time is:   ')
                toc
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                truelabel=[-ones(validnum,1);ones(validnum,1)];
                diff=Ypred-truelabel;
                diff(diff~=0)=1;
                valid_error(sigma_i,C_i)=sum(diff)/(validnum*2);
                
                % compute the testing error
                XX=test_X;
                N_test=size(XX,1);
                XX=reshape(XX,[N_test I1 I2 I3]);
                XX=permute(XX,[4 2 3 1]);
                testdata_temp = reshape(XX,[I1*I3,N_test*I2]);
                TT_test{3}=reshape(pinv(U_common)*testdata_temp,[size(TT{2},3),I2,N_test]);
                TT_test{1}=TT{1};
                TT_test{2}=TT{2};
                XX=TT_test;
                scoremat=zeros(N_test,class_num);
                Ypred = predict(XX, alpha, b, X, Y, sigma,d,weight,flag);
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                diff=Ypred-test_labels;
                diff(diff~=0)=1;
                test_error(sigma_i,C_i)=sum(diff)/N_test
            end
        end
    end
end
Accuracy_ADNI = max(1-min(test_error))
    case 'ADHD'
        load ADHD_mainset.mat
        
% stacking all tensors together 
% writing data in standard form for running experiment
tensor_data = cat(4, X{:},[]);
tensor_data = reshape(permute(tensor_data, [4,1,2,3]),200, 49,58,47 );
neg = find(label<0);
pos = find(label>0);
tensor_data_neg = tensor_data(neg, :,:,:);
tensor_data_pos = tensor_data(pos,:,:,:);


label(neg) = 2;
label_pos = label(pos);
label_neg = label(neg);



tensor_data = cat(1,tensor_data_pos, tensor_data_neg);
label = cat(1, label_pos, label_neg);


tensor_data=double(reshape(tensor_data,200,[]));
train_X=tensor_data([1:75,101:175],:);
test_X=tensor_data([76:100,176:end],:);
train_labels=label([1:75,101:175],:);
test_labels=[-ones(25,1); ones(25,1)];
I1 = size(tensor_data_neg,2);
I2 = size(tensor_data_neg,3);
I3 = size(tensor_data_neg,4);

A=tabulate(train_labels);
train_samples{1}=train_X(1:A(1,2),:);
for i=2:class_num
    train_samples{i}=train_X(1+sum(A(1:i-1,2)):sum(A(1:i,2)),:);
end

% image classification
for c1=1:class_num-1
    for c2=c1+1:class_num
      
        % training and valid data and label preparation
        samplenum=53;
        validnum=22;
        traindata=[train_samples{c1}(1:samplenum,:);train_samples{c2}(1:samplenum,:)];
        trainingL=[-ones(size(train_samples{c1}(1:samplenum,:),1),1);ones(size(train_samples{c2}(1:samplenum,:),1),1)];
        validdata=[train_samples{c1}(end-validnum+1:end,:);train_samples{c2}(end-validnum+1:end,:)]; 
        validL=[-ones(size(train_samples{c1}(end-validnum+1:end,:),1),1);ones(size(train_samples{c2}(end-validnum+1:end,:),1),1)];
        X=traindata;
        Y=trainingL;
        N=size(X,1);
        X=reshape(X,[N I1 I2 I3]);
        X=permute(X,[1 4 2 3]);
        
        d=3;
        X=permute(X,[2 3 4 1]);
        [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
        TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
        for t=2:d-1
            [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
            TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
        end
        TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
        U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
        U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
        traindata_temp = reshape(X,[I1*I3,N*I2]);
        TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),I2,N]);
        X=TT;
        
        validdata=reshape(validdata,[validnum*2 I1 I2 I3]);
        validdata=permute(validdata,[4 2 3 1]);
        validdata_temp = reshape(validdata,[I1*I3,validnum*2*I2]);
        TT_valid{3}=reshape(pinv(U_common)*validdata_temp,[size(TT{2},3),I2,validnum*2]);
        TT_valid{1}=TT{1};
        TT_valid{2}=TT{2};
        validdata=TT_valid;
       
        % training and validation start
        sigmarange =[-8:1:8];
        Crange = [-8:1:8];
        for sigma_i = 1:size(sigmarange,2)
            sigma=2^sigmarange(sigma_i);
            tic
            [ K] = kernel_mat( X, N,d,sigma,weight,flag);
            fprintf('kernel matrix costs time is   ')
            toc
            for C_i = 1:size(Crange,2)
                sigma=2^sigmarange(sigma_i);
                C=2^Crange(C_i);
                [ alpha, b] = svm_solver( K, Y, C, N);
                % for validation
                tic
                Ypred = predict(validdata, alpha, b, X, Y, sigma,d,weight,flag);
                fprintf('validation costs time is:   ')
                toc
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                truelabel=[-ones(validnum,1);ones(validnum,1)];
                diff=Ypred-truelabel;
                diff(diff~=0)=1;
                valid_error(sigma_i,C_i)=sum(diff)/(validnum*2);
                
                % compute the testing error
                XX=test_X;
                N_test=size(XX,1);
                XX=reshape(XX,[N_test I1 I2 I3]);
                XX=permute(XX,[4 2 3 1]);
                testdata_temp = reshape(XX,[I1*I3,N_test*I2]);
                TT_test{3}=reshape(pinv(U_common)*testdata_temp,[size(TT{2},3),I2,N_test]);
                TT_test{1}=TT{1};
                TT_test{2}=TT{2};
                XX=TT_test;
                scoremat=zeros(N_test,class_num);
                Ypred = predict(XX, alpha, b, X, Y, sigma,d,weight,flag);
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                diff=Ypred-test_labels;
                diff(diff~=0)=1;
                test_error(sigma_i,C_i)=sum(diff)/N_test
            end
        end
    end
end
Accuracy_ADHD = max(1-min(test_error))
    case 'HSI_salines'
        load HSI_Salines.mat
% stacking all tensors together 
% writing data in standard form for running experiment
tensor_data = cat(4, X{:},[]);
tensor_data = reshape(permute(tensor_data, [4,1,2,3]),100, 5,5,224 );
neg = find(label==0);
pos = find(label>0);
tensor_data_neg = tensor_data(neg, :,:,:);
tensor_data_pos = tensor_data(pos,:,:,:);


label(neg) = 2;
label_pos = label(pos);
label_neg = label(neg);



tensor_data = cat(1,tensor_data_pos, tensor_data_neg);
label = cat(1, label_pos, label_neg);


tensor_data=double(reshape(tensor_data,100,[]));
train_X=tensor_data([1:35,51:85],:);
test_X=tensor_data([36:50,86:end],:);
train_labels=label([1:35,51:85],:);
test_labels=[-ones(15,1);ones(15,1)];
I1 = size(tensor_data_neg,2);
I2 = size(tensor_data_neg,3);
I3 = size(tensor_data_neg,4);

A=tabulate(train_labels);
train_samples{1}=train_X(1:A(1,2),:);
for i=2:class_num
    train_samples{i}=train_X(1+sum(A(1:i-1,2)):sum(A(1:i,2)),:);
end

% image classification
for c1=1:class_num-1
    for c2=c1+1:class_num
      
        % training and valid data and label preparation
        samplenum=25;
        validnum=10;
        traindata=[train_samples{c1}(1:samplenum,:);train_samples{c2}(1:samplenum,:)];
        trainingL=[-ones(size(train_samples{c1}(1:samplenum,:),1),1);ones(size(train_samples{c2}(1:samplenum,:),1),1)];
        validdata=[train_samples{c1}(end-validnum+1:end,:);train_samples{c2}(end-validnum+1:end,:)]; 
        validL=[-ones(size(train_samples{c1}(end-validnum+1:end,:),1),1);ones(size(train_samples{c2}(end-validnum+1:end,:),1),1)];
        X=traindata;
        Y=trainingL;
        N=size(X,1);
        X=reshape(X,[N I1 I2 I3]);
        X=permute(X,[1 4 2 3]);
        
        d=3;
        X=permute(X,[2 3 4 1]);
        [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
        TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
        for t=2:d-1
            [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
            TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
        end
        TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
        U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
        U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
        traindata_temp = reshape(X,[I1*I3,N*I2]);
        TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),I2,N]);
        X=TT;
        
        validdata=reshape(validdata,[validnum*2 I1 I2 I3]);
        validdata=permute(validdata,[4 2 3 1]);
        validdata_temp = reshape(validdata,[I1*I3,validnum*2*I2]);
        TT_valid{3}=reshape(pinv(U_common)*validdata_temp,[size(TT{2},3),I2,validnum*2]);
        TT_valid{1}=TT{1};
        TT_valid{2}=TT{2};
        validdata=TT_valid;
       
        % training and validation start
        sigmarange =[-8:1:8];
        Crange = [-8:1:8];
        for sigma_i = 1:size(sigmarange,2)
            sigma=2^sigmarange(sigma_i);
            tic
            [ K] = kernel_mat( X, N,d,sigma,weight,flag);
            fprintf('kernel matrix costs time is   ')
            toc
            for C_i = 1:size(Crange,2)
                sigma=2^sigmarange(sigma_i);
                C=2^Crange(C_i);
                [ alpha, b] = svm_solver( K, Y, C, N);
                % for validation
                tic
                Ypred = predict(validdata, alpha, b, X, Y, sigma,d,weight,flag);
                fprintf('validation costs time is:   ')
                toc
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                truelabel=[-ones(validnum,1);ones(validnum,1)];
                diff=Ypred-truelabel;
                diff(diff~=0)=1;
                valid_error(sigma_i,C_i)=sum(diff)/(validnum*2);
                
                % compute the testing error
                XX=test_X;
                N_test=size(XX,1);
                XX=reshape(XX,[N_test I1 I2 I3]);
                XX=permute(XX,[4 2 3 1]);
                testdata_temp = reshape(XX,[I1*I3,N_test*I2]);
                TT_test{3}=reshape(pinv(U_common)*testdata_temp,[size(TT{2},3),I2,N_test]);
                TT_test{1}=TT{1};
                TT_test{2}=TT{2};
                XX=TT_test;
                scoremat=zeros(N_test,class_num);
                Ypred = predict(XX, alpha, b, X, Y, sigma,d,weight,flag);
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                diff=Ypred-test_labels;
                diff(diff~=0)=1;
                test_error(sigma_i,C_i)=sum(diff)/N_test
            end
        end
    end
end
Accuracy_salines = max(1-min(test_error))

    case 'HSI_Indiana'
        load HSI_Indiana.mat
% stacking all tensors together 
% writing data in standard form for running experiment
tensor_data = cat(4, X{:},[]);
tensor_data = reshape(permute(tensor_data, [4,1,2,3]),100, 5,5,220 );
neg = find(label==0);
pos = find(label>0);
tensor_data_neg = tensor_data(neg, :,:,:);
tensor_data_pos = tensor_data(pos,:,:,:);


label(neg) = 2;
label_pos = label(pos);
label_neg = label(neg);



tensor_data = cat(1,tensor_data_pos, tensor_data_neg);
label = cat(1, label_pos, label_neg);


tensor_data=double(reshape(tensor_data,100,[]));
train_X=tensor_data([1:35,51:85],:);
test_X=tensor_data([36:50,86:end],:);
train_labels=label([1:35,51:85],:);
test_labels=[-ones(15,1);ones(15,1)];
I1 = size(tensor_data_neg,2);
I2 = size(tensor_data_neg,3);
I3 = size(tensor_data_neg,4);

A=tabulate(train_labels);
train_samples{1}=train_X(1:A(1,2),:);
for i=2:class_num
    train_samples{i}=train_X(1+sum(A(1:i-1,2)):sum(A(1:i,2)),:);
end

% image classification
for c1=1:class_num-1
    for c2=c1+1:class_num
      
        % training and valid data and label preparation
        samplenum=25;
        validnum=10;
        traindata=[train_samples{c1}(1:samplenum,:);train_samples{c2}(1:samplenum,:)];
        trainingL=[-ones(size(train_samples{c1}(1:samplenum,:),1),1);ones(size(train_samples{c2}(1:samplenum,:),1),1)];
        validdata=[train_samples{c1}(end-validnum+1:end,:);train_samples{c2}(end-validnum+1:end,:)]; 
        validL=[-ones(size(train_samples{c1}(end-validnum+1:end,:),1),1);ones(size(train_samples{c2}(end-validnum+1:end,:),1),1)];
        X=traindata;
        Y=trainingL;
        N=size(X,1);
        X=reshape(X,[N I1 I2 I3]);
        X=permute(X,[1 4 2 3]);
        
        d=3;
        X=permute(X,[2 3 4 1]);
        [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
        TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
        for t=2:d-1
            [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
            TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
        end
        TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
        U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
        U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
        traindata_temp = reshape(X,[I1*I3,N*I2]);
        TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),I2,N]);
        X=TT;
        
        validdata=reshape(validdata,[validnum*2 I1 I2 I3]);
        validdata=permute(validdata,[4 2 3 1]);
        validdata_temp = reshape(validdata,[I1*I3,validnum*2*I2]);
        TT_valid{3}=reshape(pinv(U_common)*validdata_temp,[size(TT{2},3),I2,validnum*2]);
        TT_valid{1}=TT{1};
        TT_valid{2}=TT{2};
        validdata=TT_valid;
       
        % training and validation start
        sigmarange =[-8:1:8];
        Crange = [-8:1:8];
        for sigma_i = 1:size(sigmarange,2)
            sigma=2^sigmarange(sigma_i);
            tic
            [ K] = kernel_mat( X, N,d,sigma,weight,flag);
            fprintf('kernel matrix costs time is   ')
            toc
            for C_i = 1:size(Crange,2)
                sigma=2^sigmarange(sigma_i);
                C=2^Crange(C_i);
                [ alpha, b] = svm_solver( K, Y, C, N);
                % for validation
                tic
                Ypred = predict(validdata, alpha, b, X, Y, sigma,d,weight,flag);
                fprintf('validation costs time is:   ')
                toc
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                truelabel=[-ones(validnum,1);ones(validnum,1)];
                diff=Ypred-truelabel;
                diff(diff~=0)=1;
                valid_error(sigma_i,C_i)=sum(diff)/(validnum*2);
                
                % compute the testing error
                XX=test_X;
                N_test=size(XX,1);
                XX=reshape(XX,[N_test I1 I2 I3]);
                XX=permute(XX,[4 2 3 1]);
                testdata_temp = reshape(XX,[I1*I3,N_test*I2]);
                TT_test{3}=reshape(pinv(U_common)*testdata_temp,[size(TT{2},3),I2,N_test]);
                TT_test{1}=TT{1};
                TT_test{2}=TT{2};
                XX=TT_test;
                scoremat=zeros(N_test,class_num);
                Ypred = predict(XX, alpha, b, X, Y, sigma,d,weight,flag);
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                diff=Ypred-test_labels;
                diff(diff~=0)=1;
                test_error(sigma_i,C_i)=sum(diff)/N_test
            end
        end
    end
end
Accuracy_Indiana = max(1-min(test_error))
end% end for switch 
