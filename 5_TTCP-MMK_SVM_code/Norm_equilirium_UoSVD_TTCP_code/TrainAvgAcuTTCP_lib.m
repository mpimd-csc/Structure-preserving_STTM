function [Bestcv, Bestc,Bestg,time_tr,time_te]= TrainAvgAcuTTCP_lib(X,label,l,k,data_KTTCP,choose,c1,c2,g1,g2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Input
%          X          :  the input data cell array, n * 1 --- each array represents a three-order tensor
%                        n is the number of training examples
%          label      :  the output labels associated with the input data, n * 1
%          R          :  the rank of tensor decomposition
%          k          :  k-fold cross validation
%          data_KTTCP :  TTCP or KTTCP decomposition result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          c1, c2     :  the trade-off parameter range [2^c1, 2^(c1+1),..., 2^(c2-1), 2^c2] in SVM model
%          g1, g2     :  the RBF kernel width parameter range [2^g1, 2^(g1+1),..., 2^(g2-1), 2^g2] in SVM model

%%   Output:

%         Bestcv,Bestc,Bestg   :  Test accuracy obtained using k-fold cross validation in the optimal hyper-parameters (Bestc,Bestg)
%         time_tr              :  Training time     
%         time_te              :  Test time 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize
n=size(data_KTTCP,1);
global order
Order = 3;
c=ones(1,n);  
a=cumsum(c);
Bestcv=0;
Bestc=-100;
Bestg=-100;
Acctemp=zeros(k,c2-c1+1);
timetemp_tr=zeros(k,c2-c1+1);
timetemp_te=zeros(k,c2-c1+1);
time_tr=0;
time_te=0;

global l

%% k-fold cross validation
for log2g = g1:g2
    for cv=1:k
        b=setdiff(a,choose{cv,1});
        Ktrain=zeros(length(b),length(b));
        Ktest=zeros(n-length(b),length(b));
        tic;
        for p=1:length(b)
            for q=1:p
                if Order~=1   
                   Ktrain(p,q)=Ker_fTTCP(data_KTTCP{b(p),1},data_KTTCP{b(q),1},Order,2^log2g,l);
                else
                   Ktrain(p,q)=Ker_fTTCP(X(b(p),:),X(b(q),:),Order,2^log2g);
                end
                if p~=q
                    Ktrain(q,p)=Ktrain(p,q);
                end
            end
        end
        
        time_tr1=toc;
        tic;
        for r=1:n-length(b)
            for p=1:length(b)
                if Order~=1
                   Ktest(r,p)= Ker_fTTCP(data_KTTCP{choose{cv,1}(1,r),1},data_KTTCP{b(p),1},Order,2^log2g,l);
                else
                   Ktest(r,p)= ker_fTTCP(X(choose{cv,1}(1,r),:),X(b(p),:),Order,2^log2g);
                end
            end
        end
        time_te1=toc;
        Ktrain1 = [(1:length(b))',Ktrain];
        Ktest1 = [(1:n-length(b))', Ktest];
        tempc=0;
        
         for log2c=c1:c2  
            tempc=tempc+1;
            cmd=['-c ', num2str(2^log2c), ' -t ', num2str(4),' -q'];
            tic;
            model{cv}= svmtrain(label(b), Ktrain1, cmd); % training the model
            time_tr2=toc;
            timetemp_tr(cv,tempc)=time_tr1+time_tr2;
            tic;
            [~,temp,~] = svmpredict(label(choose{cv,1}), Ktest1, model{cv},'-q'); % predicting the model
            time_te2=toc;
            timetemp_te(cv,tempc)=time_te1+time_te2;
            Acctemp(cv,tempc)=temp(1)/100;  
        end % end for c
        clear b Ktrain Ktest
    end %end for cv 
    Accvector=sum(Acctemp,1);
    timetrain=sum(timetemp_tr,1);
    timetest=sum(timetemp_te,1);
    [Acc,C]=max(Accvector);
    if Acc/k>Bestcv
        Bestcv=Acc/k;       
        Bestc=(C-(1-c1));
        Bestg= log2g;
        time_tr=timetrain(C)/k;
        time_te=timetest(C)/k;
    end
    fprintf('%g %g (best c=%g, g=%g, bestacc=%g) traintime=%g,testtime=%g\n',log2c, log2g, Bestc, Bestg, Bestcv,time_tr,time_te); %results
end % end for g
end
