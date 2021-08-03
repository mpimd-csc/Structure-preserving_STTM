function choose=Divide(label,k,randomSeed)

%% Divide data into k group based on k-fold cross validation

rand('state',randomSeed);
choose=cell(k,1);
labelmid=label;
c=ones(1,length(labelmid));                                                    % Count data number
sy=cumsum(c);                                                                  % Assign sequence number

%% k-fold cross validation
for s=k:-1:2    
    c=ones(1,length(labelmid));                                                    % Count data number 
    c=cumsum(c);                                                                   % Assign sequence number
    a=unique(labelmid);                                                          
    Lc=length(a);                                                                  % Count class number
    all=0;
    for i=1:Lc
        Ad=find(labelmid==a(i));
        Ai=Ad;
        for j=1:length(Ad)*1/s     
            t=ceil(rand(1)*length(Ai));
  %         fprintf('Rand value is %g\n',rand('seed'));
            if t>length(Ai)
                t=length(Ai);
            end
            all=all+1;
            boostplace(all)=Ai(t);
            Ai=setdiff(Ai,Ai(t));
        end
    end
    choose{s,1}=sy(1,boostplace);
    a=setdiff(c,boostplace);
    labelmid=labelmid(a,1);
    sy=sy(setdiff(c,boostplace));
end
choose{1,1}=sy;
end