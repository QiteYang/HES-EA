classdef HES_EA < ALGORITHM
% <multi> <real> <expensive>
% Qi-Te Yang, Jian-Yu Li, Zhi-Hui Zhan, Yunliang Jiang, Yaochu Jin, and Jun Zhang, "A Hierarchical
% and Ensemble Surrogate-Assisted Evolutionary Algorithm with Model Reduction for Expensive
% Many-objective Optimization," IEEE Transactions on Evolutionary Computation, 2024, DOI: 10.1109/TEVC.2024.3440354.
% wmax --- 20 --- The maximum number of internal evluation
% WN --- 190 --- The number of reference vectors
% KMeans --- 4 --- The number of cluster number

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qi-Te Yang
% email: qiteyang@foxmail.com

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [wmax,WN,KMeans] = Algorithm.ParameterSet(20,190,4);   
            Model_c = cell(1,KMeans+1);
            Model_d = cell(1,KMeans+1);
            THETA_c = 5.*ones(KMeans+1,Problem.D);  
            THETA_d = 5.*ones(KMeans+1,Problem.D);  

            %% Generate initial population based on Latin hypercube sampling
            InitN          = 11*Problem.D-1;
            P          = UniformPoint(InitN,Problem.D,'Latin');
            Population = SOLUTION(repmat(Problem.upper-Problem.lower,InitN,1).*P+repmat(Problem.lower,InitN,1));
         
            %% Generate the weight vectors for convergence indicator
            [W,~] = UniformPoint(WN,Problem.M);
             %% initialize vectors for clustering
            [ClW,~] = UniformPoint(KMeans,2);

            %% Optimization
            while Algorithm.NotTerminated(Population)    
                %% database   
                [~,index]  = unique(Population.decs,'rows');
                Population = Population(index);
                PopDec = Population.decs;   
                
                % update Zmin 
                Zmin = min(Population.objs,[],1);   
                
                [N,~] = size(PopDec);
                PopObj = Population.objs;
                NormObj = (Population.objs - Zmin);

                %% clustering
                Cluster = inf(N,1);
                % randomly select two objectives
                ObjK = randperm(Problem.M,2);
                ClObj = NormObj(:,ObjK);
                temp = N;
                while temp > 0
                    for i = 1 : KMeans
                        [~,loc] = min(acos(1-pdist2(ClObj,ClW(i,:),'cosine')));
                        Cluster(loc) = i;
                        ClObj(loc,:) = inf;
                        temp = temp - 1;
                        if temp == 0
                            break;
                        end
                    end
                end
                
                %% calculate Ic and Id
                Dis = pdist2(NormObj,W);
                [~,Loc] = min(Dis,[],2);
                Ic = sum(NormObj.*W(Loc,:),2);
                Id = DiversityIndi(PopObj);
                

                %% Train cluster model  
                model_rg = fitcknn(PopDec,Cluster,'NumNeighbors',5);
                %% use KMeans model to train different local model
                for i = 1:KMeans
                    X_train_c = PopDec(Cluster==i,:); Y_train_c = Ic(Cluster==i);
                    X_train_d = PopDec(Cluster==i,:); Y_train_d = Id(Cluster==i);
                    [X_train_c, Y_train_c]   = dsmerge(X_train_c, Y_train_c);
                    [X_train_d, Y_train_d]   = dsmerge(X_train_d, Y_train_d);
                    model_c = dacefit(X_train_c,Y_train_c,'regpoly0','corrgauss',THETA_c(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));                   
                    model_d = dacefit(X_train_d,Y_train_d,'regpoly0','corrgauss',THETA_d(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    THETA_c(i,:) = model_c.theta;
                    THETA_d(i,:) = model_d.theta;   
                    Model_c{i} = model_c;
                    Model_d{i} = model_d;
                end
                %% global model
                [X_train_c,Y_train_c] = dsmerge(PopDec,Ic);
                [X_train_d,Y_train_d] = dsmerge(PopDec,Id);
                model_c = dacefit(X_train_c,Y_train_c,'regpoly0','corrgauss',THETA_c(KMeans+1,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                model_d = dacefit(X_train_d,Y_train_d,'regpoly0','corrgauss',THETA_d(KMeans+1,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                THETA_c(KMeans+1,:) = model_c.theta;
                THETA_d(KMeans+1,:) = model_d.theta;
                Model_c{KMeans+1} = model_c;
                Model_d{KMeans+1} = model_d;
                
                %% Optimization by CSS
                [ArcDec,ArcIc,ArcId,ArcClus] = CSS(PopDec,Ic,Id,Cluster,KMeans,Problem.N); 
                ArcMSE = zeros(Problem.N,2);
                
                w = 0;
                while w < wmax
                    drawnow();
                    OffDec = OperatorGA(ArcDec);
                    w = w + 1;
                    [n,~] = size(OffDec);
                    OffClus = predict(model_rg,OffDec);  % prediction of cluster 
                    
                    OffIc = zeros(n,1);
                    OffId = OffIc;
                    OffMSE = zeros(n,2);

                    for i = 1 : n
                        loc = OffClus(i);
                        %% prediction of Id and Ic
                        [Yc_local,~,mse_local] = predictor(OffDec(i,:),Model_c{loc});
                        [Yc_global,~,mse_global] = predictor(OffDec(i,:),Model_c{KMeans+1});
                        
                        if mse_local < mse_global
                            OffIc(i) = Yc_local;
                            OffMSE(i,1) = mse_local;
                        else
                            OffIc(i) = Yc_global;
                            OffMSE(i,1) = mse_global;
                        end
                        
                        [Yd_local,~,mse_local] = predictor(OffDec(i,:),Model_d{loc});
                        [Yd_global,~,mse_global] = predictor(OffDec(i,:),Model_d{KMeans+1});
                        if mse_local < mse_global
                            OffId(i) = Yd_local;
                            OffMSE(i,2) = mse_local;
                        else
                            OffId(i) = Yd_global;
                            OffMSE(i,2) = mse_global;
                        end
                    end
                    ArcDec = [ArcDec;OffDec];
                    ArcIc = [ArcIc;OffIc];
                    ArcId = [ArcId;OffId];
                    ArcClus = [ArcClus;OffClus];
                    ArcMSE = [ArcMSE;OffMSE];
                    %% clustering based sequential selection
                    ArcIc = ArcIc-min(ArcIc);
                    [ArcDec,ArcIc,ArcId,ArcClus,ArcMSE] = ModiCSS(ArcDec,ArcIc,ArcId,ArcClus,ArcMSE,KMeans,Problem.N);
                end

              %% infill criterion
              OldDec = Population.decs;
              Loc = [];
              for i = 1 : size(ArcDec,1)
                  for j = 1 : N
                      if isequal(ArcDec(i,:),OldDec(j,:))
                          Loc = [Loc,i];
                          break;
                      end
                  end
              end
              ArcDec(Loc,:) = [];
              ArcIc(Loc) = [];
              ArcId(Loc) = [];
              ArcClus(Loc) = [];
              ArcMSE(Loc,:) = [];
              
              [n,~] = size(ArcDec);
              if n <= 5
                  NewArc = ArcDec;
              else
                  NewArc = [];                  
                  clus = kmeans(ArcDec,5);
                  for i = 1 : 5
                      Ci = find(clus==i);
                      [fr,~] = NDSort(-ArcMSE(Ci,:),inf);
                      [~,uncertain] = find(fr==1);
                      % choose uncertain solution
                      choose = Ci(uncertain(randperm(numel(uncertain),1)));
                      NewArc = [NewArc;ArcDec(choose,:)];
                  end
              end
          
              if ~isempty(NewArc)
                  PopNew = SOLUTION(NewArc);
                  Population = [Population,PopNew];        
              end
              
            end
        end
    end
end