function [NewPop,NewIc,NewId,NewClus] = CSS(PopDec,Ic,Id,Cluster,KMeans,N)
NewPop = [];
NewIc = [];
NewId = [];
NewClus = [];
counter = 0;

while counter < N
    ave_c = zeros(1,KMeans);
    for i = 1 : KMeans
        Loc = find(Cluster==i);
        if ~isempty(Loc)
            ave_c(i) = 1/(sum(Ic(Loc))/length(Loc) + 1.0);
        end
    end
    Pr = ave_c./(sum(ave_c));
    for i = 1 : KMeans
        Loc = find(Cluster==i);
        if rand(1) < Pr(i) && ~isempty(Loc)
            % use Ic and Id as two objectives to make non-dominated sorting
            % and select non-dominated layer
            F = [Ic(Loc),Id(Loc)];
            [front,~] = NDSort(F,inf);
            optima = find(front==1);
            % random select a solution
            choose = Loc(optima(randperm(numel(optima),1)));
            % add the chosen solution into P
            NewPop = [NewPop;PopDec(choose,:)];
            NewIc = [NewIc;Ic(choose)];
            NewId = [NewId;Id(choose)];
            NewClus = [NewClus;Cluster(choose)];
            % remove it from the subpopulation sp_i
            PopDec(choose,:) = [];
            Ic(choose) = [];
            Id(choose) = [];
            Cluster(choose) = [];
            counter = counter + 1;
        end
        if counter == N
            break;
        end
    end
end
end