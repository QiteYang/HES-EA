function Id = DiversityIndi(PopObj)
[N,~] = size(PopObj);

 %% diversity degree by SDE
 sde = inf(N);
 for i = 1 : N
     SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
     for j = [1:i-1,i+1:N]
         sde(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
     end
 end
 [sde,~] = sort(sde,2);
 k = floor(sqrt(N));
 Id = 1./(sde(:,k) + 2.0);

end