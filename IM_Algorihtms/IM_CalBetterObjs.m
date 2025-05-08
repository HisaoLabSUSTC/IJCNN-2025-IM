function Result = IM_CalBetterObjs(PreviousPop, CurrentPop, K, disS,...
    alpha, normalization)
    if K == 0
        Result=[];
        return;
    end

    [FrontNo, MaxFNo] = NDSort(CurrentPop, 1);
    
    if normalization
        maxVal = max(CurrentPop(FrontNo==1,:),[],1);
        minVal = min(CurrentPop(FrontNo==1,:),[],1);
        CurrentPop = NormalizeObjs(CurrentPop, [maxVal; minVal]);
        PreviousPop = NormalizeObjs(PreviousPop, [maxVal; minVal]);
    end
    
    %% Nadir point is estimated by the current population
    NadirPoint = max(CurrentPop(FrontNo==1,:),[],1);
    %% Distance-based subset selection for the last front
    SelectNum = min(K, sum(FrontNo==1));
    [~, I] = DSS(NormalizeObjs(CurrentPop(FrontNo==1,:), CurrentPop(FrontNo==1,:)), SelectNum); 
    index = 1:size(CurrentPop, 1);
    index = index(FrontNo==1);
    index = index(I);
    [FrontNo, ~] = NDSort(PreviousPop, 1);
    PreviousNDSols = PreviousPop(FrontNo==1,:);
    Result = [];
    Debug = 0;
    if Debug > 0
        scatter(NadirPoint(:,1), NadirPoint(:,2),50,'filled', 'black'); hold on;
    end
    for i = index
        %% vector for the current solution
        V = CurrentPop(i,:) - NadirPoint;
        NormV = V / norm(V);
        %% vectors for the previous solutions
        num = size(PreviousNDSols,1);
        V2 = PreviousNDSols - repmat(NadirPoint, num, 1);
        %% projectin to the V
        V3 = sum(V2 .* repmat(NormV, num, 1), 2) * NormV;
        Distance = sum((V3 - V2).^2, 2);
        [~, Index] = min(Distance);

        ImpV = V3(Index,:)-V;

        mu = rand();
        if mu<=0.5
            beta = (2*mu).^(1/(disS+1));
        else
            beta  = (2-2*mu).^(-1/(disS+1));
        end
            
        
        if isnan(alpha)
            alpha = beta;
        end
        NewVector = V - alpha * ImpV + NadirPoint;
        if Debug > 0
            h1 = scatter(CurrentPop(i,1), CurrentPop(i,2), 50,...
                'MarkerFaceColor','red', 'MarkerEdgeColor','red'); hold on;
            h2 = scatter(PreviousNDSols(Index,1),PreviousNDSols(Index,2), ...
                50, 'MarkerFaceColor','blue', 'MarkerEdgeColor','blue');hold on;
            h3 = scatter(NewVector(:,1),NewVector(:,2), 50,...
                'MarkerFaceColor','green', 'MarkerEdgeColor','green'); hold on;
            set(h1, 'MarkerFaceColor', 'none');
            set(h2, 'MarkerFaceColor', 'none');
            set(h3, 'MarkerFaceColor', 'none');
        end
        Result = [Result; NewVector];
    end
    if Debug > 0
        close all;
    end

    if normalization
        maxVal = repmat(maxVal, size(Result,1), 1);
        minVal = repmat(minVal, size(Result,1), 1);
        Result = Result .* (maxVal-minVal) + minVal;  
    end
end