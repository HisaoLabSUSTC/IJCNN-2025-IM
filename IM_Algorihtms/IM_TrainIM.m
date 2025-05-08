function [Result, net] = IM_TrainIM(Objs, Decs, TestObjs, epochs, net, ...
    upper,lower,isNormalized)

    if isNormalized
        max_obj = max([Objs; TestObjs], [], 1);
        min_obj = min([Objs; TestObjs], [], 1);
        Objs = normalization(Objs, max_obj, min_obj);
        TestObjs = normalization(TestObjs, max_obj, min_obj);
        Decs = normalization(Decs, upper, lower);
    end
    if epochs > 0
        options = trainingOptions("adam",...
            MaxEpochs=epochs,...
            MiniBatchSize=size(Objs,1),...
            Plots='none',...
            InitialLearnRate=1e-3,...
            LearnRateSchedule='none',...
            ExecutionEnvironment='cpu');
        net = trainnet(Objs,Decs,net,"mse",options);
    end
    Result = predict(net,TestObjs);
    if isNormalized
        upper = repmat(upper, size(Result,1), 1);
        lower = repmat(lower, size(Result,1), 1);
        Result = Result .* (upper-lower) + lower;
    end
end

function newV = normalization(V, upper, lower)
    upper = repmat(upper, size(V,1), 1);
    lower = repmat(lower, size(V,1), 1);
    newV = (V - lower) ./ (upper - lower);
end