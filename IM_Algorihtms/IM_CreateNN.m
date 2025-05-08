function net = IM_CreateNN(M, D)
    layers = [
    featureInputLayer(M, "Normalization","none")
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(D)
    ];
    net = dlnetwork(layers);
end