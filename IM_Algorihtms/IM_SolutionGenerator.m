
classdef IM_SolutionGenerator
% IM_SolutionGenerator: Generate good offspring solutions
%
%Properties
% T       - every T generations, the model is trained.
% retrain - whether train the model from scratch
% epochs  - number of training epochs
% disS    - parameter for sampling the alpha
% alpha   - value of the alpha parameter (NaN means a random value)
% UEA_pop - unbounded external archives to store the population
% UEA_off - unbounded external archives to store the offspring
% upper   - upper bound of the decision variables (only used when the 
%           normalization is set as true)
% lower   - lower bound of the decision variables (only used when the 
%           normalization is set as true)
% normalization - whether normalize the objective vectors and decsion
%                 vectors
% Author: Tianye Shu
    properties
        T = 10;
        retrain = 0;
        Q = 0.25;
        epochs = 100;
        disS = 2;
        net = [];
        alpha = NaN;
        UEA_pop = {};
        UEA_off = {};
        upper = [];
        lower = [];
        normalization = false;
    end
    methods
        function obj = IM_SolutionGenerator(M, D, retrain, T, Q, epochs, alpha, ...
                upper, lower, normalization)
            if nargin ==10
                obj.retrain = retrain;
                obj.T = T;
                obj.Q = Q;
                obj.epochs = epochs;
                obj.net = CreateNN(M, D);
                obj.disS = 2;
                obj.alpha = alpha;
                obj.upper = upper;
                obj.lower = lower;
                obj.normalization = normalization;
            elseif nargin >= 1
                error('Please provide complete paramters or no paramters.');
            end
        end
        
        function [obj] = UpdateUEA(obj, population, offspring, gen)
            obj.UEA_pop{gen} = population;
            obj.UEA_off{gen} = offspring;
        end

        function [Offsprings, obj, num, better_objs, dataset, prediction] = Generate(obj, Offsprings, gen)
            num = 0;
            better_objs = [];
            prediction = [];
            dataset = [];
            if mod(gen, obj.T)==0 && gen > 1
                K = floor(obj.Q * size(Offsprings,1));
                PreviousObjs = obj.UEA_pop{gen-obj.T+1}.objs;
                CurrentObjs = obj.UEA_pop{gen}.objs;
                if K > 0
                    TestObjs = IM_CalBetterObjs(PreviousObjs, CurrentObjs, K,...
                        obj.disS, obj.alpha, obj.normalization);
                    K = min(K, size(TestObjs, 1));
                    
                    % gen-th generation and (gen-T+1)-th generation are
                    % used for training
                    solutions = [obj.UEA_pop{gen-obj.T+1}, obj.UEA_pop{gen}];
                    if obj.retrain>=1
                        obj.net = IM_CreateNN(solutions.objs, solutions.decs);
                    end
                    [PredictDecs, obj.net] = IM_TrainIM(solutions.objs, ...
                        solutions.decs, TestObjs, obj.epochs, obj.net,...
                        obj.upper, obj.lower, obj.normalization);
                    PredictDecs = double(PredictDecs);
                    prediction = predict(obj.net,solutions.objs);
                else
                    PredictDecs = [];
                    TestObjs = [];
                    solutions = [];
                end
                index = randperm(size(Offsprings,1), size(Offsprings,1) - K);
                Offsprings = [Offsprings(index,:); PredictDecs];
                num = K;
                better_objs = TestObjs;
                dataset = solutions;
            end
        end

    end
                

end