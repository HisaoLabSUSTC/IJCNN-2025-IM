classdef IMMOEA_modified < ALGORITHM
% <2015> <multi> <real/integer> <large/none>
% Inverse modeling based multiobjective evolutionary algorithm
% retrain --- 0 --- Whether retrain the model
% interval --- 10 --- Training interval (1, 5, 10, 20, 50)
% percentage --- 0.25 --- Percentage of replaced offsprings
% epochs --- 100 --- Number of training epochs for the model
% alpha --- NaN --- Alpha value (NaN means a random value)
% normalization --- false --- Whether normalize the objective vectors and decsion vectros

%------------------------------- Reference --------------------------------
% R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff, A multiobjective
% evolutionary algorithm using Gaussian process-based inverse modeling,
% IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            %K = Algorithm.ParameterSet(10);
            K = 10;
            assert(length(Algorithm.parameter)==6 || isempty(Algorithm.parameter));
            [retrain, interval, percentage, epochs,  alpha, normalization] = Algorithm.ParameterSet(0, 10, 0.25, 100, NaN, 0);

            %% Generate random population
            [W,K] = UniformPoint(K,Problem.M);
            W     = fliplr(sortrows(fliplr(W)));
            Problem.N     = ceil(Problem.N/K)*K;
            Population    = Problem.Initialization();
            [~,partition] = max(1-pdist2(Population.objs,W,'cosine'),[],2);
            
            % create model
            model = IM_SolutionGenerator(Problem.M, Problem.D, retrain, interval, percentage, epochs, alpha, ...
                Problem.upper, Problem.lower, normalization);
            gen = 1;
            model = model.UpdateUEA(Population,[],gen);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Modeling and reproduction
                Offsprings = [];
                for k = unique(partition)'
                    Offsprings = [Offsprings; Operator(Problem,Population(partition==k))];
                end
                
                % create new solutions
                 [Offsprings, model, num, better_objs, dataset, prediction] = model.Generate(Offsprings, gen);
                Offsprings = Problem.Evaluation(Offsprings);

                Population = [Population, Offsprings];
                % Environmental selection
                [~,partition] = max(1-pdist2(Population.objs,W,'cosine'),[],2);
                for k = unique(partition)'
                    current = find(partition==k);
                    if length(current) > Problem.N/K
                        Del = EnvironmentalSelection(Population(current),Problem.N/K);
                        Population(current(Del)) = [];
                        partition(current(Del))  = [];
                    end
                end

                gen = gen + 1;
                model = model.UpdateUEA(Population, Offsprings, gen);
            end
        end
    end
end