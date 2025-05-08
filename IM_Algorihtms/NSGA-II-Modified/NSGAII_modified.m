classdef NSGAII_modified < ALGORITHM
% <2002> <multi> <real/integer/label/binary/permutation> <constrained/none>
% Nondominated sorting genetic algorithm II
% retrain --- 0 --- Whether retrain the model
% interval --- 10 --- Training interval (1, 5, 10, 20, 50)
% percentage --- 0.25 --- Percentage of replaced offsprings
% epochs --- 100 --- Number of training epochs for the model
% alpha --- NaN --- Alpha value (NaN means a random value)
% normalization --- false --- Whether normalize the objective vectors and decsion vectros

%------------------------------- Reference --------------------------------
% K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, A fast and elitist
% multiobjective genetic algorithm: NSGA-II, IEEE Transactions on
% Evolutionary Computation, 2002, 6(2): 182-197.
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
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            assert(length(Algorithm.parameter)==6 || isempty(Algorithm.parameter));
            [retrain, interval, percentage, epochs,  alpha, normalization] = Algorithm.ParameterSet(0, 10, 0.25, 100, NaN, 0);
            % create model
            model = IM_SolutionGenerator(Problem.M, Problem.D, retrain, interval, percentage, epochs, alpha, ...
                Problem.upper, Problem.lower, normalization);
            gen = 1;
            model = model.UpdateUEA(Population,[],gen);
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool).decs);
                % create new solutions
                [Offspring, model, num, better_objs, dataset, prediction] = model.Generate(Offspring, gen);
                Offspring = Problem.Evaluation(Offspring);

                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);

                gen = gen + 1;
                model = model.UpdateUEA(Population, Offspring, gen);
            end
        end
    end
end