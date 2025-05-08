classdef IMMOEAD_modified < ALGORITHM
% <2021> <multi> <real/integer> <large/none>
% Inverse modeling MOEA/D
% retrain --- 0 --- Whether retrain the model
% interval --- 10 --- Training interval (1, 5, 10, 20, 50)
% percentage --- 0.25 --- Percentage of replaced offsprings
% epochs --- 100 --- Number of training epochs for the model
% alpha --- NaN --- Alpha value (NaN means a random value)
% normalization --- false --- Whether normalize the objective vectors and decsion vectros

%------------------------------- Reference --------------------------------
% L. R. C. Farias and A. F. R. Araujo, IM-MOEA/D: An inverse modeling
% multi-objective evolutionary algorithm based on decomposition,
% Proceedings of the IEEE International Conference on Systems, Mans and
% Cybernetics, 2021.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias

    methods
		function main(Algorithm,Problem)
			%% Parameter setting
			%K = Algorithm.ParameterSet(10);
            K = 10;
            assert(length(Algorithm.parameter)==6 || isempty(Algorithm.parameter));
            [retrain, interval, percentage, epochs,  alpha, normalization] = Algorithm.ParameterSet(0, 10, 0.25, 100, NaN, 0);

            T = ceil(Problem.N/10); % Size of neighborhood
            
			%% Generate weight vectors
			[W,Problem.N] = UniformPoint(Problem.N,Problem.M);

            
			%% Detect the neighbours of each solution
			B = pdist2(W,W);
			[~,B] = sort(B,2);
			B = B(:,1:T);
			
			%% Generate random population
			Population    = Problem.Initialization();
			Z = min(Population.objs,[],1);
			
            % create model
            model = IM_SolutionGenerator(Problem.M, Problem.D, retrain, interval, percentage, epochs, alpha, ...
                Problem.upper, Problem.lower, normalization);
            gen = 1;
            model = model.UpdateUEA(Population,[],gen);

			%% Optimization
			while Algorithm.NotTerminated(Population)
				[partition,~] = kmeans(Population.objs,K); 
				Offsprings=[];
				% Modeling and reproduction
				for k = unique(partition)'
					tmp  = Operator(Problem,Population(partition==k));
					Offsprings = [Offsprings; tmp];
                end
                [Offsprings, model, ~, ~, ~, ~] = model.Generate(Offsprings, gen);
                Offsprings = Problem.Evaluation(Offsprings);
				
				% Update the ideal point
				Z = min(Z,min(Offsprings.objs));		
				
			    for i = 1 : length(Offsprings)
					% Global Replacement
					all_g_TCH=max(abs((Offsprings(i).obj-repmat(Z,Problem.N,1)).*W),[],2);
					best_g_TCH=min(all_g_TCH);
					Chosen_one = find(all_g_TCH(:,1)==best_g_TCH);
					P = B(Chosen_one(1),randperm(size(B,2)));
					
					% Update the solutions in P by Tchebycheff approach
					g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
					g_new = max(repmat(abs(Offsprings(i).obj-Z),length(P),1).*W(P,:),[],2);
					Population(P(find(g_old>=g_new,T))) = Offsprings(i);	
                end

                gen = gen + 1;
                model = model.UpdateUEA(Population, Offsprings, gen);
			end
		end
	end
end