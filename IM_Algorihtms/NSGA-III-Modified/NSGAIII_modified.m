classdef NSGAIII_modified < ALGORITHM
% <2014> <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% Nondominated sorting genetic algorithm III
% retrain --- 0 --- Whether retrain the model
% interval --- 10 --- Training interval (1, 5, 10, 20, 50)
% percentage --- 0.25 --- Percentage of replaced offsprings
% epochs --- 100 --- Number of training epochs for the model
% alpha --- NaN --- Alpha value (NaN means a random value)
% normalization --- false --- Whether normalize the objective vectors and decsion vectros

%------------------------------- Reference --------------------------------
% K. Deb and H. Jain, An evolutionary many-objective optimization algorithm
% using reference-point based non-dominated sorting approach, part I:
% Solving problems with box constraints, IEEE Transactions on Evolutionary
% Computation, 2014, 18(4): 577-601.
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
            %% Generate the reference points and random population
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
            Zmin          = min(Population(all(Population.cons<=0,2)).objs,[],1);
            assert(length(Algorithm.parameter)==6 || isempty(Algorithm.parameter));
            [retrain, interval, percentage, epochs,  alpha, normalization] = Algorithm.ParameterSet(0, 10, 0.25, 100, NaN, 0);
            % create model
            model = IM_SolutionGenerator(Problem.M, Problem.D, retrain, interval, percentage, epochs, alpha, ...
                Problem.upper, Problem.lower, normalization);
            gen = 1;
            model = model.UpdateUEA(Population,[],gen);
            Debug = 3;
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                Offspring  = OperatorGA(Problem,Population(MatingPool).decs);
                
                % create new solutions
                [Offspring, model, num, better_objs, dataset, prediction] = model.Generate(Offspring, gen);
                Offspring = Problem.Evaluation(Offspring);
                if Debug == 1
                    %% plot the better objective vectors
                    %% objective vectors created by the model and the algorithm
                    if num > 0
                    offs = Offspring.objs;
                    pops = Population.objs;
                    s = size(offs,1);
                    %PF = Problem.GetOptimum(100);
                    % scatter(pops(:,1), pops(:,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    % hold on;
                    
                    scatter(better_objs(:,1), better_objs(:,2), 50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    hold on;
                    
                    scatter(offs(1:s-num,1), offs(1:s-num,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'blue');
                    hold on;
                    scatter(offs(s-num+1:s,1), offs(s-num+1:s,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'red');
                    title(gen);
                    % xlim([0,8]);
                    % ylim([0,8]);
                    close all;
                    end
                elseif Debug == 2 %% examine the training results
                    if num>0
                    figure('Position',[100,100,1200, 330], 'Name',num2str(gen))
                    subplot(1,3,1);
                    set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    %% populations and predicted solutions
                    data_objs = dataset.objs;
                    s = size(data_objs,1);
                    n = Problem.N;
                    PF = Problem.GetOptimum(200);
                    scatter(PF(:,1), PF(:,2), 50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor',[0.7,0.7,0.7],'MarkerFaceColor', [0.7,0.7,0.7]);  hold on;  
                    data_objs = dataset.objs;
                    data_decs = dataset.decs;
                    num = size(data_objs,1);
                    test_decs = [];
                    for i = 1:num
                        num = size(data_objs,1);
                        t = randi(num);
                        distance = sum((repmat(data_objs(t,:),num,1)-data_objs).^2, 2);
                        [~,I] = mink(distance, 20);
                        test_decs = [test_decs; mean(data_decs(I,:),1)];
                    end
                   

                    prediction_sols = Problem.Evaluation(prediction);
                    %prediction_sols = Problem.Evaluation(test_decs);
                    prediction_objs = prediction_sols.objs;
                    prediction_decs = prediction_sols.decs;
                    show_flag = false;
                    if show_flag
                        scatter(data_objs(:,1), data_objs(:,2), 50, 'o',...
                            'filled','LineWidth',1,'MarkerEdgeColor','black', 'MarkerFaceColor',[0.8,0.8,0.8]);
                        scatter(prediction_objs(:,1), prediction_objs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                    else
                        scatter(data_objs(:,1), data_objs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                    end
                    
                    %xlim([0,3]);
                    %ylim([0,6]);
                    box on;
                    
                    subplot(1,3,2);
                     set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    if show_flag
                        scatter(data_decs(:,1), data_decs(:,2), 50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black', 'MarkerFaceColor',[0.8,0.8,0.8]);
                        scatter(prediction_decs(:,1), prediction_decs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');

                    else
                        scatter(data_decs(:,1), data_decs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                        
                    end
                    %legend(["EMOA", "Model"], 'Location','northoutside', 'NumColumns',1)
                    box on; 
                        subplot(1,3,3);
                     set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    scatter(data_decs(:,1), data_decs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                    
                    box on;
                    close all;

                        % %% plot the training results
                        % figure('Position',[100,100,1200, 500])
                        % subplot(1,3,1);
                        % data_objs = dataset.objs;
                        % data_decs = dataset.decs;
                        % scatter(data_objs(:,1), data_objs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                        % hold on;
                        % prediction_sols = Problem.Evaluation(prediction);
                        % prediction_objs = prediction_sols.objs;
                        % scatter(prediction_objs(:,1), prediction_objs(:,2), 50, 'o','filled','LineWidth',1,'MarkerEdgeColor','red');
                        % subplot(1,3,2);
                        % hold on;
                        % scatter(data_decs(:,1), data_decs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                        % hold on;  
                        % scatter(prediction(:,1), prediction(:,2), 50, 'o','filled','LineWidth',1,'MarkerEdgeColor','red');
                        % subplot(1,3,3);
                        % data_decs = tsne(data_decs);
                        % scatter(data_decs(:,1), data_decs(:,2), 50, data_objs(:,1), 'o','filled','LineWidth',1,'MarkerEdgeColor','black');
                        % title('TSNE');
                        % hold on;
                        % close all;
                    end
                elseif Debug == 3
                    if num>0
                    figure('Position',[100,100,1200, 430], 'Name',num2str(gen))
                    subplot(1,3,1);
                    set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    %% populations and predicted solutions
                    data_objs = dataset.objs;
                    s = size(data_objs,1);
                    n = Problem.N;
                    PF = Problem.GetOptimum(200);
                    if Problem.M == 2
                    scatter(PF(:,1), PF(:,2), 50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor',[0.7,0.7,0.7],'MarkerFaceColor', [0.7,0.7,0.7]);  hold on;  
                    scatter(data_objs(1:s-n,1), data_objs(1:s-n,2), 80,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'red');  hold on;  
                    scatter(data_objs(s-n+1:s,1), data_objs(s-n+1:s,2), 50,...
                        'x','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'blue'); hold on;  
          
                    scatter(better_objs(:,1), better_objs(:,2), 30, ...
                        'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    else
                         scatter3(PF(:,1), PF(:,2), PF(:,3),50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor',[0.7,0.7,0.7],'MarkerFaceColor', [0.7,0.7,0.7]);  hold on;  
                    scatter3(data_objs(1:s-n,1), data_objs(1:s-n,2), data_objs(1:s-n,3),50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'red');  hold on;   
                    scatter3(data_objs(s-n+1:s,1), data_objs(s-n+1:s,2), data_objs(s-n+1:s,3),50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'blue'); hold on;   
                    scatter3(better_objs(:,1), better_objs(:,2), better_objs(:,3),50, ...
                        'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    view(3);
                    end
                    legend(["PF", "Previous", "Current", "Better"], 'Location','northoutside', 'NumColumns',2);
                    %xlim([0,3]);
                    %ylim([0,6]);
                    box on;
                    subplot(1,3,2);
                     set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    offs = Offspring.objs;
                    s = size(offs,1);
                    if Problem.M == 2
                    scatter(PF(:,1), PF(:,2), 50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor',[0.7,0.7,0.7],'MarkerFaceColor', [0.7,0.7,0.7]);  
                    hold on;  

                    scatter(offs(1:s-num,1), offs(1:s-num,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'yellow');
                    hold on;
                    scatter(offs(s-num+1:s,1), offs(s-num+1:s,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    hold on;
                    else
                    scatter3(PF(:,1), PF(:,2), PF(:,3),50,...
                        'o','filled','LineWidth',1,'MarkerEdgeColor',[0.7,0.7,0.7],'MarkerFaceColor', [0.7,0.7,0.7]);  
                    hold on;  

                    scatter3(offs(1:s-num,1), offs(1:s-num,2), offs(1:s-num,3),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'yellow');
                    hold on;
                    scatter3(offs(s-num+1:s,1), offs(s-num+1:s,2),offs(s-num+1:s,3),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    hold on;
                    view(3);
                    end
                    legend(["PF","EMOA", "Model"], 'Location','northoutside', 'NumColumns',2)
                    %xlim([0,3]);
                    %ylim([0,6]);
                    box on;
                    subplot(1,3,3);
                     set(gca, 'FontName', 'Times New Roman', 'FontSize', 24); hold on;
                    offs = Offspring.decs;
                    scatter(offs(1:s-num,1), offs(1:s-num,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'yellow');
                    hold on;
                    
                    scatter(offs(s-num+1:s,1), offs(s-num+1:s,2),50, 'o','filled','LineWidth',1,'MarkerEdgeColor','black','MarkerFaceColor', 'green');
                    legend(["EMOA", "Model"], 'Location','northoutside', 'NumColumns',1)
                    box on; 
                    close all;
                    end
                end

                Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,Z,Zmin);

                gen = gen + 1;
                model = model.UpdateUEA(Population, Offspring, gen);

            end
        end
    end
end