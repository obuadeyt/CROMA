function [optimized_weights, history] = optimize_stage(initial_weights, array, stage, stage_idx)
  
    % Optimization settings
    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'MaxIterations', 1000, ...
        'MaxFunctionEvaluations', 1000, ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-8, ... % *** need to check before going to next iteration
        'ConstraintTolerance', 1e-8, ...
        'Display', 'none'); % 'iter'
    
    % Setup constraints
    lb = 0.001 * ones(array.num_elements, 1);
    ub = 1 * ones(array.num_elements, 1); % broaden out weights
    
    % Apply damage constraints
    if ~isempty(array.damaged_elements)
        lb(array.damaged_elements) = 0;
        ub(array.damaged_elements) = 0;
    end
    
    % Power conservation
    Aeq = ones(1, array.num_elements);
    beq = 1;
    
    % Progressive constraint learning
    beta = 1 / (1 + exp(-2*(stage_idx-2)));
    
    % Cost function with stage-specific weights
    cost_func = @(w) stage_cost(w, array, stage, beta);
    
    % Run optimization
    [optimized_weights, ~, ~, output] = fmincon(cost_func, initial_weights, ...
        [], [], Aeq, beq, lb, ub, [], options);
    
    history = output.iterations;
end