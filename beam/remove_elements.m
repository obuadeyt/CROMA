function out_array = remove_elements(array)
    % Set random seed if provided for reproducibility
    if ~isempty(array.random_seed)
        rng(array.random_seed);
    end

    % Setup array characteristics based on type
    if isempty(array.damaged_elements)
        % Generate random damage pattern based on fraction
        num_damaged = ceil(array.sparsity_percentage * array.num_elements);
        array.damaged_elements = sort(randperm(array.num_elements, num_damaged));
    end

    % Set elements to zero/removing power (now a dead element)
    array.weights(array.damaged_elements) = 0;

    % Pass out the updated array
    out_array = array;
end
