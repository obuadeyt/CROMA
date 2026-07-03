function weights = initialize_weights(array)
    % Initialize with modified Dolph-Chebyshev
    rowTaper = hamming(array.Nz);         % Calculate Row taper
    colTaper = hamming(array.Ny);          % Calculate Column taper
    taper = rowTaper.*colTaper';    % Calculate taper
    amp = reshape(taper,[],1); % Reshape array for multiplication   
    % phase = linspace(0, pi, N)'; % Linear progressive phase
    phase = 0; % No phase
    weights = amp .* exp(1j * phase); % Complex weight
    
    % Apply damage constraints
    if ~isempty(array.damaged_elements)
        array.damaged_weights(array.damaged_elements) = 0;
    end
end