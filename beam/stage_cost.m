function cost = stage_cost(weights, array, stage, beta)
    beampattern = calculate_pattern(weights, array);
    
    % MAINLOBE
    % Calculate beam width with more precise control
    [~, peak_idx] = max(beampattern);
    beam_indices = find(beampattern >= max(beampattern) - 3);
    beam_width = array.angles(beam_indices(end)) - array.angles(beam_indices(1));
    
    % Quadratic penalty increases sharply as width deviates from 3 degrees
    width_error = (beam_width - 3)^4;  % Using 4th power for steeper penalty
    
    % Add extra penalty if width exceeds 3.5 degrees
    if beam_width > 3.5
        width_error = width_error * 100;
    end
    
    % NEAR SIDELOBES
    % Near sidelobe control
    near_region = (abs(array.angles) <= 45) & (abs(array.angles) >= 3); % Moved out to 45 degree from 30
    near_pattern = beampattern(near_region);
    near_angles = abs(array.angles(near_region));
    
    % Progressive sidelobe envelope
    target_envelope = -45 - beta * (near_angles - 3.5) * 0.2;
    near_violations = sum(max(0, near_pattern - target_envelope).^2);
    
    % FAR SIDELOBES
    % Far sidelobe control
    far_region = abs(array.angles) > 45;
    far_pattern = beampattern(far_region);
    far_violations = sum(max(0, far_pattern + 45).^2);
    
    % Stage-specific weighting
    switch stage
        case 'main'
            cost = width_error * 3000 + near_violations * 1000;
        case 'near'
            cost = width_error * 1000 + near_violations * 3000 + far_violations * 500;
        case 'far'
            cost = width_error * 500 + near_violations * 2000 + far_violations * 1500;
    end
end