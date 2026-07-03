function [] = plot_matlab_comparison(array, beampattern3D)
% Plot comparison of custom array calculation versus Matlab's calculation
% for verification if desired
    
    N = size(array.positions, 2);

    % Element pattern for MATLAB array
    if array.use_cosine_element == true
        element = phased.CosineAntennaElement('CosinePower', array.cosine_power);
    else
        element = phased.IsotropicAntennaElement('BackBaffled', false);
    end

    % Element normals set facing straight into X+
    % Need to adjust when we add normals function
    mat_element_normals = repmat([0; 0], 1, N);

    hArray = phased.ConformalArray( ...
        'ElementPosition', array.positions, ...
        'ElementNormal', mat_element_normals, ...
        'Element', element,...
        'Taper', array.weights);

    % MATLAB reference pattern
    [matPattern_dB] = pattern(hArray, array.fc, -180:180, -90:90, ...
        'CoordinateSystem', 'polar', 'Type', 'powerdb', 'Normalize', false);
    matPattern = db2mag(matPattern_dB)';
    matPattern = matPattern / max(matPattern(:));
    
    figure, pattern(hArray, array.fc, -180:180, -90:90, ...
        'CoordinateSystem', 'polar', 'Type', 'powerdb', 'Normalize', false);

    % 3D Plotting
    [AZ, EL] = meshgrid(-180:180, -90:90);
    PHI = deg2rad(AZ);
    THETA = deg2rad(90 - EL);

    [X1, Y1, Z1] = sph2cart(PHI, pi/2 - THETA, beampattern3D');
    [X2, Y2, Z2] = sph2cart(PHI, pi/2 - THETA, matPattern');
    
    figure;
    surf(X1, Y1, Z1, beampattern3D', 'EdgeColor', 'none');
    title('Custom Beam Pattern (Non-uniform Normals & Weights)');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    colorbar;  view(3); camlight; lighting gouraud; alpha(0.8); xlim([0 1]); 
    
    figure;
    surf(X2, Y2, Z2, matPattern', 'EdgeColor', 'none');
    title('MATLAB Beam Pattern (Reference)');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    colorbar; view(3); camlight; lighting gouraud; alpha(0.7); xlim([0 1]);
end