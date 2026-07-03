%% BEAMeR Pipeline
% Lauren Kight, Syndey Zink, Alec McAree
% 1/9/26

clear all, clc
%% Overview
% Array simulation and weighting scheme optimization to improve Tx and Rx
% beamforming

% Pipeline components/code blocks:
% - Array Parameters
% - Options
% - Necessary Varibles (creates necessary variables to run scripts)
% - Optimization Scheme (such as AMSO)
% - Calculate Array Factor (AF) and total radiation pattern or beampattern
% (BP)
% - Format Conversions (normalization, power, conversion to decible
% scale, ...)
% - Plotting and Metrics
% - Saving Results

%% Naming Convention of Variables
% Currently using the single struct "array" but planning to transition to a
% more organized structure after getting the code debugged

%% Array Parameters
% Physical array parameters. This code supports uniform linear arrays
% (ULAs), 2D planar arrays, and 3D conformal arrays. 

% Array dimensions
array.Ny = 4; % Number elements in y plane
array.Nz = 2; % Number elements in z plane (Note: Matlab's default orientation
    % for phased array toolbox is the array facing into the x-plane so for
    % verification we are modeling the same way)
array.num_elements = array.Ny * array.Nz;

% Array 
array.lambda = 1; % Wavelength
array.k = 2*pi/array.lambda; % Wavenumber
array.fc = 3e8; % Center frequency
% *Grating Lobes* -- If wanting to add grating lobes the spacing between
% the elements must be larger than lambda/2 
array.dy = array.lambda/2; % Spacing in y
array.dz = array.lambda/2; % Spacing in z

% Steering vector, k0
array.theta0 = 90; % elevation angle
array.phi0 = 0; % azimuth angle

% Type of element
array.use_cosine_element = true; % Cosine element = true; Isotropic = false;
array.cosine_power = 1; % Cosine power of element

% Element pattern selections 
% array.element_pattern_type = 'custom_antenna';
% array.element_pattern.az   = deg2rad(azGrid);    % radians
% array.element_pattern.el   = deg2rad(elGrid);    % radians
% array.element_pattern.gain = Pat_lin;             % linear power

array.element_pattern_type = 'isotropic' ; % Options include: 'isotropic',
% 'cosine', and 'cosine_squared'
% If selecting Gaussian 

% Angle range of optimization 
array.angle_range = [-90 90];
array.angles = linspace(array.angle_range(1), array.angle_range(2), 181);

% Note: Future work shoud include improving the below process
% Type of array if running a single instance. ~*~ Comment out if running
% statistical analysis ~*~
array.array_type = 'sparse'; % 'ideal' or 'sparse'

% Set sparsity parameters if desired *~* Comment out if running an ideal
% array or statistical analysis *~*
% array.sparsity_percentage = 0.1;

array.damaged_elements = [1]; % Alternatively the elements can set
% otherwise they are randomly selected by use of the sparsity_percentage.
% *Upper Left 8x8 Block of 32x32 Element Array: 
% array.damaged_elements = [25:32 57:64 89:96 121:128 153:160 185:192 217:224 249:256];

% Set look direction of elements
% Options include:
% - 'broadside' which points elements directly into X+
% - 'all_tilt' which causes all elements to look the same direction and
% must include variable values for variables el_in_deg and az_in_deg
% - 'random' which allows for a specified amount of random variablility to
% be added. Must include values for el_in_deg, az_in_deg, el_variability,
% and az_variability. 
% - 'custom' which allows the user to specify the look direction here at
% the top of the script
% Comment out other options

% Broadside
ele.type = 'broadside';

% % All tilted | Directly into X+ is 0
% ele.type = 'all_tilt';
% ele.el_in_deg = 10;
% ele.az_in_deg = -5;
% 
% % Random
% ele.type = 'random';
% ele.el_in_deg = 0;
% ele.az_in_deg = 0;
% % Variability is multiplied by rand() which produces a random number
% % between 0 and 1 and then that total is added to the elevation or azimuth
% % respectively 
% ele.el_variability = 15;
% ele.az_variability = 0;
% 
% % Custom
% % Must be a 3 by N matrix. Note that el and az at 0 are broadside
% (directly into X+) and values need to be in radians
% ele.type = 'custom';
% array.element_normals = ;

%% Options
% Plotting options
array.plot_mat_comparison = true; % 'true' or 'false'
array.plot_array = true; % 'true' or 'false'

% Save workspace variable to folder
array.save_workspace = true;

array.random_seed = []; % Set random seed if desired

%% Necessary Variables

% Create array geometry in Y-Z plane (pointing X+)
% Assuming equal spacing currently - future work will include allowing for
% a varied geometry, but this will require figuring out how to properly
% plot the beampattern.
[y_grid, z_grid] = meshgrid(array.dy*(0:array.Ny-1), array.dz*(0:array.Nz-1));
y = y_grid(:); 
z = z_grid(:);
x = zeros(size(y));
array.positions = [x'; y'; z']; % Ideal array

theta_scan = linspace(0,pi,181);
phi_scan = linspace(-pi, pi, 361); % -180 to 180° AKA azimuth angle % !! Need to fix this in Sydney's script but for the sake of running it now I'm chanign this
[TH, PH] = meshgrid(theta_scan, phi_scan);
TH = rad2deg(TH);
PH = rad2deg(PH);
array.TH = TH;
array.PH = PH;

% Initialize weights for the array
amp = ones(array.num_elements,1);
phase = 0; % No phase currently (we're not using analog arrays, currently)
array.weights = amp .* exp(1j * phase); % Ideal weights

% Set element normals 
% Creating more flexibility--Code in progress
% el = 0 and az = 0 are broadside/pointing directly into X+
array.element_normals = assign_element_normals(array,ele);
if array.plot_mat_comparison == true 
    array.mat_element_normals = azel2norm(array.element_normals); % For Matlab array [az;el]
end

% Create ideal and damaged array structs
ideal_array = array;
damaged_array = array; % if no damage has been specific this will remain the same as the ideal array

% Sparse array or array with dead elements
if strcmp(array.array_type, 'sparse')
        updated_array = remove_elements(damaged_array);
        clear damaged_array;
        damaged_array = updated_array; % Awkward, but Matlab is funny about passing structs
end

% Calculate reference patterns
% Ideal array
[ideal_beampattern_AZ_slice, ideal_BP_3D_norm, ideal_BP_raw, ideal_BP_3D_raw] = calculate_pattern(ideal_array.weights, ideal_array);
% Structs cannot be saved from a multiple output function call
% (idk--don't ask me why)
array.ideal_pattern = ideal_beampattern_AZ_slice;
array.ideal_pattern_3D = ideal_BP_3D_norm;
% Damaged array 
[damaged_beampattern_AZ_slice, damaged_BP_3D_norm, damaged_BP_raw, damaged_BP_3D_raw] = calculate_pattern(damaged_array.weights, damaged_array);
array.damaged_pattern = damaged_beampattern_AZ_slice;
array.damaged_pattern_3D = damaged_BP_3D_norm;


% Plot Arrays and Beampattern Comparison if Selected
% Plot the array layout showing location of elements, normals, and
% which elements are disabled
if array.plot_array == true
    damaged_flag = false; % A flag to denote whether the damaged array is being represented or not
    plot_array_layout(ideal_array,damaged_flag)
    damaged_flag = true;
    plot_array_layout(damaged_array,damaged_flag);
end

% Plot beampattern comparison with Matlab's builtin functions
if array.plot_mat_comparison == true
    plot_matlab_comparison(ideal_array, array.ideal_pattern_3D)
    plot_matlab_comparison(damaged_array, array.damaged_pattern_3D)
end

% Create array for optimization 
optimizing_array = damaged_array;

%% Optimization
% Multi-stage optimization
stages = {'main', 'near', 'far'};
current_weights = initialize_weights(damaged_array); % Hamming weights to give the optimization algorithm a "leg up"

% Intialize
metricIndex = [];

for stage_idx = 1:length(stages)
    stage = stages{stage_idx};

    [optimized_beampattern_AZ_slice, ~, ~, ~] = calculate_pattern(current_weights, optimizing_array);
    metricIndex = update_metric_index(metricIndex, stage_idx, damaged_beampattern_AZ_slice, optimized_beampattern_AZ_slice, current_weights); %results, weights)
    
    [current_weights, stage_history] = optimize_stage(current_weights, optimizing_array, stage, stage_idx);
    
    if stage_idx == 1
        convergence_history = stage_history;
    end
end

% Calculate final pattern
[optimized_beampattern_AZ_slice, optimized_BP_3D_norm, optimized_BP_raw, optimized_pattern_BP_dB] = calculate_pattern(current_weights, optimizing_array);
metricIndex = update_metric_index(metricIndex, (length(stages)+ 1), damaged_beampattern_AZ_slice, optimized_beampattern_AZ_slice, current_weights); %results, weights)

%% Plotting and Metrics
% Plot Az and El plots for Ideal, Damaged, and Optimized
AZ_EL_plots(ideal_BP_raw,'Ideal Array');
AZ_EL_plots(damaged_BP_raw, 'Damaged Array');
AZ_EL_plots(optimized_BP_raw, 'After AMSO Optimization')

% Output Metrics 
metricNames = {
    'Sidelobe Suppression'
    'SINR'
    'Mainbeam Width'
    'Pattern Quality'
    'Power Efficiency'
    'Directivity'
    'Robustness'};

[numMetrics, numStages] = size(metricIndex);

for metricIdx = 1:numMetrics
    metricName = metricNames{metricIdx};
    for stageIdx = 1:numStages
        metricValue = metricIndex(metricIdx, stageIdx);
        % Optimization Stage Label 
        if stageIdx == numStages
            stageLabel = 'Final';
        else
            stageLabel = ['Stage ' num2str(stageIdx)];
        end
        % Print Result 
        fprintf('%s %s: %.6f\n', metricName, stageLabel, metricValue);
    end
end

%% Storing Results
% Store results
results = struct();
results.type = array.array_type;
results.original_weights = array.weights;
results.damaged_weights = damaged_array.weights;
results.optimized_weights = current_weights;
results.original_pattern = array.ideal_pattern;
results.damaged_pattern = array.damaged_pattern;
results.optimized_pattern = optimized_beampattern_AZ_slice;
results.damaged_elements = damaged_array.damaged_elements;
results.convergence_history = convergence_history;

% Save the workspace if toggled on to the Results_workspace folder
if array.save_workspace == true
    outputDir = fullfile(pwd, 'Results_workspace');
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    ts = datestr(now, 'yyyymmdd_HHMMSS');  % e.g., 20251230_115712
    fname = sprintf('workspace_%s.mat', ts);
    save(fullfile(outputDir, fname));       % saves entire workspace
end

%% Metrics Call-in

% EX: Sidelobe_Suppression = beamforming_metrics('sidelobe suppression',

% beamforming_metrics next update some parametric inputs will change for
% some metrics

% Sidelobe_Suppression = beamforming_metrics('sidelobe suppression', results);
% SINR = beamforming_metrics('sinr', pattern);
% MB_width = beamforming_metrics('mainbeam width', pattern);
% Pattern_qual = beamforming_metrics('pattern quality', results);
% Power_Efficiency = beamforming_metrics('power efficiency', weights);
% Directivity = beamforming_metrics('directivity', pattern);
% Robustness = beamforming_metrics('robustness', weights);

function metricIndex = update_metric_index(metricIndex, stage, damaged_pattern, opt_pattern, weights) % [metricIndex, finalMetrics] = update_metric_index(metricIndex, stage, totalStages, damaged_pattern, opt_pattern, weights) 
    m1 = beamforming_metrics('sidelobe suppression', opt_pattern, damaged_pattern);
    m2 = beamforming_metrics('sinr', opt_pattern);
    m3 = beamforming_metrics('mainbeam width', opt_pattern);
    m4 = beamforming_metrics('pattern quality', damaged_pattern, opt_pattern); %results);
    m5 = beamforming_metrics('power efficiency', weights);
    m6 = beamforming_metrics('directivity', opt_pattern);
    m7 = beamforming_metrics('robustness', weights);
    
    metrics_column = [
        ensure_scalar(m1);
        ensure_scalar(m2);
        ensure_scalar(m3);
        ensure_scalar(m4);
        ensure_scalar(m5);
        ensure_scalar(m6)
        ensure_scalar(m7)];
    
    if isempty(metricIndex)
        metricIndex = metrics_column;
    else
        metricIndex(:,stage) = metrics_column;
    end
end

function val = ensure_scalar(x)
    if isempty(x)
        val = NaN;
    elseif numel(x) ~= 1
        error('Metric must return scalar.');
    else
        val = x;
    end
end