function [output] = beamforming_metrics(type, varargin)
%
%
% for modularity, can use direct application in place of local functions in
% analyze_beamforming.m
%
% EX: Sidelobe_Suppression = beamforming_metrics('sidelobe suppression',
% results);
%
% for results struct so can read optimized_pattern, damaged_pattern,
% original_pattern
%
% directivity and robustness to have respective new code incorporated
%

    switch lower(type)
        case 'sidelobe suppression'
            output = computeSLS(varargin{:});
        case 'sinr'
            output = computeSINR(varargin{:});
        case 'mainbeam width'
            output = computeMBW(varargin{:});
        case 'pattern quality'
            output = computePQ(varargin{:});
        case 'power efficiency'
            output = computePE(varargin{:});
        case 'directivity'
            output = compute_direct(varargin{:});
        case 'robustness'
            output = compute_robust(varargin{:});
        otherwise
            error('Unknown metric type: %s', type);
    end

    function sidelobe_suppression = computeSLS(opt_pattern, damaged_pattern)
        % Calculate improvement in sidelobe levels
        angles = linspace(-90, 90, length(opt_pattern)); %results.optimized_pattern
        
        % Define near-sidelobe region (±30° excluding main beam)
        near_region = (abs(angles) <= 30) & (abs(angles) >= 5);
        
        % Calculate maximum sidelobe levels for both patterns
        % if isfield(results, 'damaged_pattern')
            orig_sll = max(damaged_pattern(near_region)); %results.damaged_pattern
        % else
            % orig_sll = max(results.original_pattern(near_region));
        % end
        opt_sll = max(opt_pattern(near_region));
        
        % Return improvement in dB
        sidelobe_suppression = abs(orig_sll - opt_sll);  
    end

    function sinr = computeSINR(pattern)
        % Calculate Signal-to-Interference-plus-Noise Ratio
        angles = linspace(-90, 90, length(pattern));
        
        % Define main beam region (±3°) and interference region
        main_beam = abs(angles) <= 3;
        interference = ~main_beam;
        
        % Calculate powers in linear scale
        signal_power = sum(10.^(pattern(main_beam)/10));
        interference_power = sum(10.^(pattern(interference)/10));
        noise_floor = 10^(-60/10);  % -60 dB noise floor
        
        % Calculate SINR in dB
        sinr = 10 * log10(signal_power / (interference_power + noise_floor));   
    end

    function width = computeMBW(pattern_raw)
        
        % ADDED - Convert to dB - can remove this depending if already
        % in dB
        pattern = 10*log10(pattern_raw);

        % Calculate 3dB beamwidth of the main lobe
        angles = linspace(-90, 90, length(pattern));
        
        % Find peak and 3dB points
        [peak_val, peak_idx] = max(pattern);
        threshold = peak_val - 3;
        
        % Find -3dB points
        left_idx = find(pattern(1:peak_idx) <= threshold, 1, 'last');
        right_idx = peak_idx + find(pattern(peak_idx:end) <= threshold, 1) - 1;
        
        % Calculate width in degrees
        width = angles(right_idx) - angles(left_idx);

    end

    function quality = computePQ(damaged_pattern, opt_pattern)%(results)
        % if isfield(results, 'damaged_pattern')
            ref_pattern = damaged_pattern; %results.damaged_pattern;
        % else
        %     ref_pattern = results.original_pattern;
        % end
        
        % Use log-scale differences to handle large dynamic ranges
        pattern_diff = abs(opt_pattern - ref_pattern); %results.optimized_pattern - ref_pattern);
        pattern_range = max(ref_pattern) - min(ref_pattern);
        
        % Add small epsilon to avoid division by zero
        epsilon = 1e-10;
        quality = 1 - mean(pattern_diff) / (pattern_range + epsilon);
        
        % Ensure quality is in [0,1] range
        quality = max(0, min(1, quality));
        end
        
    function efficiency = computePE(weights)
        % Calculate array power efficiency using proper normalization
        total_power = sum(abs(weights).^2);
        num_elements = length(weights);
        
        % Calculate uniformity of power distribution
        ideal_power_per_element = total_power / num_elements;
        actual_powers = abs(weights).^2;
        power_deviation = std(actual_powers) / ideal_power_per_element;
        
        % Convert to percentage with proper normalization
        efficiency = (1 - power_deviation) * 100;
    end
    
    function directivity = compute_direct(pattern)
        % Calculate array directivity
        angles = linspace(-90, 90, length(pattern));
    
        % Convert pattern to linear scale
        pattern_linear = 10.^(pattern/10);
    
        % Calculate directivity using standard formula
        numerator = max(pattern_linear);
        denominator = mean(pattern_linear);
    
        directivity = 10 * log10(numerator / denominator);
    
        % angles = linspace(-90, 90, length(pattern));
        % fc = 300e6;
        % directivity1 = mat_directivity(pattern,fc,angles);
    end

    function robustness = compute_robust(weights)
        % Calculate robustness based on weight distribution stability
        num_elements = length(weights);
        
        % Analyze weight distribution properties
        normalized_weights = weights / max(abs(weights));
        weight_gradient = diff(normalized_weights);
        
        % Calculate stability metrics
        amplitude_stability = 1 - std(normalized_weights);
        gradient_stability = 1 - std(weight_gradient);
        
        % Combine metrics with proper scaling
        robustness = (0.6 * amplitude_stability + 0.4 * gradient_stability) * 100;
    end

end

