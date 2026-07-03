function visualize_array_results(results, config_name)
    % Ensure output directories exist
    setup_experiment_directories();
    
    % Create main figure
    h = figure('Position', [100 100 1200 800], 'Name', config_name);
    
    % Create subplots
    subplot(2,1,1);
    plot_beam_patterns(results);
    
    subplot(2,2,3);
    plot_weights_bars(results);
    
    subplot(2,2,4);
    plot_weights_continuous(results);
    
    % Set proper figure properties for export
    set_figure_properties_for_export(h);
    
    % Format config_name for filename
    timestamp =datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
    valid_filename = strrep(lower(config_name), ' ', '_');
    valid_filename = regexprep(valid_filename, '[^a-z0-9_]', '');
    valid_filename = [valid_filename '_' timestamp]
    
    % Save figure
    saveas(h, fullfile(pwd, 'figures', [valid_filename '_analysis.pdf']));
    saveas(h, fullfile(pwd, 'figures', [valid_filename '_analysis.fig']));
    % close(h);
end

function plot_beam_patterns(results)
    angles = linspace(-90, 90, length(results.optimized_pattern));
    
    if isfield(results, 'damaged_pattern')
        plot(angles, results.damaged_pattern, 'r--', 'LineWidth', 1.5, ...
             'DisplayName', 'Damaged');
    else
        plot(angles, results.original_pattern, 'b-', 'LineWidth', 1.5, ...
             'DisplayName', 'Original');
    end
    
    hold on;
    plot(angles, results.optimized_pattern, 'g-', 'LineWidth', 2, ...
         'DisplayName', 'Optimized');
    
    % Add region markers
    plot([-30 -30], [-60 0], 'k:', 'LineWidth', 1, 'DisplayName', 'Near/Far Boundary');
    plot([30 30], [-60 0], 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
    plot([-90 90], [-40 -40], 'r:', 'LineWidth', 1, 'DisplayName', '-40dB Level');
    
    grid on;
    xlabel('Angle (degrees)');
    ylabel('Magnitude (dB)');
    title('Beam Pattern');
    legend('Location', 'best');
    ylim([-60 0]);
    hold off;
end

function plot_weights_bars(results)
    num_elements = length(results.optimized_weights);
    element_indices = 1:num_elements;
    
    if isfield(results, 'damaged_weights')
        bar(element_indices, [results.damaged_weights(:), results.optimized_weights(:)], ...
            'grouped');
        legend('Damaged', 'Optimized');
    else
        bar(element_indices, [results.original_weights(:), results.optimized_weights(:)], ...
            'grouped');
        legend('Original', 'Optimized');
    end
    
    if isfield(results, 'damaged_elements') && ~isempty(results.damaged_elements)
        hold on;
        plot(results.damaged_elements, zeros(size(results.damaged_elements)), ...
             'rx', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
    end
    
    grid on;
    xlabel('Element Index');
    ylabel('Weight');
    title('Element Weights (Bar Plot)');
    ylim([0 0.06]);
end

function plot_weights_continuous(results)
    num_elements = length(results.optimized_weights);
    element_positions = (0:num_elements-1) * 0.5;
    
    if isfield(results, 'damaged_weights')
        plot(element_positions, results.damaged_weights, 'r--', ...
             'LineWidth', 1.5, 'DisplayName', 'Damaged');
    else
        plot(element_positions, results.original_weights, 'b-', ...
             'LineWidth', 1.5, 'DisplayName', 'Original');
    end
    
    hold on;
    plot(element_positions, results.optimized_weights, 'g-', ...
         'LineWidth', 2, 'DisplayName', 'Optimized');
    
    if isfield(results, 'damaged_elements') && ~isempty(results.damaged_elements)
        damaged_positions = element_positions(logical(results.damaged_elements));
        plot(damaged_positions, zeros(size(damaged_positions)), 'rx', ...
             'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Disabled');
    end
    
    grid on;
    xlabel('Element Position (wavelengths)');
    ylabel('Weight');
    title('Element Weights (Continuous)');
    ylim([0 0.06]);
    legend('Location', 'best');
    hold off;
end

function generate_paper_figures(all_results)
    % Generate all figures required for the paper
    
    % Figure 1: Pattern Comparison
    generate_pattern_comparison_figure(all_results);
    
    % Figure 2: Performance Comparison
    generate_performance_comparison_figure(all_results);
    
    % Additional analysis figures
    generate_statistical_analysis_figures(all_results);
end

function generate_pattern_comparison_figure(all_results)
    % Create figure
    h = figure('Position', [100 100 1200 800]);
    
    % Get mean metrics for visualization
    configs = fieldnames(all_results);
    num_angles = 181; % Standard from paper
    angles = linspace(-90, 90, num_angles);
    
    % Perfect Array (subplot 1)
    subplot(2,1,1);
    perfect_metrics = mean(all_results.perfect_array.metrics, 1);
    plot_theoretical_pattern(angles, 'perfect', perfect_metrics);
    title('Perfect Array Pattern');
    
    % 10% Damaged Array (subplot 2)
    subplot(2,1,2);
    damaged_metrics = mean(all_results.random_10pct_damage.metrics, 1);
    plot_theoretical_pattern(angles, 'damaged', damaged_metrics);
    title('10% Damaged Array Pattern');
    
    % Set proper figure properties for export
    set_figure_properties_for_export(h);
    
    % Save figure
    saveas(h, 'figures/pattern_comparison.pdf');
    saveas(h, 'figures/pattern_comparison.fig');
    % close(h);
end

function plot_theoretical_pattern(angles, config_type, metrics)
    % Generate theoretical patterns based on metrics
    main_beam_width = metrics(3);
    sidelobe_level = -metrics(1);
    directivity = metrics(6);
    
    % Create idealized pattern
    pattern = zeros(size(angles));
    
    % Main beam (Gaussian approximation)
    main_beam = exp(-4*log(2)*(angles/main_beam_width).^2);
    
    % Sidelobes
    sidelobe_mask = abs(angles) > main_beam_width/2;
    sidelobe_pattern = sidelobe_level * ones(size(angles));
    sidelobe_pattern(~sidelobe_mask) = 0;
    
    % Combine patterns
    pattern = 20*log10(main_beam + 10^(sidelobe_level/20) * sidelobe_mask);
    pattern = pattern - max(pattern); % Normalize
    
    % Plot patterns
    if strcmp(config_type, 'perfect')
        plot(angles, pattern, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original');
        hold on;
        plot(angles, pattern - metrics(4), 'g-', 'LineWidth', 2, 'DisplayName', 'Optimized');
    else
        plot(angles, pattern + sidelobe_level/2, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Damaged');
        hold on;
        plot(angles, pattern - metrics(4), 'g-', 'LineWidth', 2, 'DisplayName', 'Optimized');
    end
    
    % Add standard annotations
    plot([-30 -30], [-60 0], 'k:', 'LineWidth', 1, 'DisplayName', 'Near/Far Boundary');
    plot([30 30], [-60 0], 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
    plot([-90 90], [-40 -40], 'r:', 'LineWidth', 1, 'DisplayName', '-40dB Level');
    
    grid on;
    xlabel('Angle (degrees)');
    ylabel('Magnitude (dB)');
    legend('Location', 'best');
    ylim([-60 0]);
    hold off;
end

function generate_performance_comparison_figure(all_results)
    % Create figure
    h = figure('Position', [100 100 1200 400]);
    
    % Get metrics from results
    configs = fieldnames(all_results);
    num_configs = length(configs);
    
    % Metrics to plot
    metric_names = {'SINR (dB)', 'Sidelobe Level (dB)', 'Beam Width (deg)'};
    metric_indices = [2 1 3];  % Corresponding indices in metrics array
    
    for i = 1:3
        subplot(1,3,i);
        
        % Extract metric data
        metric_data = zeros(1, num_configs);
        for j = 1:num_configs
            metric_data(j) = mean(all_results.(configs{j}).metrics(:, metric_indices(i)));
        end
        
        % Create bar plot
        bar(metric_data);
        
        % Customize plot
        title(metric_names{i});
        set(gca, 'XTickLabel', strrep(configs, '_', ' '));
        xtickangle(45);
        grid on;
        
        % Add value labels on top of bars
        text(1:length(metric_data), metric_data, num2str(metric_data', '%.1f'), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    set_figure_properties_for_export(h);
    saveas(h, 'figures/performance_comparison.pdf');
    saveas(h, 'figures/performance_comparison.fig');
    % close(h);
end

function generate_statistical_analysis_figures(all_results)
    h = figure('Position', [100 100 1200 800]);
    
    % Plot distributions of metrics
    subplot(2,1,1);
    configs = fieldnames(all_results);
    metric_names = {'Sidelobe', 'SINR', 'Beam Width', 'Quality', ...
                   'Power Eff.', 'Direct.', 'Robust.'};
    
    hold on;
    colors = get(gca, 'ColorOrder');
    markers = {'o', 's', 'd', '^'};
    
    for i = 1:length(configs)
        data = all_results.(configs{i}).metrics;
        means = mean(data, 1);
        stds = std(data, [], 1);
        
        errorbar(1:length(metric_names), means, stds, ...
                ['-' markers{i}], 'Color', colors(i,:), ...
                'LineWidth', 1.5, 'MarkerSize', 8, ...
                'DisplayName', strrep(configs{i}, '_', ' '));
    end
    
    grid on;
    xlabel('Metric');
    ylabel('Value');
    title('Distribution of Performance Metrics');
    set(gca, 'XTick', 1:length(metric_names), 'XTickLabel', metric_names);
    xtickangle(45);
    legend('Location', 'eastoutside');
    
    set_figure_properties_for_export(h);
    saveas(h, 'figures/statistical_analysis.pdf');
    saveas(h, 'figures/statistical_analysis.fig');
    % close(h);
end