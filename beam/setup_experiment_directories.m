function setup_experiment_directories()
    % Create and verify all necessary directories for the experiment
    
    % List of required directories
    required_dirs = {...
        'figures', ...
        'results', ...
        'metrics'};
    
    % Get current directory
    base_dir = pwd;
    
    % Create each directory if it doesn't exist
    for i = 1:length(required_dirs)
        dir_path = fullfile(base_dir, required_dirs{i});
        if ~exist(dir_path, 'dir')
            [success, msg] = mkdir(dir_path);
            if ~success
                error('Failed to create directory %s: %s', required_dirs{i}, msg);
            end
        end
        
        % Verify directory is writable
        test_file = fullfile(dir_path, 'test_write.tmp');
        try
            fid = fopen(test_file, 'w');
            if fid == -1
                error('Directory %s is not writable', required_dirs{i});
            end
            fclose(fid);
            delete(test_file);
        catch
            error('Directory %s exists but is not writable', required_dirs{i});
        end
    end
end