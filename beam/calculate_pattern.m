function [beampattern_AZ_slice, BP_3D_norm, BP_3D_raw, BP_3D_dB] = ...
          calculate_pattern(weights, array)

    % Extract grid and array info
    TH = array.TH;   % elevation angles (deg)
    PH = array.PH;   % azimuth angles (deg)
    [Nr, Nc] = size(TH);
    N = size(array.positions, 2);   % number of elements

    % Steering direction unit vector
    k0 = [
          sind(array.theta0) * cosd(array.phi0);
          sind(array.theta0) * sind(array.phi0);
          cosd(array.theta0)
         ];

    % Preallocate
    BP_3D_raw = zeros(Nr, Nc);

    % Loop over scan directions
    for i = 1:Nr
        for j = 1:Nc

            theta = TH(i,j);
            phi   = PH(i,j);

            % Scan direction unit vector
            k_scan = [
                      sind(theta) * cosd(phi);
                      sind(theta) * sind(phi);
                      cosd(theta)
                     ];

            % Delta-k
            delta_k = k_scan - k0;

            % Phase shifts
            phase_shifts = exp(1j * array.k * ...
                              (array.positions.' * delta_k));

            % Element pattern model
            % Angle between scan direction and element normal
            cos_angle = k_scan.' * array.element_normals;   % 1×N
            cos_angle = max(min(cos_angle,1),-1);
            element_angles = acos(cos_angle);               % radians

            % ELEMENT PATTERN (CUSTOM ANTENNA TOOLBOX ELEMENT)
         
            % Convert scan direction to azimuth / elevation
            el = asin(k_scan(3));                 % elevation (rad)
            az = atan2(k_scan(2), k_scan(1));     % azimuth (rad)

            switch lower(array.element_pattern_type)

                case 'custom_antenna'
            
                    element_gain = zeros(1, N);
            
                    for n = 1:N
                        % Assume all elements share same orientation
                        az_n = az;
                        el_n = el;
            
                        element_gain(n) = interp2( ...
                            array.element_pattern.az, ...
                            array.element_pattern.el, ...
                            array.element_pattern.gain, ...
                            az_n, el_n, ...
                            'linear', 0 );
                    end


                case 'isotropic'
                    element_gain = ones(1, N);

                case 'cosine'
                    element_gain = max(cos(element_angles),0) ...
                                   .^ array.cosine_power;

                case 'cosine_squared'
                    element_gain = cos(element_angles).^2;
                    element_gain(element_angles > pi/2) = 0;

                otherwise
                    error('Unknown element pattern type');
            end


            % Total field
            BP_3D_raw(i,j) = abs( sum( ...
                weights(:).' .* ...
                phase_shifts.' .* ...
                element_gain ));
        end
    end

    % Normalization
    BP_3D_norm = abs(BP_3D_raw).^2;
    BP_3D_norm = BP_3D_norm ./ max(BP_3D_norm(:));
    BP_3D_dB   = 10*log10(max(BP_3D_norm, eps));


    % Azimuth Cut (theta ≈ boresight)
    [~, row0] = min(abs(TH(:,1) - array.theta0));
    beampattern_AZ_slice = BP_3D_norm(row0, :);

end
