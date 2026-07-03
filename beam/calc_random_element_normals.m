function [element_normals] = calc_random_element_normals(array,ele)

N = size(array.positions, 2);   
element_normals = zeros(3, N);
    for i = 1:N
        tilt_el = deg2rad(ele.el_in_deg + ele.el_variability*rand()); % tilt in elevation
        tilt_az = deg2rad(ele.az_in_deg + ele.az_variability*rand()); % tilt in azimuth
        element_normals(:, i) = [
        cos(tilt_el)*cos(tilt_az);
        cos(tilt_el)*sin(tilt_az);
        sin(tilt_el)
        ];
    end


end