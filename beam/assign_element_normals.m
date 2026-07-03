function element_normals = assign_element_normals(array,ele)
% Function that allows for the look direction to be set at broadside (az =
% 0 and el = 0, all tilted the same direction, or set an amount of random
% variability


if strcmp(ele.type,'broadside')
    el_in_deg = 0;
    az_in_deg = 0;
    [element_normals] = calc_element_normals(array,el_in_deg,az_in_deg);
elseif strcmp(ele.type,'all_tilt')
    [element_normals] = calc_element_normals(array,ele.el_in_deg,ele.az_in_deg);
elseif strcmp(ele.type,'random')
    [element_normals] = calc_random_element_normals(array,ele);
elseif strcmp(ele.type,'custom')
    element_normals = array.element_normals;
end

end
