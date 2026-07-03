function [] = plot_array_layout(array,damaged_flag) 

    % Plot the array layout
    % Custom 
    % damaged_flag - A flag to denote whether the damaged array is being represented or not
    figure;
    plot3(array.positions(1,:), array.positions(2,:), array.positions(3,:), 'ko', 'MarkerSize', 6, 'LineWidth', 1.5);
    xlabel('X (λ)'); ylabel('Y (λ)'); zlabel('Z (λ)');
    title('Antenna Array Geometry (Custom)');
    grid on; axis equal; view(3); hold on;
    normals = array.element_normals;
    quiver3(array.positions(1,:), array.positions(2,:), array.positions(3,:), ...
    normals(1,:), normals(2,:), normals(3,:), ...
    0.5, 'b'); % 0.5 = arrow scaling factor
    if ~isempty(array.damaged_elements) && damaged_flag == true
        damaged_positions = array.positions(:,array.damaged_elements);
        plot3(damaged_positions(1,:),damaged_positions(2,:),damaged_positions(3,:),'.','Color','r','MarkerSize', 15)
    end

end