function set_figure_properties_for_export(fig)
    % Set figure properties to ensure proper export without warnings
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPositionMode', 'manual');
    set(fig, 'PaperOrientation', 'landscape');
    
    % Get the figure's size in inches
    pos = get(fig, 'Position');
    screen_dpi = get(0, 'ScreenPixelsPerInch');
    width = pos(3) / screen_dpi;
    height = pos(4) / screen_dpi;
    
    % Set the paper size to match the figure size
    set(fig, 'PaperSize', [width height]);
    set(fig, 'PaperPosition', [0 0 width height]);
end