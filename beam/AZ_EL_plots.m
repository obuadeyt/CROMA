function [] = AZ_EL_plots(BP, super_title)
    theta_scan = linspace(0,pi,181);
    phi_scan = linspace(-pi, pi, 361);

    figure, 
    subplot(2,2,1), plot(rad2deg(theta_scan),BP(180,:)), xlabel('Degrees'), ylabel('Amplitude (dBi)'), title('Elevation Slice at 180');
    subplot(2,2,2), polarplot(theta_scan,BP(180,:)), title('Elevation Slice at 180')
    subplot(2,2,3), plot(rad2deg(phi_scan),BP(:,90)), xlabel('Degrees'), ylabel('Amplitude (dBi)'), title('Azimuth Slice at 90');
    subplot(2,2,4), polarplot(phi_scan,BP(:,90)), title('Azimuth Slice at 90')

    sgtitle(super_title);


end