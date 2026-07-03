% Utility function for normals (az, el format if needed)
function normvecs = azel2norm(vectors)
    normvecs = zeros(2, size(vectors, 2));
    for i = 1:size(vectors, 2)
        dir = vectors(:, i) / norm(vectors(:, i));
        az = atan2d(dir(2), dir(1));
        el = asind(dir(3));
        normvecs(:, i) = [az; el];
    end
end
