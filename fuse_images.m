function fused_image = fuse_images(image1, image2)
    if overlap_percentage(image1, image2) > 0.95
        image1(image1 < 150) = 0;
        image1(image1 > 150) = 1;
        fused_image = image1 * 255;
        return;
    end

    % Normalize pixel values to range [0, 1]
    image1_norm = double(image1) / 255.0;
    image2_norm = double(image2) / 255.0;

    if sum(image1_norm(:)) < 20
        fused_image = image2_norm * 255;
    elseif sum(image2_norm(:)) < 20
        fused_image = image1_norm * 255;
    else
        fused_image = (image1_norm + image2_norm) / 2.0 * 255;
    end
end