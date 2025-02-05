function overlap_percentage = overlap_percentage(img1, img2)
    overlap = sum((img1 > 0) & (img2 > 0), 'all');
    img1_non_black = sum(img1 > 0, 'all');
    img2_non_black = sum(img2 > 0, 'all');
    
    if img1_non_black == 0 || img2_non_black == 0
        overlap_percentage = 0;
        return;
    end
    
    overlap_percentage = overlap / min(img1_non_black, img2_non_black);
end