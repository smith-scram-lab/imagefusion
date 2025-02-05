function overlap_score = score(label, predict)
    overlap = 0;
    total = 0;
    
    for i = 1:size(label, 1)
        for j = 1:size(label, 2)
            if label(i, j) >= 150 && predict(i, j) >= 150
                overlap = overlap + 1;
                total = total + 1;
            end
            if (label(i, j) >= 150) ~= (predict(i, j) >= 150)
                total = total + 1;
            end
        end
    end
    
    if total == 0
        overlap_score = 0;
    else
        overlap_score = overlap / total;
    end
end
