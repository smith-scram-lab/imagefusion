    clc;
    clear all;
    count = [];
    avg_score = 0;
    score_before = 0;

    for i = 0:955
        image1 = imread(fullfile('../polar_tt/predict_C_raw', sprintf('%d_predict.png', i)));
        image2 = imread(fullfile('../polar_vp/predict_C_raw', sprintf('%d_predict.png', i)));
        result1 = imread(fullfile('../polar_tt/label', sprintf('%d.tif', i)));
        result2 = imread(fullfile('../polar_vp/label', sprintf('%d.tif', i)));
        result3 = imread(fullfile('../label', sprintf('%d.tif', i)));
        %imshow(image1);
        %imshow(image2);
        %result = zeros(256,256);
        if(size(image1, 3)>1)
            image1 = rgb2gray(image1);
        end
        if(size(image2, 3)>1)
            image2 = rgb2gray(image2);
        end
        fused_image = fuse_images(image1, image2);
        
        %imshow(fused_image);

        score1 = score(result3, image1);
        score2 = score(result3, image2);
        score_new = score(result3, fused_image);
        avg_score = avg_score + score_new;
        score_before = score_before + max(score1, score2);

        fprintf('Score1: %f\n', score1);
        fprintf('Score2: %f\n', score2);
        fprintf('The new IOU: %f\n', score_new);
        
        if score_new > score1 && score_new > score2
            count = [count; i];
            imwrite(image1, fullfile('fused', sprintf('%d_tt.png', i)));
            imwrite(image2, fullfile('fused', sprintf('%d_vp.png', i)));
            imwrite(fused_image, fullfile('fused', sprintf('%d_predict.png', i)));
            imwrite(result3, fullfile('fused', sprintf('%d_label.png', i)));
        end
    end

    disp(count);
    disp(avg_score / 956);
    disp(score_before / 956);