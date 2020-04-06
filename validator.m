load('my_svm.mat')

pos_imageDir = 'cropped_validation_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg', pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_validation_images_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg', neg_imageDir));
neg_nImages = length(neg_imageList);

valid_tp = 0;
valid_fn = 0;
fn_imageDir = 'false_negative_validation_images_faces';
if ~exist(fn_imageDir, 'dir')
    mkdir(fn_imageDir);
end

for i=1:pos_nImages
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    confidence = feat(:)' * w + b;
    if confidence >= 0
        valid_tp = valid_tp + 1;
    else
        valid_fn = valid_fn + 1;
        % copyfile(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name), fn_imageDir);
    end
end

valid_tn = 0;
valid_fp = 0;
fp_imageDir = 'false_positive_validation_images_faces';
if ~exist(fp_imageDir, 'dir')
    mkdir(fp_imageDir);
end

for i=1:neg_nImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    confidence = feat(:)' * w + b;
    if confidence < 0
        valid_tn = valid_tn + 1;
    else
        valid_fp = valid_fp + 1;
        % copyfile(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name), fp_imageDir);
    end
end

fprintf('  true  positive: %d out of %d images\n', valid_tp, pos_nImages);
fprintf('  false positive: %d out of %d images\n', valid_fp, neg_nImages);
fprintf('  true  negative: %d out of %d images\n', valid_tn, neg_nImages);
fprintf('  false negative: %d out of %d images\n', valid_fn, pos_nImages);