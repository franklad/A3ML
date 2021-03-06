close all
clear
% run('../vlfeat-0.9.20/toolbox/vl_setup')
debug = false;

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(fullfile(pos_imageDir, '*.jpg'));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(fullfile(neg_imageDir, '*.jpg'));
neg_nImages = length(neg_imageList);

cellSize = 4;
featSize = 31 * round(36 / cellSize) ^ 2;

pos_feats = zeros(pos_nImages, featSize);
for i = 1 : pos_nImages
	if ~debug
		im = im2single(imread(fullfile(pos_imageDir, pos_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		pos_feats(i, :) = feat(:);
	else
		fprintf('got feat for pos image %d/%d\n', i, pos_nImages);
		imhog = vl_hog('render', feat);
		subplot(1, 2, 1);
		imshow(im);
		subplot(1, 2, 2);
		imshow(imhog)
		pause;
	end
end

neg_feats = zeros(neg_nImages, featSize);
for i = 1 : neg_nImages
	if ~debug
		im = im2single(imread(fullfile(neg_imageDir, neg_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		neg_feats(i, :) = feat(:);
	else
		fprintf('got feat for neg image %d/%d\n', i, neg_nImages);
		imhog = vl_hog('render', feat);
		subplot(1, 2, 1);
		imshow(im);
		subplot(1, 2, 2);
		imshow(imhog)
		pause;
	end
end

save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages')
