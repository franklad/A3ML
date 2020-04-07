close all
clear
% run('../vlfeat-0.9.20/toolbox/vl_setup')
debug = false;

pos_imageDir = 'cropped_training_images_faces';

pos_imageList = dir(sprintf('%s/*.jpg', pos_imageDir));
pos_nImages = length(pos_imageList);

% split images_faces to 80% training and 20% validation
pos_imageDir_v = 'cropped_validation_images_faces';
if ~exist(pos_imageDir_v, 'dir')
	mkdir(pos_imageDir_v);
	pos_nValid = round(pos_nImages * 0.2);

	for i = 1:pos_nValid
		name = pos_imageList(i).name;
		src = sprintf('%s/%s', pos_imageDir, name);
		movefile(src, pos_imageDir_v);
	end

	pos_imageList = dir(sprintf('%s/*.jpg', pos_imageDir));
	pos_nImages = length(pos_imageList);
end

neg_imageDir = 'cropped_training_images_notfaces';

neg_imageList = dir(sprintf('%s/*.jpg', neg_imageDir));
neg_nImages = length(neg_imageList);

% split images_notfaces to 80% training and 20% validation
neg_imageDir_v = 'cropped_validation_images_notfaces';
if ~exist(neg_imageDir_v, 'dir')
	mkdir(neg_imageDir_v);
	neg_nValid = round(neg_nImages * 0.2);

	for i = 1:neg_nValid
		name = neg_imageList(i).name;
		src = sprintf('%s/%s', neg_imageDir, name);
		movefile(src, neg_imageDir_v);
	end

	neg_imageList = dir(sprintf('%s/*.jpg', neg_imageDir));
	neg_nImages = length(neg_imageList);
end


cellSize = 4;
featSize = 31 * round(36 / cellSize) ^ 2;

pos_feats = zeros(pos_nImages, featSize);
for i=1:pos_nImages
	if ~debug
		im = im2single(imread(sprintf('%s/%s', pos_imageDir, pos_imageList(i).name)));
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
for i = 1:neg_nImages
	if ~debug
		im = im2single(imread(sprintf('%s/%s', neg_imageDir, neg_imageList(i).name)));
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
