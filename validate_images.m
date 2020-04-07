load('my_svm.mat')
debug = false;

pos_imageDir = 'cropped_validation_images_faces';
pos_imageList = dir(fullfile(pos_imageDir, '*.jpg'));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_validation_images_notfaces';
neg_imageList = dir(fullfile(neg_imageDir, '*.jpg'));
neg_nImages = length(neg_imageList);

if debug
	valid_tp = 0;
	valid_fn = 0;
	for i=1:pos_nImages
		im = im2single(imread(fullfile(pos_imageDir, pos_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		confidence = feat(:)' * w + b;
		if confidence >= 0
			valid_tp = valid_tp + 1;
		else
			valid_fn = valid_fn + 1;
			fprintf('index: %d name: %s\n', i, fullfile(pos_imageDir, pos_imageList(i).name));
		end
	end

	valid_tn = 0;
	valid_fp = 0;
	for i=1:neg_nImages
		im = im2single(imread(fullfile(neg_imageDir, neg_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		confidence = feat(:)' * w + b;
		if confidence < 0
			valid_tn = valid_tn + 1;
		else
			valid_fp = valid_fp + 1;
			fprintf('index: %d name: %s\n', i, fullfile(neg_imageDir, neg_imageList(i).name));
		end
	end

	fprintf('true  positive: %d out of %d images\n', valid_tp, pos_nImages);
	fprintf('false positive: %d out of %d images\n', valid_fp, neg_nImages);
	fprintf('true  negative: %d out of %d images\n', valid_tn, neg_nImages);
	fprintf('false negative: %d out of %d images\n', valid_fn, pos_nImages);
else
	pos_feats = zeros(pos_nImages, featSize);
	for i = 1:pos_nImages
		im = im2single(imread(fullfile(pos_imageDir, pos_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		pos_feats(i, :) = feat(:);
	end

	neg_feats = zeros(neg_nImages, featSize);
	for i = 1:neg_nImages
		im = im2single(imread(fullfile(neg_imageDir, neg_imageList(i).name)));
		feat = vl_hog(im, cellSize);
		neg_feats(i, :) = feat(:);
	end

	feats = cat(1, pos_feats, neg_feats);
	labels = cat(1, ones(pos_nImages, 1), -1 * ones(neg_nImages, 1));

	[~, ~, ~, scores] = vl_svmtrain(feats', labels', 0, 'model', w, 'bias', b, 'solver', 'none');
	fprintf('Classifier performance on validation data:\n')
	report_accuracy(scores', labels);
end
