%run('../vlfeat-0.9.20/toolbox/vl_setup')
imageDir = 'test_images';
load('my_svm.mat');
imageList = dir(fullfile(imageDir, '/*.jpg'));
nImages = length(imageList);

bboxes = zeros(0, 4);
confidences = zeros(0, 1);
image_names = cell(0, 1);

cellSize = 6;
dim = 36;
for i = 1:nImages
	% load and show the image
	im = im2single(imread(fullfile(imageDir, imageList(i).name)));
	imshow(im);
	hold on;

	% generate a grid of features across the entire image. you may want to 
	% try generating features more densely (i.e., not in a grid)
	feats = vl_hog(im, cellSize);

	% concatenate the features into 6x6 bins, and classify them (as if they
	% represent 36x36-pixel faces)
	[rows, cols, ~] = size(feats);
	confs = zeros(rows, cols);
	for r = 1:rows - 8
		for c = 1:cols - 8
		% create feature vector for the current window and classify it using the SVM model, 
		% take dot product between feature vector and w and add b,
		% store the result in the matrix of confidence scores confs(r,c)
		feat_range = [r r + 8 c c + 8];
		feat = feats(feat_range(1):feat_range(2), feat_range(3):feat_range(4), :);
		feat = feat(:)';
		conf = feat * w + b;
		confs(r, c) = conf;
		end
	end

	% get the most confident predictions 
	[~, inds] = sort(confs(:), 'descend');
	inds = inds(1: 20); % (use a bigger number for better recall)
	for n = 1:numel(inds)
		[row, col] = ind2sub([size(feats, 1) size(feats, 2)], inds(n));

		bbox = [col * cellSize ...
				row * cellSize ...
				(col + cellSize - 1) * cellSize ...
				(row + cellSize - 1) * cellSize];
		conf = confs(row, col);
		image_name = {imageList(i).name};

		% plot
		plot_rectangle = [bbox(1), bbox(2); ...
			bbox(1), bbox(4); ...
			bbox(3), bbox(4); ...
			bbox(3), bbox(2); ...
			bbox(1), bbox(2)];
		plot(plot_rectangle(:, 1), plot_rectangle(:, 2), 'g-');

		% save
		bboxes = [bboxes; bbox];
		confidences = [confidences; conf];
		image_names = [image_names; image_name];
	end
	pause;
	fprintf('got preds for image %d/%d\n', i, nImages);
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
	evaluate_detections(bboxes, confidences, image_names, label_path);
