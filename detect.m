%run('../vlfeat-0.9.20/toolbox/vl_setup')
imageDir = 'test_images';
load('my_svm.mat');
imageList = dir(fullfile(imageDir, '/*.jpg'));
nImages = length(imageList);

bboxes = zeros(0, 4);
confidences = zeros(0, 1);
image_names = cell(0, 1);

cellSize = 8;
offset = 8;
scale_factors = [0.1 0.3 1.3];
for i = 1:nImages
	% load and show the image
	im = im2single(imread(fullfile(imageDir, imageList(i).name)));
	close all
	imshow(im);
	hold on;

	conf_vals = [];
	for s = scale_factors(1):scale_factors(2):scale_factors(3)
		% generate a grid of features across the entire image. you may want to 
		% try generating features more densely (i.e., not in a grid)
		im_s = imresize(im, s);
		feats = vl_hog(im_s, cellSize);
		% concatenate the features into 6x6 bins, and classify them (as if they
		% represent 36x36-pixel faces)
		[rows, cols, ~] = size(feats);
		confs = [];
		for r = 1:rows - offset
			for c = 1:cols - offset
				% create feature vector for the current window and classify it using the SVM model, 
				% take dot product between feature vector and w and add b,
				% store the result in the matrix of confidence scores confs(r,c)
				feat = feats(r:r + offset, c:c + offset, :);
				conf = feat(:)' * w + b;
				confs = [confs; conf ...
						c / s * cellSize ...
						r / s * cellSize ...
						(c / s + cellSize - 1) * cellSize...
						(r / s + cellSize - 1) * cellSize];
			end
		end
		if isempty(confs)
			continue
		end
		[~, idx] = sort(confs(:, 1), 'descend');
		confs = confs(idx, :);
		confs = NMS(confs);
		conf_len = min([+inf size(confs, 1)]);
		confs = confs(1:conf_len, :);
		conf_vals = [conf_vals; confs];
	end
	% get the most confident predictions 
	[~, idx] = sort(conf_vals(:, 1), 'descend');
	conf_vals = conf_vals(idx, :);
	max_recall = 40;
	max_recall = min([size(conf_vals, 1) max_recall]);
	conf_vals = conf_vals(1:max_recall, :);
	conf_vals = NMS(conf_vals);
	for n = 1:size(conf_vals, 1)
		conf = conf_vals(n, 1);

		bbox = [conf_vals(n, 2) ...
				conf_vals(n, 3) ...
				conf_vals(n, 4) ...
				conf_vals(n, 5)];
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
