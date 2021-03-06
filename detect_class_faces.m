%run('../vlfeat-0.9.20/toolbox/vl_setup')
imageDir = 'test_images';
load('my_svm.mat');

bboxes = zeros(0, 4);
confidences = zeros(0, 1);
image_names = cell(0, 1);

cellSize_factors = [7, 8, 11, 16];
offset = 8;
scale_factors = [1 1 1];
% load and show the image
im_name = 'class.jpg';
im = im2single(imread(im_name));
close all
imshow(im);
hold on;

conf_vals = [];
s = 1;
for j = 1:3
	% generate a grid of features across the entire image. you may want to 
	% try generating features more densely (i.e., not in a grid)
	cellSize = cellSize_factors(j);
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
					(c / s + cellSize - 1) * cellSize ...
					(r / s + cellSize - 1) * cellSize ...
					cellSize];
		end
	end
	if isempty(confs)
		continue
	end
	[~, idx] = sort(confs(:, 1), 'descend');
	confs = confs(idx, :);
	confs = NMS(confs, 0);
	conf_len = min([+inf size(confs, 1)]);
	confs = confs(1:conf_len, :);
	conf_vals = [conf_vals; confs];
end
% get the most confident predictions 
conf_vals(:, 1) = conf_vals(:, 1) .* conf_vals(:, 6);
[~, idx] = sort(conf_vals(:, 1), 'descend');
conf_vals = conf_vals(idx, :);
max_recall = 130;
max_recall = min([size(conf_vals, 1) max_recall]);
conf_vals = conf_vals(1:max_recall, :);
conf_vals = NMS(conf_vals, 0.01);
for n = 1:size(conf_vals, 1)
	conf = conf_vals(n, 1);
		bbox = [conf_vals(n, 2) ...
			conf_vals(n, 3) ...
			conf_vals(n, 4) ...
			conf_vals(n, 5)];
	image_name = {im_name};

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