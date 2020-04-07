% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(fullfile(imageDir, '*.jpg'));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

while n_have < n_want
	% fetch a random image
	rng_img_idx = randi(nImages);
	img_path = imageList(rng_img_idx).name;
	img = rgb2gray(imread(fullfile(imageDir, img_path)));
	[~, name, ext] = fileparts(img_path);
	
	% generate a random 36x36 crop from the non-face image
	[h, w] = size(img);
	rx = randi(h - dim);
	ry = randi(w - dim);
	cropped_image = imcrop(img, [ry rx dim - 1 dim - 1]);
	
	% write the image to file
	new_file = sprintf('%s{%d}%s', name, n_have, ext);
	imwrite(cropped_image, fullfile(new_imageDir, new_file));
	n_have = numel(dir(fullfile(new_imageDir, '*.jpg')));
end
