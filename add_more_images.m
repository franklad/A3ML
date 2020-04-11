function i = add_more_images(dir_name)
imageDir = dir_name;
imageList = dir(fullfile(imageDir, '*.jpg'));
nImages = length(imageList);

for i = 1:nImages
	img_path = imageList(i).name;
	img = imread(fullfile(imageDir, img_path));
	[~, name, ext] = fileparts(img_path);
	
	for k = 1:3
		im_lr = rot90(img, k);
		new_file = sprintf('%s{rot90-%d}%s', name, 90 * k, ext);
		imwrite(im_lr, fullfile(imageDir, new_file));
	end

	img_comp = imcomplement(img);
	
	for k = 1:4
		im_lr = rot90(img_comp, k);
		new_file = sprintf('%s{rot90-comp-%d}%s', name, 90 * k, ext);
		imwrite(im_lr, fullfile(imageDir, new_file));
	end

	img_noise = imnoise(img,'gaussian', 0.001);
	
	for k = 1:4
		im_lr = rot90(img_noise, k);
		new_file = sprintf('%s{rot90-noisy-%d}%s', name, 90 * k, ext);
		imwrite(im_lr, fullfile(imageDir, new_file));
	end

	img_noise_comp = imcomplement(img_noise);
	
	for k = 1:4
		im_lr = rot90(img_noise_comp, k);
		new_file = sprintf('%s{rot90-comp-noisy-%d}%s', name, 90 * k, ext);
		imwrite(im_lr, fullfile(imageDir, new_file));
	end
end
end