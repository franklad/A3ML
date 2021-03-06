pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(fullfile(pos_imageDir, '*.jpg'));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(fullfile(neg_imageDir, '*.jpg'));
neg_nImages = length(neg_imageList);

% split images_faces to 80% training and 20% validation
pos_imageDir_v = 'cropped_validation_images_faces';
if ~exist(pos_imageDir_v, 'dir')
	mkdir(pos_imageDir_v);
	pos_nValid = round(pos_nImages * 0.2);
	pos_imageList = pos_imageList(randperm(length(pos_imageList)), :);
	
	for i = 1:pos_nValid
		name = pos_imageList(i).name;
		src = fullfile(pos_imageDir, name);
		movefile(src, pos_imageDir_v);
	end
end

% split images_notfaces to 80% training and 20% validation
neg_imageDir_v = 'cropped_validation_images_notfaces';
if ~exist(neg_imageDir_v, 'dir')
	mkdir(neg_imageDir_v);
	neg_nValid = round(neg_nImages * 0.2);
	neg_imageList = neg_imageList(randperm(length(neg_imageList)), :);

	for i = 1:neg_nValid
		name = neg_imageList(i).name;
		src = fullfile(neg_imageDir, name);
		movefile(src, neg_imageDir_v);
	end
end