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