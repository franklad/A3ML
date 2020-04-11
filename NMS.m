function conf_out = NMS(conf_in, thresh)
	conf_out = [];
	NSM_thresh = 0.15;
	if nargin == 2
		NSM_thresh = thresh;
	end
	pad = 1;
	while ~isempty(conf_in)
		conf = conf_in(1, :);
		box = conf(2:5);
		box_area = (box(3) - box(1) + pad) * (box(4) - box(2) + pad);
		bi = [max(box(1), conf_in(:, 2)) ...
			 max(box(2), conf_in(:, 3)) ...
			 min(box(3), conf_in(:, 4)) ...
			 min(box(4), conf_in(:, 5))];
		iw = bi(:, 3) - bi(:, 1) + pad;
		iw(iw < 0) = 0;
		ih = bi(:, 4) - bi(:, 2) + pad;
		ih(ih < 0) = 0;
		int_area = iw .* ih;
		iarea = (conf_in(:, 4) - conf_in(:, 2) + pad) .*...
				(conf_in(:, 5) - conf_in(:, 3) + pad);
		unn_area = iarea + box_area - int_area;
		iou = int_area ./ unn_area;
		idxs_to_remove = iou > NSM_thresh;
		conf_in(idxs_to_remove, :) = [];
		conf_out = [conf_out; conf];
	end
end