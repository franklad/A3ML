%run('../vlfeat-0.9.20/toolbox/vl_setup')
load('pos_neg_feats.mat')
clear w b
lambda = 0.01;
folds = 5;

pos_nImages_fold = round(pos_nImages * 1 / folds);
neg_nImages_fold = round(neg_nImages * 1 / folds);

pos_feats = pos_feats(randperm(length(pos_feats)), :);
neg_feats = neg_feats(randperm(length(neg_feats)), :);

for i = 1:folds
	fold_range = [i - 1 i];
	pos_range = [(fold_range(1) * pos_nImages_fold) + 1 fold_range(2) * pos_nImages_fold - 1];
	neg_range = [(fold_range(1) * neg_nImages_fold) + 1 fold_range(2) * neg_nImages_fold - 1];
	
	pos_feats_fold = pos_feats(pos_range(1) : pos_range(2), :);
	neg_feats_fold = neg_feats(neg_range(1) : neg_range(2), :);
	
	feats = cat(1, pos_feats_fold, neg_feats_fold);
	labels = cat(1, ones(pos_nImages_fold - 1, 1), -1 * ones(neg_nImages_fold - 1, 1));
	
	[w, b] = vl_svmtrain(feats', labels', lambda);
	if i == 1
		w_avg = w;
		b_avg = b;
	else
		b_avg = (b + b_avg) / 2;
		w_avg = (w + w_avg) / 2;
	end
	
end

save('my_svm.mat', 'w', 'b');
fprintf('Classifier performance on train data:\n')
confidences = [pos_feats_fold; neg_feats_fold] * w + b;
[tp_rate, fp_rate, tn_rate, fn_rate] = report_accuracy(confidences, labels);
