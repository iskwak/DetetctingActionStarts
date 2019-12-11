% augment combined.jab
combined = loadAnonymous('/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab');
data_mat = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');
gtdata = data_mat.rawdata;

exp2day = regexp(combined.expDirNames,'/M\d+_(\d{8})','tokens','once');
exp2day = [exp2day{:}];
num_exp = numel(combined.labels);
exp_names = regexp(combined.expDirNames,'(M\d+_\w+)', 'tokens', 'once');
exp_names = [exp_names{:}];
labels = {'Lift', 'Handopen', 'Grab', 'Sup', 'Atmouth', 'Chew'};
headers = {'lift', 'hand', 'grab', 'suppinate', 'mouth', 'chew'};


gt_mice = regexp({gtdata.expdir},'(M\d+)_\d+', 'tokens', 'once');
gt_mice = [gt_mice{:}];
gt_days = regexp({gtdata.expdir},'M\d+_(\d+)', 'tokens', 'once');
gt_days = [gt_days{:}];
gt_exps = regexp({gtdata.expdir},'(M\d+_\w+)', 'tokens', 'once');
gt_exps = [gt_exps{:}];

% m134_idx = find(strcmp('M134', gt_mice));
datamat_exps = {};
datamat_days = {};
prev_len = 0;
label_fields = {'Lift_labl_t0sPos', 'Handopen_labl_t0sPos', 'Grab_labl_t0sPos', 'Sup_labl_t0sPos', 'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos'};

empty_label = combined.labels(100);
% loop over gt data, if an experiment has gt labels and doesn't exist in
% combined, add it to the experiment list... and is an M134 exp.
for i = 1:length(gt_exps),
    if strcmp(gt_mice{i}, 'M134') ~= 1
        continue
    end
    
    % check to see if the gt data has any labels
    gt_idx = i;
    gt_haslabels = false;
    if isempty(gt_idx),
        % if this experiment isn't in the gt data, continue.
        continue;
    end
    % if it is, see if the label fields has anything.
    for j = 1:length(label_fields),
        if ~isempty(gtdata(gt_idx).(label_fields{j})),
            gt_haslabels = true;
            break;
        end
    end
    if gt_haslabels == false,
        continue
    end
    
    % check to see if the bout labeled data has labels. if it does, don't
    % use these set (trained with gt info).
    combined_idx = find(strcmp(gt_exps{i}, exp_names));
    if any(combined_idx)
        continue
    end    
    disp(gt_exps{i});
    
    % augment!
    % add expDirNames, expDirTags, labels entries
    % get directory name, move it and add it to the combined list.

    base_exp_dir = '/media/drive1/data/hantman/M134C3VGATXChR2_anno/';
    output_dir = '/nrs/branson/kwaki/M134C3VGATXChR2_anno/';
    found_path = find_paths(gt_exps{i}, base_exp_dir);
    if strcmp(found_path, '') ~= 1
        % disp(found_path)
        fprintf('\t%s\n', found_path);
        % copy to the output space... skip score mat files?
        % create the output space.
        exp_out_dir = fullfile(output_dir, gt_exps{i});
        if exist(exp_out_dir, 'dir')
            % keyboard
        end
        mkdir(exp_out_dir);
        copy_exp_files(found_path, exp_out_dir)

        combined.labels(end+1) = empty_label;
        combined.expDirTags{end+1} = combined.expDirTags{end};
        combined.expDirNames{end+1} = exp_out_dir;
    end

end
out_name = '/nrs/branson/kwaki/M134C3VGATXChR2_anno/augmented.jab';
saveAnonymous(out_name, combined);