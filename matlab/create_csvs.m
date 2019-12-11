% create csv files
% header:
% frame,<behav>,<behav gt>,image
combined = loadAnonymous('/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab');
data_mat = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');
gtdata = data_mat.rawdata;

base_dir = '/localhome/kwaki/outputs/test_gt_comp/';

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

w = gausswin(19);

i = 1;

for i = 1:length(exp_names),
    % check to see if the bout labeled data has labels.
    if isempty(combined.labels(i).t0s) || isempty(combined.labels(i).t0s{1})
        continue;
    end
    % once the labels are known to exist. Make sure positive labels exist.
    combined_haslabels = false;
    for j = 1:length(labels),
        if any(strcmp(labels{j}, combined.labels(i).names{1})),
            combined_haslabels = true;
            break;
        end
    end
    % make sure that there positive labels.
    if combined_haslabels == false,
        continue;
    end

    % check to see if the gt data has any labels
    gt_idx = find(strcmp(exp_names{i}, gt_exps));
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

    % both have labels, create output space
    outdir = fullfile(base_dir, exp_names{i});
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    copy_templates(outdir);
    % create a symbolic link to the frames
    system(['ln -s /media/drive1/data/hantman_frames/', exp_names{i}, '/frames ', ...
            outdir, '/frames']);
    
    num_frames = combined.labels(i).imp_t1s{1};
    for j = 1:length(labels),
        % get the behavior
        cvs_data = zeros(num_frames, 2);

        % apply the predictions
        label_idx = find(strcmp(labels{j}, combined.labels(i).names{:}));
        cvs_data(combined.labels(i).t0s{1}(label_idx), 1) = 1;

        cvs_data(gtdata(gt_idx).(label_fields{j}), 2) = 1;

        % filter the data
        cvs_data = filter(w, 1, cvs_data);

        outname = fullfile(outdir, ['predict_', headers{j}, '.csv']);
        cvs_file = fopen(outname, 'w');
        % write the header
        fprintf(cvs_file, ['frame,', headers{j}, ',', headers{j}, ' ground truth,image\n']);
        for k = 1:size(cvs_data, 1),
            fprintf(cvs_file, '%d,%f,%f,frames/%05d.jpg\n', k, cvs_data(k, 1), cvs_data(k, 2), k);
        end
        fclose(cvs_file);
    end
end