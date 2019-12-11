% create csv files
% header:
% frame,<behav>,<behav gt>,image
%combined = loadAnonymous('/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150427/combined.jab');
combined = loadAnonymous('/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_base/combined.jab');

data_mat = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');
gtdata = data_mat.rawdata;
% postproc = load('/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150427/data.mat');
postproc = load('/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_base/data.mat');
postproc = postproc.data;

base_dir = '/localhome/kwaki/outputs/JAABA/M173VGATXChR2_base_test/';

exp2day = regexp(combined.expDirNames,'/M\d+_(\d{8})','tokens','once');
exp2day = [exp2day{:}];
num_exp = numel(combined.labels);
exp_names = regexp(combined.expDirNames,'(M\d+_\w+)', 'tokens', 'once');
exp_names = [exp_names{:}];
% labels = {'Lift', 'Handopen', 'Grab', 'Sup', 'Atmouth', 'Chew'};
labels = {'Lift', 'Handopen', 'Grab'};
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
% label_fields = {'Lift_labl_t0sPos', 'Handopen_labl_t0sPos', 'Grab_labl_t0sPos', 'Sup_labl_t0sPos', 'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos'};
label_fields = {'Lift_labl_t0sPos', 'Handopen_labl_t0sPos', 'Grab_labl_t0sPos'};

w = gausswin(19);


% setup more variables...
postproc_mice = regexp({postproc.exp},'(M\d+)_\d+', 'tokens', 'once');
postproc_mice = [postproc_mice{:}];
postproc_days = regexp({postproc.exp},'M\d+_(\d+)', 'tokens', 'once');
postproc_days = [postproc_days{:}];
postproc_exps = regexp({postproc.exp},'(M\d+_\w+)', 'tokens', 'once');
postproc_exps = [postproc_exps{:}];
% postproc_fields = { ...
%     'auto_Lift_t0s', 'auto_Handopen_t0s', 'auto_Grab_t0s', ...
%     'auto_Sup_t0s', 'auto_Atmouth_t0s', 'auto_Chew_t0s'};
postproc_fields = { ...
    'auto_GS00_Lift_0', 'auto_GS00_Handopen_0', 'auto_GS00_Grab_0'};


for i = 1:length(postproc_exps),
    if isempty(strfind(postproc_exps{i}, 'M173_20150423'))
        continue
    end
    disp(postproc_exps{i});
    % see if there are labels for the post processed data
    postproc_haslabels = false;
    for j = 1:length(postproc_fields),
        if ~isempty(postproc(i).(postproc_fields{j})),
            postproc_haslabels = true;
            break;
        end
    end

    % check to see if the bout labeled data has labels. if it does, don't
    % use these set (trained with gt info).
    combined_idx = find(strcmp(postproc_exps{i}, exp_names));
    combined_haslabels = false;
    if ~isempty(combined_idx),
        % does the experiment exist?
        if ~isempty(combined.labels(combined_idx).names),
            % does it have labels?
            for j = 1:length(labels),
                if any(strcmp(labels{j}, combined.labels(combined_idx).names{1})),
                    combined_haslabels = true;
                    break;
                end
            end
            if combined_haslabels == true,
                continue;
            end
        end
    end
    % check to see if the gt data has any labels
    gt_idx = find(strcmp(postproc_exps{i}, gt_exps));
    gt_haslabels = false;
    if isempty(gt_idx),
        % if this experiment isn't in the gt data, continue.
        continue;
    end
    % if it is, see if the label fields has anything.
    for j = 1:length(postproc_fields),
        if ~isempty(gtdata(gt_idx).(postproc_fields{j})),
            gt_haslabels = true;
            break;
        end
    end
    if gt_haslabels == false,
        continue
    end

    % both have labels, create output space
    outdir = fullfile(base_dir, postproc_exps{i});
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    copy_templates(outdir);
    % create a symbolic link to the frames
    system(['ln -s /media/drive1/data/hantman_frames/', postproc_exps{i}, '/frames ', ...
            outdir, '/frames']);
    
    num_frames = postproc(i).trxt1;
    for j = 1:length(postproc_fields),
        % get the behavior
        cvs_data = zeros(num_frames, 2);

        % apply the predictions
        if ~isnan(postproc(i).(postproc_fields{j}))
            cvs_data(postproc(i).(postproc_fields{j}), 1) = 1;
        end
        if ~isnan(gtdata(gt_idx).(postproc_fields{j}))
            % apply the gt data.
            cvs_data(gtdata(gt_idx).(postproc_fields{j}), 2) = 1;
        end
        %if ~isempty(gtdata(gt_idx).(label_fields{j}))
        %    cvs_data(gtdata(gt_idx).(label_fields{j}), 2) = 1;
        %end

        % filter the data
        % cvs_data = filter(w, 1, cvs_data);
        cvs_data(:, 1) = conv(cvs_data(:, 1), w, 'same');
        cvs_data(:, 2) = conv(cvs_data(:, 2), w, 'same');

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
