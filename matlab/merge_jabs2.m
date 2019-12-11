% helper script to merge jab files

% rename:
% scorefilename, names

% merge/append
% names, t0s, t1s, timestamp, timelinestamp, expDirTags

% helper script to merge jab files
% first merge one mouse.
jab_files = { ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150504.jab', ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150505.jab', ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150506.jab', ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150507.jab', ...
    % '/nrs/branson/kwaki/jab_exp_4/M134w_20150522.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150504.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150505.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150506.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150507.jab', ...
    % '/nrs/branson/kwaki/M134C3VGATXChR2_anno/M134w_20150522.jab', ...
    % '/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150427/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp_3/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp_3/M174_20150408.jab'
    % '/nrs/branson/kwaki/jab_exp/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp/M174_20150416.jab'
    % '/nrs/branson/kwaki/jab_exp/M134w_20150504.jab'
    % '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_day1/M173_20150415_manual.jab'
    '/groups/branson/home/patilr/hantman_mdays/M173_20150423.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150416.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150417.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150420.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150424.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150501.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150504.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150505.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150506.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150511.jab', ...
    % 0512 is a dupe of 0511
    % '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_auto_train_rest/M173_20150512.jab', ...
    % '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_auto_day1/M173_20150416.jab', ...
};

% the following fields need to have mouse modifieres changed (remove m134, m174, etc...)
change_values = {
    {'behaviors', 'names'}, ...
    {'file', 'scorefilename'}, ...
    {'labels', 'names'}
};

% the following fields need to be merged.
merged_values = {
    'labels', 'expDirNames', 'expDirTags' ...
};


jab1 = load(jab_files{1}, '-mat');
jab1 = jab1.x;
jab2 = load(jab_files{2}, '-mat');
jab2 = jab2.x;

% % first remove the bad date...
% names = strfind(jab1.expDirNames, 'M173_20150415');
% bad_dates = ~cellfun(@isempty, names);
% jab1.labels(bad_dates) = [];
% jab1.expDirNames(bad_dates) = [];
% jab1.expDirTags(bad_dates) = [];

% clear jab1's labels?
for i = 1:length(jab1.labels)
    jab1.labels(i).t0s = cell(1, 0);
    jab1.labels(i).t1s = cell(1, 0);
    jab1.labels(i).names = cell(1, 0);
    jab1.labels(i).flies = zeros(0, 1);
    jab1.labels(i).off = zeros(1, 0);
    jab1.labels(i).timestamp = cell(1, 0);
    jab1.labels(i).timelinetimestamp = cell(1, 0);
    jab1.labels(i).imp_t0s = cell(1, 0);
    jab1.labels(i).imp_t1s = cell(1, 0);
end

% use jab1 as the basis.

% update jab1.behaviors.names and jab1.file.scorefilename
temp = jab1.behaviors.names;
for i = 1:length(temp)
    digit_idx = regexp(temp{i}, '^\D+', 'end');
    new_name = temp{i}(1:digit_idx);
    if new_name(end) == 'm'
        new_name = new_name(1:end - 1);
    end
    temp{i} = new_name;
end

jab1.behaviors.names = temp;

% do the same for the score file names
temp = jab1.file.scorefilename;
for i = 1:length(temp)
    digit_idx = regexp(temp{i}, '^\D+', 'end');
    new_name = temp{i}(1:digit_idx);
    if new_name(end) == 'm'
        new_name = new_name(1:end - 1);
    end
    new_name = [new_name, '.mat'];
    temp{i} = new_name;
end

jab1.file.scorefilename = temp;

% next merge the fields that need merging
for i = 2:length(jab_files)
    jab2 = load(jab_files{i}, '-mat');
    jab2 = jab2.x;
    names = strfind(jab2.expDirNames, 'M173_20150415');
    bad_dates = ~cellfun(@isempty, names);
    jab2.labels(bad_dates) = [];
    jab2.expDirNames(bad_dates) = [];
    jab2.expDirTags(bad_dates) = [];

    if ~isempty(intersect(jab1.expDirNames, jab2.expDirNames))
        1;
    end

    jab1.labels = [jab1.labels, jab2.labels];
    jab1.expDirNames = [jab1.expDirNames, jab2.expDirNames];
    jab1.expDirTags = [jab1.expDirTags; jab2.expDirTags];
end

% loop over the jab1.labels and fix the label names
for i_label = 1:length(jab1.labels)
    if isempty(jab1.labels(i_label).names)
        continue
    end
    temp = jab1.labels(i_label).names{1};
    for i = 1:length(temp)
        digit_idx = regexp(temp{i}, '^\D+', 'end');
        new_name = temp{i}(1:digit_idx);
        if new_name(end) == 'm'
            new_name = new_name(1:end - 1);
        end
        % new_name = [new_name, '.mat'];
        temp{i} = new_name;
    end
    jab1.labels(i_label).names{1} = temp;

    % rename the struct fields for the timelinetimestamps
    temp_struct = struct();
    struct_fields = fieldnames(jab1.labels(i_label).timelinetimestamp{1});
    for i = 1:length(struct_fields)
        digit_idx = regexp(struct_fields{i}, '^\D+', 'end');
        new_name = struct_fields{i}(1:digit_idx);
        if new_name(end) == 'm'
            new_name = new_name(1:end - 1);
        end
        % new_name = [new_name, '.mat'];
        % struct_fields{i} = new_name;
        temp_struct.(new_name) = ...
            jab1.labels(i_label).timelinetimestamp{1}.(struct_fields{i});
    end
    jab1.labels(i_label).timelinetimestamp{1} = temp_struct;
end

saveAnonymous('/groups/branson/home/patilr/hantman_mdays/combined.jab', jab1)
