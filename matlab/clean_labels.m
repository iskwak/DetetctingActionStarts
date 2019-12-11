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
    '/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150504/M134w_20150504.jab', ...
    % '/nrs/branson/kwaki/jab_exp_3/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp_3/M174_20150408.jab'
    % '/nrs/branson/kwaki/jab_exp/M134w_20150427.jab', ...
    % '/nrs/branson/kwaki/jab_exp/M174_20150416.jab'
    % '/nrs/branson/kwaki/jab_exp/M134w_20150504.jab'
};
output_file = '/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150504/combined.jab';


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

% next merge the fields that need merging
for i = 2:length(jab_files)
    jab2 = load(jab_files{i}, '-mat');
    jab2 = jab2.x;

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

saveAnonymous(output_file, jab1)