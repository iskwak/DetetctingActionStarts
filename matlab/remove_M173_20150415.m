% for some reason the M173 jab files all have M173_20150415 with no labels.
% for now just remove all M173_20150415 experiments.
jab_files = { ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150416.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150417.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150420.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150423.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150424.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150501.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150504.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150505.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150506.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150511.jab', ...
    '/groups/branson/home/patilr/hantman_mdays/M173_20150512_b.jab'
%     '/nrs/branson/kwaki/jab_experiments2/M259_20180426.jab'
};

% next merge the fields that need merging
for i = 1:length(jab_files)
    disp(jab_files{i});
    jab = load(jab_files{i}, '-mat');
    jab = jab.x;
    names = strfind(jab.expDirNames, 'M173_20150415');
    bad_dates = ~cellfun(@isempty, names);
    % sum(bad_dates)
    jab.labels(bad_dates) = [];
    jab.expDirNames(bad_dates) = [];
    jab.expDirTags(bad_dates) = [];

    saveAnonymous(jab_files{i}, jab);
end