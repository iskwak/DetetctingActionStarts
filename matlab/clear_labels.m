% helper to blank out exisiting labels to make a "test" set.
jab_files = { ...
    '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150417.jab', ...
    '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150501.jab', ...
    '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150504.jab', ...
    '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150511.jab'
};

for i = 1:length(jab_files)
    jab = load(jab_files{i}, '-mat');
    jab = jab.x;
    % for each jab_file, loop through the labels, and empty the values.
    for j = 1:length(jab.labels)
        jab.labels(j).t0s = cell(1, 0);
        jab.labels(j).t1s = cell(1, 0);
        jab.labels(j).names = cell(1, 0);
        jab.labels(j).flies = zeros(0, 1);
        jab.labels(j).off = zeros(1, 0);
        jab.labels(j).timestamp = cell(1, 0);
        jab.labels(j).timelinetimestamp = cell(1, 0);
        jab.labels(j).imp_t0s = cell(1, 0);
        jab.labels(j).imp_t1s = cell(1, 0);
    end

    % save out the experiment
    saveAnonymous(jab_files{i}, jab);
end
