% helper to blank out exisiting labels to make a "test" set.
jab_files = { ...
    '/nrs/branson/kwaki/jab_experiments2/M259_20180426.jab'
};

for i = 1:length(jab_files)
    jab = load(jab_files{i}, '-mat');
    jab = jab.x;

    all_names = jab.expDirNames;
    unique_names = unique(all_names);
    % loop over the names and find the dupe.
    for j = 1:length(unique_names)
        idx = find(strcmp(unique_names{j}, all_names));
        if length(idx) > 1
            disp(unique_names{j});
            disp(num2str(length(idx)));
            % clear out the dupe
            jab.labels(idx(1)) = [];
            jab.expDirNames(idx(1)) = [];
            jab.expDirTags(idx(1)) = [];
        end
    end
    
    % save out the experiment
    tempname = [jab_files{i}(1:end-4), '_clean.jab'];
    saveAnonymous(tempname, jab);
end
