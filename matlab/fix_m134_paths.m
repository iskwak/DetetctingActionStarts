jab = loadAnonymous('/media/drive1/data/hantman/M134C3VGATXChR2_anno/Final JAB/M134wmanual20141203to20150326.jab');

% loop over the experiment directories and fix the directories
for i = 1:numel(jab.expDirNames)
    % [temp1, search_dir, temp2] = fileparts(jab.expDirNames{i});
    % chop up the windows path.
    temp_path = jab.expDirNames{i}(3:end);
    temp_path = strrep(temp_path, '\', '/');
    [temp1, search_dir, temp2] = fileparts(temp_path);

    % disp(search_dir);
    fprintf('%s\n', search_dir);
    found_path = find_paths(search_dir, '/media/drive1/data/hantman');
    if strcmp(found_path, '') ~= 1
        % disp(found_path)
        % fprintf('\t%s\n', found_path);
        jab.expDirNames{i} = found_path;
    end
end

% saveAnonymous('/localhome/kwaki/theano-env/jabfiles/M134w_mod.jab', jab);