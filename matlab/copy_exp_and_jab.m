% selected jab files.
jab_files = { ...
%     '/media/drive1/data/hantman/M134C3VGATXChR2_anno/Final JAB/20141203to20150326dailySubset/M134w_20150302auto.jab'
%     '/groups/hantman/hantmanlab/from_tier2/Jay/videos/M259VGATXChR2TRN_TH/FinalJAB/M259_20180301.jab'
'/groups/hantman/hantmanlab/from_tier2/Jay/videos/M259VGATXChR2TRN_TH/FinalJAB/M259_20180426.jab'
};


base_exp_dir = '/groups/hantman/hantmanlab/from_tier2/Jay/videos/M259VGATXChR2TRN_TH';
output_dir = '/nrs/branson/kwaki/jab_experiments2';

% loop over the jab file names.
for jab_i = 1:length(jab_files)
    jab = load(jab_files{jab_i}, '-mat');
    jab = jab.x;

    % loop over the experiments, and copy them to the output directory.
    for i = 1:numel(jab.expDirNames)
        % [temp1, search_dir, temp2] = fileparts(jab.expDirNames{i});
        % chop up the windows path.
        temp_path = jab.expDirNames{i}(3:end);
        temp_path = strrep(temp_path, '\', '/');
        [temp1, search_dir, temp2] = fileparts(temp_path);

        % disp(search_dir);
        fprintf('%s\n', search_dir);
        found_path = find_paths(search_dir, base_exp_dir);
        if strcmp(found_path, '') ~= 1
            % disp(found_path)
            fprintf('\t%s\n', found_path);
            % copy to the output space... skip score mat files?
            % create the output space.
            exp_out_dir = fullfile(output_dir, search_dir);
            if exist(exp_out_dir, 'dir')
                % keyboard
            end
            mkdir(exp_out_dir);
            copy_exp_files(found_path, exp_out_dir)
        end
    end

    % fix the paths
    temp = fix_paths_jab(jab, output_dir);
    [~, base_name, ~] = fileparts(jab_files{jab_i});
    out_name = fullfile(output_dir, [base_name, '.jab']);
    saveAnonymous(out_name, temp);
end