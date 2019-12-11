% selected jab files.
jab_files = { ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150416.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150417.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150420.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150423.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150424.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150501.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150504.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150505.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150506.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150511.jab', ...
    '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150512_b.jab'
    % '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150512.jab', ...
    % '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150512.a', ...
    % '/media/drive1/data/hantman/M173VGATXChR2/FinalJab/auto/M173_20150512_a.jab', ...
};
base_exp_dir = '/media/drive1/data/hantman';
%output_dir = '/nrs/branson/kwaki/jab_experiments/M134C3VGATXChR2_20150504';
% output_dir = '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_auto';
% output_dir = '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_base';
output_dir = '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday';

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
