function copy_jab_exps(input_dir, output_dir, jab_name)
% COPY_JAB_EXP Coies a jab experiment and the aossciated jab experiments.
% Copies the jab file and the associated esxperiments to a new folder.

% base_exp_dir = '/media/drive1/data/hantman';
% output_dir = '/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_auto_train_rest2';

jab = load(jab_name, '-mat');
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
    found_path = find_paths(search_dir, input_dir);
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
[~, base_name, ~] = fileparts(jab_name);
out_name = fullfile(output_dir, [base_name, '.jab']);
saveAnonymous(out_name, temp);
