
jabfile = '/groups/branson/home/kwaki/test2.jab';

jd = loadAnonymous(jabfile);
exps = jd.expDirNames;

for exps_i = exps
    exps_i = exps_i{1}; %#ok<FXSET>
    disp(exps_i);
    % setup a backup space
    if ~exist(fullfile(exps_i, 'backup'), 'file')
        mkdir(fullfile(exps_i, 'backup'))
    end

    % get the experiments in the directory.
    score_files = dir(fullfile(exps_i, 'scores*'));
    for score_files_i = {score_files.name}
        score_name = score_files_i{1};
        % disp(score_name)
        copyfile(fullfile(exps_i, score_name), ...
                fullfile(exps_i, 'backup', score_name))
    end
    % disp('hi')
end
