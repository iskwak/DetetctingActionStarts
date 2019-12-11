function copy_exp_files2(input_dir, output_dir)
% copy_exp_files  copy files within a jaaba experiment.

% features.mat
% perframes
% trx.mat

mkdir(fullfile(output_dir, 'perframe'));
copyfile(fullfile(input_dir, 'perframe'), fullfile(output_dir, 'perframe'));
copyfile(fullfile(input_dir, 'features.mat'), output_dir);
copyfile(fullfile(input_dir, 'trx.mat'), output_dir);
% copyfile(fullfile(input_dir, 'movie_comb.avi'), output_dir);
system(['ln -s ', fullfile(input_dir, 'movie_comb.avi'), ' ', output_dir]);