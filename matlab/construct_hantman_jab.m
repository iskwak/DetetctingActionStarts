% add the JAABA path.
jlabelpath = '/localhome/kwaki/theano-env/checkouts/JAABA/perframe';

addpath(jlabelpath);
baseDir = fileparts(jlabelpath);
addpath(fullfile(baseDir,'misc'));
addpath(fullfile(baseDir,'filehandling'));
addpath(fullfile(jlabelpath,'larva_compute_perframe_features'));
addpath(fullfile(jlabelpath,'compute_perframe_features'));
addpath(fullfile(baseDir,'perframe','params'));
addpath(fullfile(baseDir,'tests'));


% % create an hantman lab mouse Macguffin.
% macguffin = Macguffin('adam_mice');
% macguffin.setMainBehaviorName('Lift');
% macguffin.setScoreFileName({'scores_Lift.mat'});
% macguffin.setTrxFileName('trx.mat');
% macguffin.setMovieFileName('movie_comb.avi');

% data=JLabelData('setstatusfn',@(str)(fprintf('%s\n',str)), ...
%                 'clearstatusfn',@()(nop()), ...
%                 'isInteractive',false);

% data.newJabFile(macguffin);


% % get the experiments from the file list.
% % exp_filelist = '/media/drive1/data/hantman_processed/hdf5_data/full_paths.txt';
% exp_filelist = '/media/drive1/data/hantman_processed/hdf5_data/full_paths.txt';
% exp_names = fileread(exp_filelist);
% exp_names = strsplit(exp_names, char(10));
% % remove the last element
% exp_names(end) = [];

% for i = 1:length(exp_names)
%     data.AddExpDir(exp_names{i});
% end

% data.saveJabFile('test.jab');

% jab = loadAnonymous('lift.jab');
% 
% % load the labels
% label_mat = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');
% 
% % loop over the labels and add them to the list
% num_exp = length(jab.labels);
% for i = 1:num_exp
%     num_bouts = length(label_mat.rawdata(i).Lift_t0s);
%     num_flies = length(jab.labels(i).flies);
%     if num_flies == 0
%         disp(num2str(i));
%         continue
%     end
%     % jab.labels(i).t0s{1} = [];
%     % jab.labels(i).t1s{1} = [];
%     for j = 1:num_bouts
%         jab.labels(i).t0s{1}(end+1) = label_mat.rawdata(i).Lift_t0s(j);
%         jab.labels(i).t1s{1}(end+1) = label_mat.rawdata(i).Lift_t0s(j);
%     end
%     % jab.labels(i).flies(end+1) = 1;
% end
% 
% saveAnonymous('test.jab', jab);
% 
% % check /media/drive1/data/hantman/<..stuff..>/Final\ Jab
