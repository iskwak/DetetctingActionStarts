% loop over each of the jab files in the anno directory.
% for each date, find the one that has No_* labels.

base_dir = '/media/drive1/data/hantman/M134C3VGATXChR2_anno/Final JAB/';
% base_dir = '/media/drive1/data/hantman/M147VGATXChrR2_anno/final JAB/';
% base_dir = '/media/drive1/data/hantman/M174VGATXChR2/Final_JAB/';
jab_fnames = dir(base_dir);


% create a list of jab files to copy over.
jab_list = {};
for i = 1:length(jab_fnames)
    % disp(jab_fnames(i).name);
    % regexp to see if it is a jab file, without a ~ at the end of it.
    contains_name = regexp(jab_fnames(i).name, 'M134w\_.*jab.*(?<!\~)$');
    % contains_name = regexp(jab_fnames(i).name, 'M147w\_.*jab.*(?<!\~)$');
    % contains_name = regexp(jab_fnames(i).name, 'M174\_.*jab.*(?<!\~)$');
    if ~isempty(contains_name)
        disp(jab_fnames(i).name);
        jab = load(fullfile(base_dir, jab_fnames(i).name), '-mat');
        jab = jab.x;

        contains_neg = contains_negative_labels(jab);

        if contains_neg == true
            disp('   contains negative labels');
            jab_list{end + 1} = jab_fnames(i).name;
        end

        % break;
    end
end

fprintf('\n\n');
jab_list{:}
% % loop over the jab files, and see if there are duplicate experiments.
% for i = 1:length(jab_list)
%     idx = regexp(jab_list{i}, '\D+');
%     % the first idx should be MXXX.
%     if length(idx) < 2
%         continue;
%     end
%     rest = jab_list{i}(idx(2) + 1:end);
%     date_str1 = sscanf(rest, '%d%s');
%     date_str1 = date_str1(1);

%     for j = (i + 1):length(jab_list)
%         % disp(j)
%         idx = regexp(jab_list{j}, '\D+');
%         rest = jab_list{j}(idx(2) + 1:end);
%         date_str2 = sscanf(rest, '%d%s');
%         date_str2 = date_str2(1);

%         if date_str1 == date_str2
%             % fprintf('%s -- %s\n', jab_list{i}, jab_list{j});
%             % fprintf('%d -- %d\n', date_str1, date_str2);
%             jab1 = load(fullfile(base_dir, jab_list{i}), '-mat');
%             jab1 = jab1.x;
%             jab2 = load(fullfile(base_dir, jab_list{j}), '-mat');
%             jab2 = jab2.x;

%             if ~isempty(setdiff(jab1.expDirNames, jab2.expDirNames))
%                 moo = setdiff(jab1.expDirNames, jab2.expDirNames);
%                 moo{:}
%             end
%         end
%     end
%     break;
% end