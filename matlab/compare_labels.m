data_mat = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');
% compare to combined.jab
combined = loadAnonymous('/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab');

% load up the stats?
postproc = load('/nrs/branson/kwaki/M134C3VGATXChR2_anno/data.mat');

% loop over the combined jab first?


% get the exp days for combined
exp2day = regexp(combined.expDirNames,'/M\d+_(\d{8})','tokens','once');
exp2day = [exp2day{:}];
num_exp = numel(combined.labels);
exps_names = regexp(combined.expDirNames,'(M\d+_\w+)', 'tokens', 'once');
labelled_exps = {};
labelled_days = {};
for i = 1:num_exp,
    if isempty(combined.labels(i).t0s),
        continue;
    end
    % labelled_exps(end+1) = combined.expDirNames(i); %#ok<SAGROW>
    labelled_exps(end+1) = exps_names{i}; %#ok<SAGROW>
    % parse the day out
    labelled_days(end+1) = exp2day(i); %#ok<SAGROW>
end


% get the exp days for combined
all_mice = regexp({data_mat.rawdata.expdir},'(M\d+)_\d+', 'tokens', 'once');
all_days = regexp({data_mat.rawdata.expdir},'M\d+_(\d+)', 'tokens', 'once');
all_exps = regexp({data_mat.rawdata.expdir},'(M\d+_\w+)', 'tokens', 'once');

m134_idx = find(strcmp('M134', [all_mice{:}]));
datamat_exps = {};
datamat_days = {};
prev_len = 0;
label_fields = {'Lift_labl_t0sPos', 'Handopen_labl_t0sPos', 'Grab_labl_t0sPos', 'Sup_labl_t0sPos', 'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos'};
for i = 1:length(m134_idx),
    idx = m134_idx(i);
%     if isempty(data_mat.rawdata.labels(idx).t0s),
%         continue;
%     end
    for j = 1:length(label_fields),
        if ~isempty(data_mat.rawdata(idx).(label_fields{j})),
            datamat_exps(end+1) = all_exps{idx}; %#ok<SAGROW>
            % parse the day out
            datamat_days(end+1) = all_days{idx}; %#ok<SAGROW>
            break;
        end
    end
%     if prev_len == length(datamat_exps),
%         1;
%     end
    prev_len = length(datamat_exps);
end



% get the exp days for combined
allpost_mice = regexp({postproc.data.exp},'(M\d+)_\d+', 'tokens', 'once');
allpost_days = regexp({postproc.data.exp},'M\d+_(\d+)', 'tokens', 'once');
allpost_exps = regexp({postproc.data.exp},'(M\d+_\w+)', 'tokens', 'once');

m134_idx = find(strcmp('M134', [allpost_mice{:}]));
post_exps = {};
post_days = {};
for i = 1:length(m134_idx),
    idx = m134_idx(i);
%     if isempty(data_mat.rawdata.labels(idx).t0s),
%         continue;
%     end
    post_exps(end+1) = allpost_exps{idx}; %#ok<SAGROW>
    % parse the day out
    post_days(end+1) = allpost_days{idx}; %#ok<SAGROW>
end



% nbhs = numel(jd.behaviors.names);
% nexps = numel(jd.expDirNames);

% framecounts = zeros(nbhs,nexps);
% boutcounts = zeros(nbhs,nexps);

  
% for i = 1:nexps,
%   if isempty(jd.labels(i).t0s),
%     continue;
%   end
%   [~,bhidx] = ismember(jd.labels(i).names{1},jd.behaviors.names);
%   for j = 1:numel(jd.labels(i).t0s{1}),
%     bhi = bhidx(j);
%     framecounts(bhi,i) = framecounts(bhi,i) + jd.labels(i).t1s{1}(j)-jd.labels(i).t0s{1}(j);
%     boutcounts(bhi,i) = boutcounts(bhi,i) + 1;
%   end
% end

% exp2day = regexp(jd.expDirNames,'/M\d+_(\d{8})','tokens','once');
% exp2day = [exp2day{:}];
% [uniquedays,~,exp2dayidx] = unique(exp2day);
% for i = 1:numel(uniquedays),
%   islabeledcurr = any(boutcounts(:,exp2dayidx==i),1);
%   nlabeledtrials = nnz(islabeledcurr);
%   fprintf('\n%s: %d labeled trials / %d trials\n',uniquedays{i},nlabeledtrials,nnz(exp2dayidx==i));
  
%   for bhi = 1:nbhs,
    
%     fprintf('%s, %s: %d bouts labeled\n',uniquedays{i},jd.behaviors.names{bhi},sum(boutcounts(bhi,exp2dayidx==i)));
    
%   end
% end

% jabfiles = {'/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab'};
% expdirs = jd.expDirNames(70);