

jabfile = '/groups/branson/home/kwaki/test3.jab';
% rootdatadir0 = 'C:\Documents\DATA';
% rootdatadir = '/tier2/branson/FlyBubble_testing/Steffi_antgrmlines/withPPF_hoghof';

fracframes = logspace(-2,0,20);

%% change file locations

jd = loadAnonymous(jabfile);

groundTruthingMode = false;
data = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
data.openJabFile(jabfile,groundTruthingMode);
data.StoreLabelsAndPreLoadWindowData();
[success,msg] = data.PreLoadPeriLabelWindowData();
assert(success,msg);
moo = data.Train();

% classifyMovie

% %% precompute whatever we can, training
% 
% iCls = 1;
% islabeled = data.windowdata(iCls).labelidx_new~=0 & data.windowdata(iCls).labelidx_imp;
% idxlabeled = find(islabeled);
% cls2IdxBeh = data.iCls2iLbl;
% % form label vec that has 1 for 'behavior present'
% labelidxnew = data.windowdata(iCls).labelidx_new(islabeled);
% assert(numel(cls2IdxBeh{iCls})==2);
% valBeh = cls2IdxBeh{iCls}(1);
% valNoBeh = cls2IdxBeh{iCls}(2);
% assert(all(labelidxnew==valBeh | labelidxnew==valNoBeh));
% labels12 = 2*ones(size(labelidxnew));
% labels12(labelidxnew==valBeh) = 1;
% bins = findThresholdBins(data.windowdata(iCls).X(islabeled,:),...
%   data.windowdata(iCls).binVals);
% 
% 
% ntrain = 0;
% for expi = 1:numel(data.labels),
%   for flyi = 1:size(data.labels(expi).flies,1),
%     flies = data.labels(expi).flies(flyi,:);
%     labelidx = GetLabelIdx(data,expi,flies);
%     ntrain = ntrain + nnz(labelidx.vals);
%   end
% end
% assert(ntrain == nnz(islabeled));
% 
% %% check for overlap
% 
% for expi = 1:numel(gtdata.labels),
%   
%   expdir = gtdata.expdirs{expi};
%   expj = find(strcmp(data.expdirs,expdir));
%   if isempty(expj),
%     continue;
%   end
%   for flyi = 1:numel(gtdata.labels(expi)),
%     flies = gtdata.labels(expi).flies(flyi,:);
%     gtlabelidx = GetLabelIdx(gtdata,expi,flies);
%     labelidx = GetLabelIdx(data,expj,flies);
%     assert(gtlabelidx.T0==labelidx.T0 && gtlabelidx.T1==labelidx.T1);
%     noverlap = nnz(gtlabelidx.vals>0 & labelidx.vals>0);
%     if noverlap > 0,
%       fprintf('Exp: %d, %s, fly: %d, noverlap = %d\n',expi,expdir,flies,noverlap);
%     end
%   end    
%   
% end
% 
% %% precompute whatever we can, gt
% 
% gtislabeled = gtdata.windowdata(iCls).labelidx_new~=0;
% gt_labels = 2*gtdata.windowdata(iCls).labelidx_new(gtislabeled) + ...
%   - gtdata.windowdata(iCls).labelidx_imp(gtislabeled);
% 
% ntest = 0;
% for expi = 1:numel(gtdata.labels),
%   for flyi = 1:size(gtdata.labels(expi).flies,1),
%     flies = gtdata.labels(expi).flies(flyi,:);
%     labelidx = GetLabelIdx(gtdata,expi,flies);
%     ntest = ntest + nnz(labelidx.vals);
%     assert(nnz(labelidx.vals)==nnz(gtdata.windowdata.exp==expi & gtdata.windowdata.flies==flies));
%   end
% end
% assert(ntest == nnz(gtislabeled));
%       
% %% figure out which window features are st, which are traj
% pff = cellfun(@(x) x{1},data.windowdata.featurenames,'Uni',0);
% isstfeat = regexp(pff,'^(hs|hf)','once');
% isstfeat = ~cellfun(@isempty,isstfeat);
% 
% gtpff = cellfun(@(x) x{1},gtdata.windowdata.featurenames,'Uni',0);
% assert(all(strcmp(pff,gtpff)));
% 
% 
% %% break into bouts
% 
% label_timestamp = data.GetLabelTimestamps(data.windowdata.exp(islabeled),data.windowdata.flies(islabeled,:),data.windowdata.t(islabeled));
% [unique_timestamps,firstidx,boutidx] = unique(label_timestamp);
% bout_labels = labels12(firstidx);
% ntrain_bouts = numel(unique_timestamps);
% % choose limits on frames based on fracframes
% nframes_train_total = nnz(islabeled);
% frame_lims = max(1,round(fracframes*nframes_train_total));
% nframes_per_bout = nan(1,ntrain_bouts);
% for i = 1:ntrain_bouts,
%   nframes_per_bout(i) = nnz(label_timestamp == unique_timestamps(i));
% end
% nbouts = numel(nframes_per_bout);
% nboutspos = nnz(bout_labels==1);
% nboutsneg = nnz(bout_labels==2);
% boutidxpos = find(bout_labels==1);
% boutidxneg = find(bout_labels==2);
% boutidxshort = find(nframes_per_bout <= median(nframes_per_bout));
% 
% %% choose which frames to train on for each subset of the training data
% 
% nretrain = 10;
% 
% trainframeorder = nan(numel(label_timestamp),nretrain);
% trainframeorder(:,1) = 1:numel(label_timestamp);
% 
% for trainiteri = 2:nretrain,
%   
%   % random permutation of bouts
%   % train on bout order(i) ith
%   
%   cansamplepos = false(1,nbouts);
%   cansamplepos(boutidxpos) = true;
%   cansampleneg = false(1,nbouts);
%   cansampleneg(boutidxneg) = true;
%   
%   order = nan(1,nbouts);
%   nposcurr = 0;
%   nnegcurr = 0;
%   for i = 1:nbouts,
%     idxposcurr = find(cansamplepos);
%     idxnegcurr = find(cansampleneg);
%     if i == 1,
%       order(i) = randsample(boutidxshort,1);
%     else
%       if ~any(cansampleneg),
%         order(i) = idxposcurr(randsample(numel(idxposcurr),1));
%       elseif ~any(cansamplepos)
%         order(i) = idxnegcurr(randsample(numel(idxnegcurr),1));
%       elseif nposcurr / i >= nboutspos / nbouts,
%         order(i) = idxnegcurr(randsample(numel(idxnegcurr),1));
%       else
%         order(i) = idxposcurr(randsample(numel(idxposcurr),1));
%       end
%     end
% 
%     if i > 1,
%       assert(~any(order(i)==order(1:i-1)));
%     end
%     
%     cansamplepos(order(i)) = false;
%     cansampleneg(order(i)) = false;
% 
%     if bout_labels(order(i)) == 1,
%       nposcurr = nposcurr + 1;
%     else
%       nnegcurr = nnegcurr + 1;
%     end
%     
%   end
%   
%   % trainframeorder(i,trainiteri) is the ith frame to train on 
%   off = 0;
%   for i = 1:nbouts,
%     idxcurr = find(boutidx==order(i));
%     trainframeorder(off+1:off+numel(idxcurr),trainiteri) = idxcurr;
%     off = off + numel(idxcurr);
%   end
%   
% end
% 
% %% train on increasing amounts of training data
% 
% niters = numel(fracframes);
% 
% tmp = struct('nlabelpos_scorepos',nan(niters,nretrain),...
%   'nlabelpos_scoreneg',nan(niters,nretrain),...
%   'nlabelneg_scorepos',nan(niters,nretrain),...
%   'nlabelneg_scoreneg',nan(niters,nretrain));
% tmp2 = struct;
% tmp2.imp = tmp;
% tmp2.notimp = tmp;
% clear tmp;
% crosserror = struct;
% featuretypes = {'st','traj','both'};
% for i = 1:numel(featuretypes),
%   crosserror.(featuretypes{i}) = tmp2;
% end
% 
% nframestrain = frame_lims;
% colors = lines(3);
% 
% for i = 1:niters,
%   
%   for shufflei = 1:nretrain,
%     
%     dotraincurr = false(1,numel(label_timestamp));
%     dotraincurr(trainframeorder(1:frame_lims(i),shufflei)) = true;
%     
%     assert(nnz(dotraincurr)==nframestrain(i));
%     
%     %   bouti = bouti_lims(i);
%     %   if bouti == ntrain_bouts,
%     %     dotraincurr = true(size(label_timestamp));
%     %   else
%     %     dotraincurr = label_timestamp < unique_timestamps(bouti+1);
%     %   end
%     fprintf('Iteration %d / %d, training on %d / %d frames, shuffle %d...\n',i,niters,nframestrain(i),nframes_train_total,shufflei);
%     
%     pstr = sprintf('\nTraining %s classifier from %d examples...',data.labelnames{iCls},nnz(dotraincurr));
%     
%     % check for presence of both positive and negative labels
%     npos = nnz(labels12(dotraincurr)==1);
%     nneg = nnz(labels12(dotraincurr)~=1);
%     if npos < 1 || nneg < 1
%       warning('Classifier %s: Only behavior or nones have been labeled. Not training classifier.',...
%         data.labelnames{iCls});
%       continue;
%     end
%     
%     [classifier_st,~,trainstats_st] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),isstfeat), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals(:,isstfeat),...
%       bins(isstfeat,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     [classifier_traj,~,trainstats_traj] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),~isstfeat), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals(:,~isstfeat),...
%       bins(~isstfeat,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     [classifier_both,~,trainstats_both] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),:), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals,...
%       bins(:,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     scores = struct;
%     scores.st = myBoostClassify(gtdata.windowdata.X(gtislabeled,isstfeat),classifier_st);
%     scores.traj = myBoostClassify(gtdata.windowdata.X(gtislabeled,~isstfeat),classifier_traj);
%     scores.both = myBoostClassify(gtdata.windowdata.X(gtislabeled,:),classifier_both);
%     
%     % label key:
%     % 1 = positive important
%     % 2 = positive notimportant
%     % 3 = negative important
%     % 4 = negative notimportant
%     
%     clf;
%     h = nan(1,numel(featuretypes));
%     for j = 1:numel(featuretypes),
%       fn = featuretypes{j};
%       crosserror.(fn).imp.nlabelpos_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 1);
%       crosserror.(fn).imp.nlabelpos_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 1);
%       crosserror.(fn).imp.nlabelneg_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 3);
%       crosserror.(fn).imp.nlabelneg_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 3);
%       
%       crosserror.(fn).notimp.nlabelpos_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 2);
%       crosserror.(fn).notimp.nlabelpos_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 2);
%       crosserror.(fn).notimp.nlabelneg_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 4);
%       crosserror.(fn).notimp.nlabelneg_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 4);
%       
%       balacc = ((crosserror.(fn).imp.nlabelpos_scorepos ./ ...
%         (crosserror.(fn).imp.nlabelpos_scorepos + crosserror.(fn).imp.nlabelpos_scoreneg)) + ...
%         (crosserror.(fn).imp.nlabelneg_scoreneg ./ ...
%         (crosserror.(fn).imp.nlabelneg_scoreneg + crosserror.(fn).imp.nlabelneg_scorepos)))/2;
%       
%       hcurr = plot(nframestrain,balacc,'o-','Color',colors(j,:));
%       h(j) = hcurr(1);
%       hold on;
%       
%     end
%     legend(h,featuretypes);
%     axisalmosttight;
%     
%   end
% 
% end
%   
% save TrajVsVideoFeatures_AntennalGrooming_WithStd.mat nframestrain crosserror trainframeorder;
% 
% 
% %% plot results
% 
% hfig = 1;
% figure(hfig);
% clf;
% balacc_imp_st_trials = ((crosserror.st.imp.nlabelpos_scorepos ./ ...
%   (crosserror.st.imp.nlabelpos_scorepos + crosserror.st.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.st.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.st.imp.nlabelneg_scoreneg + crosserror.st.imp.nlabelneg_scorepos)))/2;
% balacc_imp_traj_trials = ((crosserror.traj.imp.nlabelpos_scorepos ./ ...
%   (crosserror.traj.imp.nlabelpos_scorepos + crosserror.traj.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.traj.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.traj.imp.nlabelneg_scoreneg + crosserror.traj.imp.nlabelneg_scorepos)))/2;
% balacc_imp_both_trials = ((crosserror.both.imp.nlabelpos_scorepos ./ ...
%   (crosserror.both.imp.nlabelpos_scorepos + crosserror.both.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.both.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.both.imp.nlabelneg_scoreneg + crosserror.both.imp.nlabelneg_scorepos)))/2;
% 
% balacc_imp_st_mean = nanmean(balacc_imp_st_trials,2);
% balacc_imp_traj_mean = nanmean(balacc_imp_traj_trials,2);
% balacc_imp_both_mean = nanmean(balacc_imp_both_trials,2);
% balacc_imp_st_stderr = nanstd(balacc_imp_st_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_st_trials),2));
% balacc_imp_traj_stderr = nanstd(balacc_imp_traj_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_traj_trials),2));
% balacc_imp_both_stderr = nanstd(balacc_imp_both_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_both_trials),2));
% 
% off = 1;
% %miny = min([min(balacc_imp_st),min(balacc_imp_traj),min(balacc_imp_both)]);
% xlim = [nframestrain(max(1,off-1))*.8,nframestrain(end)*1.2];
% %ylim = [miny*.95,1];
% ylim = [0.4,1];
% 
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_st_mean(off:end)-balacc_imp_st_stderr(off:end);...
%   flipud(balacc_imp_st_mean(off:end)+balacc_imp_st_stderr(off:end))],...
%   [0,.8,.8]*.5+.5,'EdgeColor','none');
% hold on;
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_traj_mean(off:end)-balacc_imp_traj_stderr(off:end);...
%   flipud(balacc_imp_traj_mean(off:end)+balacc_imp_traj_stderr(off:end))],...
%   [.8,0,0]*.5+.5,'EdgeColor','none');
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_both_mean(off:end)-balacc_imp_both_stderr(off:end);...
%   flipud(balacc_imp_both_mean(off:end)+balacc_imp_both_stderr(off:end))],...
%   [0,0,0]*.5+.5,'EdgeColor','none');
% 
% h = nan(1,3);
% h(1) = plot(nframestrain(off:end),balacc_imp_st_mean(off:end),'o-','Color',[0,.8,.8],'MarkerFaceColor',[0,.8,.8]);
% hold on;
% plot(nframestrain(off:end),balacc_imp_st_mean(off:end),'wo');
% h(2) = plot(nframestrain(off:end),balacc_imp_traj_mean(off:end),'o-','Color',[.8,0,0],'MarkerFaceColor',[.8,0,0]);
% plot(nframestrain(off:end),balacc_imp_traj_mean(off:end),'wo');
% h(3) = plot(nframestrain(off:end),balacc_imp_both_mean(off:end),'o-','Color',[0,0,0],'MarkerFaceColor',[0,0,0]);
% plot(nframestrain(off:end),balacc_imp_both_mean(off:end),'wo');
% 
% 
% legend(h,{'Video-based','Trajectory-based','Both'},'Location','NorthWest');
% box off;
% set(gca,'XScale','log','XLim',xlim,'YLim',ylim);
% xlabel('Number of training examples');
% ylabel('Classifier accuracy');
% 
% %SaveFigLotsOfWays(hfig,'TrajVsVideoFeatures_AntennalGrooming_WithStd_YLinear',{'pdf','fig','png'});
% 
% 
% %% chase classifiers
% 
% %% prepare jab file
% 
% jabfile = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/CompareTrajAndSTFeatures/Chase.jab';
% labeledjabfile = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/CompareTrajAndSTFeatures/ChaseWithLabels_Aso.jab';
% gtlabeledjabfile = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/CompareTrajAndSTFeatures/ChaseWithGTLabels_Aso.jab';
% gtsuggestfilename = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/usability/GTSuggestions.txt';
% groundTruthingMode = false;
% data = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
% data.openJabFile(jabfile,groundTruthingMode);
% 
% % load in all the labels from one user
% loadlabeldir = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/usability/Aso_Yoshinori_Chase_20120515T172429';
% rootexpdir = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/experiments/CompareTrajAndSTFeatures';
% 
% labelfiles = dir(fullfile(loadlabeldir,'*_labeledChases.mat'));
% gtlabelfiles = dir(fullfile(loadlabeldir,'*_labeledChases_gt.mat'));
% 
% labelexpnames = regexp({labelfiles.name},'^(.*)_labeledChases.mat','tokens','once');
% labelexpnames = [labelexpnames{:}];
% gtlabelexpnames = regexp({gtlabelfiles.name},'^(.*)_labeledChases_gt.mat','tokens','once');
% gtlabelexpnames = [gtlabelexpnames{:}];
% 
% [ism,dataexpi] = ismember(labelexpnames,data.expnames);
% assert(all(ism));
% [ism,gtdataexpi] = ismember(gtlabelexpnames,data.expnames);
% assert(all(ism));
% gtexpdirs = data.expdirs(gtdataexpi);
% 
% for i = 1:numel(labelfiles),
%   
%   ld = load(fullfile(loadlabeldir,labelfiles(i).name));
%   if isempty(ld.flies),
%     continue;
%   end
%   expi = dataexpi(i);
%   data.loadLabelsFromStructForOneExp(expi,ld);
% 
% end
% 
% data.saveJabFile(labeledjabfile);
% 
% % % trained within JAABA GUI to make sure labels looked right
% % input('Train classifier!!');
% % keyboard;
% 
% groundTruthingMode = true;
% gtdata = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
% gtdata.openJabFile(labeledjabfile,groundTruthingMode);
% 
% 
% for i = 1:numel(gtlabelfiles),
%   
%   ld = load(fullfile(loadlabeldir,gtlabelfiles(i).name));
%   if isempty(ld.flies),
%     continue;
%   end
%   
%   gtdata.AddExpDir(gtexpdirs{i});
%   
%   expi = find(strcmp(gtdata.expdirs,gtexpdirs{i}));
%   assert(numel(expi)==1);
%   gtdata.loadLabelsFromStructForOneExp(expi,ld);
% 
% end
% 
% gtdata.saveJabFile(gtlabeledjabfile);
% 
% jabfile = gtlabeledjabfile;
% 
% %% create JLabelData object to train classifier
% 
% groundTruthingMode = false;
% data = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
% data.openJabFile(jabfile,groundTruthingMode);
% data.StoreLabelsAndPreLoadWindowData();
% [success,msg] = data.PreLoadPeriLabelWindowData();
% assert(success,msg);
% data.Train();
% 
% %% create JLabelData object for test data
% 
% groundTruthingMode = true;
% 
% gtdata = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
% gtdata.openJabFile(jabfile,groundTruthingMode);
% gtdata.StoreLabelsAndPreLoadWindowData();
% [success,msg] = gtdata.PreLoadPeriLabelWindowData();
% assert(success,msg);
% 
% 
% %% precompute whatever we can, training
% 
% iCls = 1;
% islabeled = data.windowdata(iCls).labelidx_new~=0 & data.windowdata(iCls).labelidx_imp;
% idxlabeled = find(islabeled);
% cls2IdxBeh = data.iCls2iLbl;
% % form label vec that has 1 for 'behavior present'
% labelidxnew = data.windowdata(iCls).labelidx_new(islabeled);
% assert(numel(cls2IdxBeh{iCls})==2);
% valBeh = cls2IdxBeh{iCls}(1);
% valNoBeh = cls2IdxBeh{iCls}(2);
% assert(all(labelidxnew==valBeh | labelidxnew==valNoBeh));
% labels12 = 2*ones(size(labelidxnew));
% labels12(labelidxnew==valBeh) = 1;
% bins = findThresholdBins(data.windowdata(iCls).X(islabeled,:),...
%   data.windowdata(iCls).binVals);
% 
% 
% ntrain = 0;
% for expi = 1:numel(data.labels),
%   for flyi = 1:size(data.labels(expi).flies,1),
%     flies = data.labels(expi).flies(flyi,:);
%     labelidx = GetLabelIdx(data,expi,flies);
%     ntrain = ntrain + nnz(labelidx.vals);
%   end
% end
% assert(ntrain == nnz(islabeled));
% fprintf('N. train = %d\n',ntrain);
% 
% %% check for overlap
% 
% for expi = 1:numel(gtdata.labels),
%   
%   expdir = gtdata.expdirs{expi};
%   expj = find(strcmp(data.expdirs,expdir));
%   if isempty(expj),
%     fprintf('%s not in training expdirs\n',expdir);
%     continue;
%   end
%   for flyi = 1:numel(gtdata.labels(expi)),
%     flies = gtdata.labels(expi).flies(flyi,:);
%     gtlabelidx = GetLabelIdx(gtdata,expi,flies);
%     labelidx = GetLabelIdx(data,expj,flies);
%     assert(gtlabelidx.T0==labelidx.T0 && gtlabelidx.T1==labelidx.T1);
%     noverlap = nnz(gtlabelidx.vals>0 & labelidx.vals>0);
%     fprintf('Exp: %d, %s, fly: %d, noverlap = %d\n',expi,expdir,flies,noverlap);
%   end    
%   
% end
% 
% %% precompute whatever we can, gt
% 
% gtislabeled = gtdata.windowdata(iCls).labelidx_new~=0;
% gt_labels = 2*gtdata.windowdata(iCls).labelidx_new(gtislabeled) + ...
%   - gtdata.windowdata(iCls).labelidx_imp(gtislabeled);
% 
% ntest = 0;
% for expi = 1:numel(gtdata.labels),
%   for flyi = 1:size(gtdata.labels(expi).flies,1),
%     flies = gtdata.labels(expi).flies(flyi,:);
%     labelidx = GetLabelIdx(gtdata,expi,flies);
%     ntest = ntest + nnz(labelidx.vals);
%     assert(nnz(labelidx.vals)==nnz(gtdata.windowdata.exp==expi & gtdata.windowdata.flies==flies));
%   end
% end
% assert(ntest == nnz(gtislabeled));
%       
% fprintf('N. test important = %d, n. test not important = %d\n',...
%   nnz(gt_labels==1|gt_labels==3),nnz(gt_labels==2|gt_labels==4));
% 
% 
% %% figure out which window features are st, which are traj
% pff = cellfun(@(x) x{1},data.windowdata.featurenames,'Uni',0);
% isstfeat = regexp(pff,'^(hs|hf)','once');
% isstfeat = ~cellfun(@isempty,isstfeat);
% 
% gtpff = cellfun(@(x) x{1},gtdata.windowdata.featurenames,'Uni',0);
% assert(all(strcmp(pff,gtpff)));
% 
% 
% %% break into bouts
% 
% label_timestamp = data.GetLabelTimestamps(data.windowdata.exp(islabeled),data.windowdata.flies(islabeled,:),data.windowdata.t(islabeled));
% [unique_timestamps,firstidx,boutidx] = unique(label_timestamp);
% bout_labels = labels12(firstidx);
% ntrain_bouts = numel(unique_timestamps);
% % choose limits on frames based on fracframes
% nframes_train_total = nnz(islabeled);
% frame_lims = max(1,round(fracframes*nframes_train_total));
% nframes_per_bout = nan(1,ntrain_bouts);
% for i = 1:ntrain_bouts,
%   nframes_per_bout(i) = nnz(label_timestamp == unique_timestamps(i));
% end
% nbouts = numel(nframes_per_bout);
% nboutspos = nnz(bout_labels==1);
% nboutsneg = nnz(bout_labels==2);
% boutidxpos = find(bout_labels==1);
% boutidxneg = find(bout_labels==2);
% boutidxshort = find(nframes_per_bout <= median(nframes_per_bout));
% 
% %% choose which frames to train on for each subset of the training data
% 
% nretrain = 10;
% 
% trainframeorder = nan(numel(label_timestamp),nretrain);
% trainframeorder(:,1) = 1:numel(label_timestamp);
% 
% for trainiteri = 2:nretrain,
%   
%   % random permutation of bouts
%   % train on bout order(i) ith
%   
%   cansamplepos = false(1,nbouts);
%   cansamplepos(boutidxpos) = true;
%   cansampleneg = false(1,nbouts);
%   cansampleneg(boutidxneg) = true;
%   
%   order = nan(1,nbouts);
%   nposcurr = 0;
%   nnegcurr = 0;
%   for i = 1:nbouts,
%     idxposcurr = find(cansamplepos);
%     idxnegcurr = find(cansampleneg);
%     if i == 1,
%       order(i) = randsample(boutidxshort,1);
%     else
%       if ~any(cansampleneg),
%         order(i) = idxposcurr(randsample(numel(idxposcurr),1));
%       elseif ~any(cansamplepos)
%         order(i) = idxnegcurr(randsample(numel(idxnegcurr),1));
%       elseif nposcurr / i >= nboutspos / nbouts,
%         order(i) = idxnegcurr(randsample(numel(idxnegcurr),1));
%       else
%         order(i) = idxposcurr(randsample(numel(idxposcurr),1));
%       end
%     end
% 
%     if i > 1,
%       assert(~any(order(i)==order(1:i-1)));
%     end
%     
%     cansamplepos(order(i)) = false;
%     cansampleneg(order(i)) = false;
% 
%     if bout_labels(order(i)) == 1,
%       nposcurr = nposcurr + 1;
%     else
%       nnegcurr = nnegcurr + 1;
%     end
%     
%   end
%   
%   % trainframeorder(i,trainiteri) is the ith frame to train on 
%   off = 0;
%   for i = 1:nbouts,
%     idxcurr = find(boutidx==order(i));
%     trainframeorder(off+1:off+numel(idxcurr),trainiteri) = idxcurr;
%     off = off + numel(idxcurr);
%   end
%   
% end
% 
% %% train on increasing amounts of training data
% 
% niters = numel(fracframes);
% 
% tmp = struct('nlabelpos_scorepos',nan(niters,nretrain),...
%   'nlabelpos_scoreneg',nan(niters,nretrain),...
%   'nlabelneg_scorepos',nan(niters,nretrain),...
%   'nlabelneg_scoreneg',nan(niters,nretrain));
% tmp2 = struct;
% tmp2.imp = tmp;
% tmp2.notimp = tmp;
% clear tmp;
% crosserror = struct;
% featuretypes = {'st','traj','both'};
% for i = 1:numel(featuretypes),
%   crosserror.(featuretypes{i}) = tmp2;
% end
% 
% nframestrain = frame_lims;
% colors = lines(3);
% 
% for i = 1:niters,
%   
%   for shufflei = 1:nretrain,
%     
%     dotraincurr = false(1,numel(label_timestamp));
%     dotraincurr(trainframeorder(1:frame_lims(i),shufflei)) = true;
%     
%     assert(nnz(dotraincurr)==nframestrain(i));
%     
%     %   bouti = bouti_lims(i);
%     %   if bouti == ntrain_bouts,
%     %     dotraincurr = true(size(label_timestamp));
%     %   else
%     %     dotraincurr = label_timestamp < unique_timestamps(bouti+1);
%     %   end
%     fprintf('Iteration %d / %d, training on %d / %d frames, shuffle %d...\n',i,niters,nframestrain(i),nframes_train_total,shufflei);
%     
%     pstr = sprintf('\nTraining %s classifier from %d examples...',data.labelnames{iCls},nnz(dotraincurr));
%     
%     % check for presence of both positive and negative labels
%     npos = nnz(labels12(dotraincurr)==1);
%     nneg = nnz(labels12(dotraincurr)~=1);
%     if npos < 1 || nneg < 1
%       warning('Classifier %s: Only behavior or nones have been labeled. Not training classifier.',...
%         data.labelnames{iCls});
%       continue;
%     end
%     
%     [classifier_st,~,trainstats_st] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),isstfeat), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals(:,isstfeat),...
%       bins(isstfeat,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     [classifier_traj,~,trainstats_traj] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),~isstfeat), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals(:,~isstfeat),...
%       bins(~isstfeat,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     [classifier_both,~,trainstats_both] =...
%       boostingWrapper(data.windowdata(iCls).X(idxlabeled(dotraincurr),:), ...
%       labels12(dotraincurr),data,...
%       data.windowdata(iCls).binVals,...
%       bins(:,dotraincurr), ...
%       data.classifier_params{iCls},pstr);
%     fprintf('\n');
%     
%     scores = struct;
%     scores.st = myBoostClassify(gtdata.windowdata.X(gtislabeled,isstfeat),classifier_st);
%     scores.traj = myBoostClassify(gtdata.windowdata.X(gtislabeled,~isstfeat),classifier_traj);
%     scores.both = myBoostClassify(gtdata.windowdata.X(gtislabeled,:),classifier_both);
%     
%     % label key:
%     % 1 = positive important
%     % 2 = positive notimportant
%     % 3 = negative important
%     % 4 = negative notimportant
%     
%     clf;
%     h = nan(1,numel(featuretypes));
%     for j = 1:numel(featuretypes),
%       fn = featuretypes{j};
%       crosserror.(fn).imp.nlabelpos_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 1);
%       crosserror.(fn).imp.nlabelpos_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 1);
%       crosserror.(fn).imp.nlabelneg_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 3);
%       crosserror.(fn).imp.nlabelneg_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 3);
%       
%       crosserror.(fn).notimp.nlabelpos_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 2);
%       crosserror.(fn).notimp.nlabelpos_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 2);
%       crosserror.(fn).notimp.nlabelneg_scorepos(i,shufflei) = nnz(scores.(fn) >= 0 & gt_labels == 4);
%       crosserror.(fn).notimp.nlabelneg_scoreneg(i,shufflei) = nnz(scores.(fn) < 0 & gt_labels == 4);
%       
%       balacc = ((crosserror.(fn).imp.nlabelpos_scorepos ./ ...
%         (crosserror.(fn).imp.nlabelpos_scorepos + crosserror.(fn).imp.nlabelpos_scoreneg)) + ...
%         (crosserror.(fn).imp.nlabelneg_scoreneg ./ ...
%         (crosserror.(fn).imp.nlabelneg_scoreneg + crosserror.(fn).imp.nlabelneg_scorepos)))/2;
%       
%       hcurr = plot(nframestrain,balacc,'o-','Color',colors(j,:));
%       h(j) = hcurr(1);
%       hold on;
%       
%     end
%     legend(h,featuretypes);
%     axisalmosttight;
%     
%   end
% 
% end
%   
% save TrajVsVideoFeatures_Chase_WithStd_Aso.mat nframestrain crosserror trainframeorder;
% 
% 
% %% plot results
% 
% hfig = 1;
% figure(hfig);
% clf;
% balacc_imp_st_trials = ((crosserror.st.imp.nlabelpos_scorepos ./ ...
%   (crosserror.st.imp.nlabelpos_scorepos + crosserror.st.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.st.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.st.imp.nlabelneg_scoreneg + crosserror.st.imp.nlabelneg_scorepos)))/2;
% balacc_imp_traj_trials = ((crosserror.traj.imp.nlabelpos_scorepos ./ ...
%   (crosserror.traj.imp.nlabelpos_scorepos + crosserror.traj.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.traj.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.traj.imp.nlabelneg_scoreneg + crosserror.traj.imp.nlabelneg_scorepos)))/2;
% balacc_imp_both_trials = ((crosserror.both.imp.nlabelpos_scorepos ./ ...
%   (crosserror.both.imp.nlabelpos_scorepos + crosserror.both.imp.nlabelpos_scoreneg)) + ...
%   (crosserror.both.imp.nlabelneg_scoreneg ./ ...
%   (crosserror.both.imp.nlabelneg_scoreneg + crosserror.both.imp.nlabelneg_scorepos)))/2;
% 
% balacc_imp_st_mean = nanmean(balacc_imp_st_trials,2);
% balacc_imp_traj_mean = nanmean(balacc_imp_traj_trials,2);
% balacc_imp_both_mean = nanmean(balacc_imp_both_trials,2);
% balacc_imp_st_stderr = nanstd(balacc_imp_st_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_st_trials),2));
% balacc_imp_traj_stderr = nanstd(balacc_imp_traj_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_traj_trials),2));
% balacc_imp_both_stderr = nanstd(balacc_imp_both_trials,1,2) ./ ...
%   sqrt(sum(~isnan(balacc_imp_both_trials),2));
% 
% off = 1;
% %miny = min([min(balacc_imp_st),min(balacc_imp_traj),min(balacc_imp_both)]);
% xlim = [nframestrain(max(1,off-1))*.8,nframestrain(end)*1.2];
% %ylim = [miny*.95,1];
% ylim = [0.4,1];
% 
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_st_mean(off:end)-balacc_imp_st_stderr(off:end);...
%   flipud(balacc_imp_st_mean(off:end)+balacc_imp_st_stderr(off:end))],...
%   [0,.8,.8]*.5+.5,'EdgeColor','none');
% hold on;
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_traj_mean(off:end)-balacc_imp_traj_stderr(off:end);...
%   flipud(balacc_imp_traj_mean(off:end)+balacc_imp_traj_stderr(off:end))],...
%   [.8,0,0]*.5+.5,'EdgeColor','none');
% patch([nframestrain(off:end),nframestrain(end:-1:off)]',...
%   [balacc_imp_both_mean(off:end)-balacc_imp_both_stderr(off:end);...
%   flipud(balacc_imp_both_mean(off:end)+balacc_imp_both_stderr(off:end))],...
%   [0,0,0]*.5+.5,'EdgeColor','none');
% 
% h = nan(1,3);
% h(1) = plot(nframestrain(off:end),balacc_imp_st_mean(off:end),'o-','Color',[0,.8,.8],'MarkerFaceColor',[0,.8,.8]);
% hold on;
% plot(nframestrain(off:end),balacc_imp_st_mean(off:end),'wo');
% h(2) = plot(nframestrain(off:end),balacc_imp_traj_mean(off:end),'o-','Color',[.8,0,0],'MarkerFaceColor',[.8,0,0]);
% plot(nframestrain(off:end),balacc_imp_traj_mean(off:end),'wo');
% h(3) = plot(nframestrain(off:end),balacc_imp_both_mean(off:end),'o-','Color',[0,0,0],'MarkerFaceColor',[0,0,0]);
% plot(nframestrain(off:end),balacc_imp_both_mean(off:end),'wo');
% 
% 
% legend(h,{'Video-based','Trajectory-based','Both'},'Location','NorthWest');
% box off;
% set(gca,'XScale','log','XLim',xlim,'YLim',ylim);
% xlabel('Number of training examples');
% ylabel('Classifier accuracy');
% 
% %SaveFigLotsOfWays(hfig,'TrajVsVideoFeatures_Chase_Aso_WithStd_YLinear',{'pdf','fig','png'});
% 
% 
% %% plot time series for top features
% % 
% % ndimplot = 10;
% % x = data.windowdata.X(:,[classifier_st(1:ndimplot).dim]);
% % ispos = data.windowdata.labelidx_cur == 1;
% % t = data.windowdata.t;
% % t = t + (data.windowdata.flies-1)*max(t);
% % t = t + (data.windowdata.exp-1)*max(t);
% % [sortedt,order] = sort(t);
% % sortedx = x(order,:);
% % sortedispos = ispos(order);
% % sortedlabel_timestamp = label_timestamp(order);
% % clf;
% % hax = createsubplots(ndimplot,1,0);
% % for i = 1:ndimplot,
% %   plot(hax(i),sortedt,sortedx(:,i),'k.');
% %   hold(hax(i),'on');
% %   plot(hax(i),sortedt(sortedispos),sortedx(sortedispos,i),'r.');
% % end
% % 
% % i0s = 1;
% % i1s = [];
% % for i = 2:numel(sortedt),
% %   if sortedt(i) ~= sortedt(i-1)+1,
% %     i1s(end+1) = i-1;
% %     i0s(end+1) = i;
% %   end
% % end
% % i1s(end+1) = numel(sortedt);
% % % 
% % % [~,tmp] = min(sortedlabel_timestamp);
% % % i = find(i0s<=tmp & i1s >= tmp);
% % 
% % nposbouts = nan(1,numel(i1s));
% % nnegbouts = nan(1,numel(i1s));
% % for i = 1:numel(i1s),
% %   i0 = i0s(i);
% %   i1 = i1s(i);
% %   nposbouts(i) = nnz(sortedispos(i0+1:i1)==1 & sortedispos(i0:i1-1)==0) + double(sortedispos(i0));
% %   nnegbouts(i) = nnz(sortedispos(i0+1:i1)==0 & sortedispos(i0:i1-1)==1) + double(sortedispos(i0)==0);
% % end
% % 
% % [~,i] = max(min(nposbouts,nnegbouts));
% % i0 = i0s(i);
% % i1 = i1s(i);
% % figure;
% % clf;
% % hax = createsubplots(ndimplot,1,0);
% % tmpt = sortedt(i0:i1);
% % tmpispos = sortedispos(i0:i1);
% % tmpx = sortedx(i0:i1,:);
% % for d = 1:ndimplot,
% %   plot(hax(d),tmpt,tmpx(:,d),'k.');
% %   hold(hax(d),'on');
% %   plot(hax(d),tmpt(tmpispos),tmpx(tmpispos,d),'r.');
% % end
% % mu = mean(tmpx,1);
% % sig = std(tmpx,1,1);
% % tmpxnorm = bsxfun(@rdivide,bsxfun(@minus,tmpx,mu),sig);
% % 
% % template = mean(sortedx(sortedispos,:),1);
% % templatenorm = (template-mu)./sig;
