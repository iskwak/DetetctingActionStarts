inrootdir = '/tier2/hantman/Jay/videos';
outrootdir = '/media/drive1/data/hantman';
behaviornamesuffix = 'm134w';
nframespad = 5;
savefile = '~/test.jab';

jd = load('-mat','/media/drive1/data/hantman/M134C3VGATXChR2_anno/Final JAB/M134w_20150507_manual.jab');
jd = jd.x;
ld = load('/media/drive1/data/hantman/ToneVsLaserData20150717.mat');

jdnew = jd;
jdnew.expDirNames = {};
jdnew.labels = jdnew.labels(1);
behaviorfns = {'Lift','Handopen','Grab','Atmouth','Sup','Chew'};
behaviornames = cellfun(@(x) [x,behaviornamesuffix],behaviorfns,'Uni',0);
nobehaviornames = cellfun(@(x) ['No_',x,behaviornamesuffix],behaviorfns,'Uni',0);

for i = 1:numel(ld.rawdata),
  expdir = ld.rawdata(i).expfull;
  expname = expdir(numel(inrootdir)+2:end);
  fp = regexp(expname,'/','split');
  outexpdir = fullfile(outrootdir,[fp{1},'_anno'],fp{2:end});
  jdnew.expDirNames{i} = outexpdir;
  newlabels = jdnew.labels(1);
  newlabels.t0s = cell(1,1);
  newlabels.t1s = cell(1,1);
  newlabels.names = cell(1,1);
  newlabels.names{1} = {};
  newlabels.timestamp = cell(1,1);
  nframes = numel(ld.trxdata(i).x1);
  newlabels.imp_t0s = {1};
  newlabels.imp_t1s = {nframes-1};
  for j = 1:numel(behaviorfns),
    behaviorfn = behaviorfns{j};
    behaviorname = behaviornames{j};
    nobehaviorname = nobehaviornames{j};
    lablname = [behaviorfn,'_labl_t0sPos'];
    isbehavior = false(1,nframes-1);
    isbehavior(ld.rawdata(i).(lablname)) = true;
    isnobehavior = isbehavior;
    isnobehavior = imfilter(isnobehavior,ones(1,2*nframespad+1),'same');
    isnobehavior = ~isnobehavior;
    [not0s,not1s] = get_interval_ends(isnobehavior);
    for k = 1:numel(ld.rawdata(i).(lablname)),
      newlabels.t0s{1}(end+1) = ld.rawdata(i).(lablname)(k);
      newlabels.t1s{1}(end+1) = ld.rawdata(i).(lablname)(k)+1;
      newlabels.names{1}{end+1} = behaviorname;
      newlabels.timestamp{1}(end+1) = now;
    end
    for k = 1:numel(not0s),
      newlabels.t0s{1}(end+1) = not0s(k);
      newlabels.t1s{1}(end+1) = not1s(k)-1;
      newlabels.names{1}{end+1} = nobehaviorname;
      newlabels.timestamp{1}(end+1) = now;
    end
  end
  jdnew.labels(i) = newlabels;
end
jdnew.expDirTags = cell(numel(jdnew.expDirNames),1);
[jdnew.expDirTags{:}] = deal(cell(1,0));

x = jdnew;
save('-mat',savefile,'x')
