jd = load('/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab','-mat');
jd = jd.x;
nbhs = numel(jd.behaviors.names);
nexps = numel(jd.expDirNames);

framecounts = zeros(nbhs,nexps);
boutcounts = zeros(nbhs,nexps);

  
for i = 1:nexps,
  if isempty(jd.labels(i).t0s),
    continue;
  end
  [~,bhidx] = ismember(jd.labels(i).names{1},jd.behaviors.names);
  for j = 1:numel(jd.labels(i).t0s{1}),
    bhi = bhidx(j);
    framecounts(bhi,i) = framecounts(bhi,i) + jd.labels(i).t1s{1}(j)-jd.labels(i).t0s{1}(j);
    boutcounts(bhi,i) = boutcounts(bhi,i) + 1;
  end
end

exp2day = regexp(jd.expDirNames,'/M\d+_(\d{8})','tokens','once');
exp2day = [exp2day{:}];
[uniquedays,~,exp2dayidx] = unique(exp2day);
for i = 1:numel(uniquedays),
  islabeledcurr = any(boutcounts(:,exp2dayidx==i),1);
  nlabeledtrials = nnz(islabeledcurr);
  fprintf('\n%s: %d labeled trials / %d trials\n',uniquedays{i},nlabeledtrials,nnz(exp2dayidx==i));
  
  for bhi = 1:nbhs,
    
    fprintf('%s, %s: %d bouts labeled\n',uniquedays{i},jd.behaviors.names{bhi},sum(boutcounts(bhi,exp2dayidx==i)));
    
  end
end

jabfiles = {'/nrs/branson/kwaki/M134C3VGATXChR2_anno/combined.jab'};
expdirs = jd.expDirNames(70);