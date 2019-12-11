
jabfile = '/localhome/kwaki/theano-env/jabfiles/test_one_day.jab';

jd = loadAnonymous(jabfile);

% first test, just delete labels most of the behaviors.

num_labels = length(jd.labels);
for i = 1:(num_labels - 1)
    disp(jd.expDirNames{i})
    jd.labels(i).t0s{1} = jd.labels(i).t0s{1}(1:3);
    jd.labels(i).t1s{1} = jd.labels(i).t1s{1}(1:3);
    jd.labels(i).names{1} = jd.labels(i).names{1}(1:3);
    jd.labels(i).timestamp{1} = jd.labels(i).timestamp{1}(1:3);
end

jd.labels(num_labels).t0s{1} = [];
jd.labels(num_labels).t1s{1} = [];
jd.labels(num_labels).names{1} = [];
jd.labels(num_labels).timestamp{1} = [];

saveAnonymous('/localhome/kwaki/theano-env/jabfiles/test_save.jab', jd);
%saveAnonymous('/groups/branson/home/kwaki/test_save.jab', jd);
% jd is the Macguffin()? yes

% % test train?
% jd = loadAnonymous('/localhome/kwaki/theano-env/jabfiles/test.jab');
% 
% groundTruthingMode = false;
% data = JLabelData('setstatusfn',@fprintf_wrapper,'clearstatusfn',@() fprintf('Done.\n'));
% data.openJabFile(jabfile,groundTruthingMode);
% data.StoreLabelsAndPreLoadWindowData();
% [success,msg] = data.PreLoadPeriLabelWindowData();
% assert(success,msg);
% data.Train();
% 
% % test classify?
% classifyMovie(data.expdirs{1}, data.getMacguffin(), 'verbose', 1);
