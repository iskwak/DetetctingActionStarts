% M173_20150415_v001
jab = loadAnonymous('/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150417.jab');

i = 1;

num_labels = length(jab.expDirNames);
while i <= num_labels
    disp(jab.expDirNames{i});
    if ~isempty(strcmp(jab.expDirNames{i}, 'M173_20150415'))
        jab.expDirNames{i} = [];
        i = i - 1;
        num_labels = length(jab.expDirNames);
    end

    i = i + 1;
end