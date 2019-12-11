jab = loadAnonymous('/nrs/branson/kwaki/jab_experiments/M173VGATXChR2_multiday/M173_20150512_b.jab');

% combined_idx = find(strcmp(postproc_exps{i}, exp_names));
% combined_haslabels = false;
% if ~isempty(combined_idx),
%     % does the experiment exist?
%     if ~isempty(combined.labels(combined_idx).names),
%         % does it have labels?
%         for j = 1:length(labels),
%             if any(strcmp(labels{j}, combined.labels(combined_idx).names{1})),
%                 combined_haslabels = true;
%                 break;
%             end
%         end
%         if combined_haslabels == true,
%             continue;
%         end
%     end
pos_labels = { ...
    'Liftm173', ...
    'Handopenm173', ...
    'Grabm173', ...
    'Supm173', ...
    'Atmouthm173', ...
    'Chewm173'
};

label_counts = [0, 0, 0, 0, 0, 0];
num_labeled = zeros(length(jab.labels), 1);
for i = 1:length(jab.labels)
    labels = jab.labels(i);
    if isempty(labels.names)
        continue;
    end
    label_names = labels.names{1};
    for j = 1:length(label_names)
        disp(label_names{j});
        if any(strcmp(label_names{j}, pos_labels))
            idx = find(strcmp(label_names{j}, pos_labels));
            start_idx = labels.t0s{1}(j);
            end_idx = labels.t1s{1}(j);
            num_labeled(i) = num_labeled(i) + (end_idx - start_idx);
            label_counts(idx) = label_counts(idx) + (end_idx - start_idx);
        end
    end
end
num_labeled
label_counts