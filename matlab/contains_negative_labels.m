function contains_neg = contains_negative_labels(jab)

% loop over the names to see if there are No_ labels.
% length(jab.labels);
contains_neg = false;
for j = 1:length(jab.labels)
    if ~isempty(jab.labels(j).names)
        if any(~cellfun(@isempty, strfind(jab.labels(j).names{1}, 'No_')));
            % jab.labels(j).names{:}
            contains_neg = true;
            break;
        end
    end
end