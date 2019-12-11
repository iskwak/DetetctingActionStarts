function [all_exps, all_days] = get_labelled_dates(data, inc_unlab)

data_fields = fieldnames(data);
if any(strcmp('expdir', data_fields)),
    exp_field = 'expdir';
else
    exp_field = 'exps';
end

% get the exp days for combined
all_mice = regexp({data.(exp_field)},'(M\d+)_\d+', 'tokens', 'once');
all_days = regexp({data.(exp_field)},'M\d+_(\d+)', 'tokens', 'once');
all_exps = regexp({data.(exp_field)},'(M\d+_\w+)', 'tokens', 'once');

m134_idx = find(strcmp('M134', [all_mice{:}]));
datamat_exps = {};
datamat_days = {};
% prev_len = 0;
label_fields = {'Lift_labl_t0sPos', 'Handopen_labl_t0sPos', 'Grab_labl_t0sPos', 'Sup_labl_t0sPos', 'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos'};
for i = 1:length(m134_idx),
    idx = m134_idx(i);

    % may not want to includ unlabled.
    if inc_unlab == 0
        for j = 1:length(label_fields),
            if ~isempty(data(idx).(label_fields{j})),
                datamat_exps(end+1) = all_exps{idx}; %#ok<AGROW>
                % parse the day out
                datamat_days(end+1) = all_days{idx}; %#ok<AGROW>
                break;
            end
        end
    else
        datamat_exps(end+1) = all_exps{idx}; %#ok<AGROW>
        % parse the day out
        datamat_days(end+1) = all_days{idx}; %#ok<AGROW>
    end
end


end