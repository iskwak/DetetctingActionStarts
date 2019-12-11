function [diffs, s1, s2] = jab_diff(j1, j2)
% empty structs to grow
s1 = struct();
s2 = struct();
diffs = {};

[diffs, s1, s2] = jab_diff_helper(j1, j2, diffs, s1, s2);
end


function [diffs, s1, s2] = jab_diff_helper(j1, j2, diffs, s1, s2)

fields1 = fieldnames(j1);
fields2 = fieldnames(j2);

% first check field name differences.
if ~isempty(setdiff(fields1, fields2))
    keyboard
end

% loop over the fields, and potentially recursively call the helper.
for i_field = 1:length(fields1)
    subj1 = j1.(fields1{i_field});
    subj2 = j2.(fields1{i_field});

    subtype = get_type(subj1);
    switch(subtype)
        case 'struct'
            if ~isempty(subj1) && ~isempty(subj2)
                [diffs, s1, s2] = jab_diff_helper(subj1, subj2, diffs, s1, s2);
            else
                diffs{end + 1} = fields1{i_field};
                s1.(fields1{i_field}) = subj1;
                s2.(fields1{i_field}) = subj2;
            end
        case 'cell'
            diffs{end + 1} = fields1{i_field};
            s1.(fields1{i_field}) = subj1;
            s2.(fields1{i_field}) = subj2;
        case 'cellstring'
            temp_diffs = setdiff(subj1, subj2);
            if ~isempty(temp_diffs)
                diffs{end + 1} = fields1{i_field};
                s1.(fields1{i_field}) = subj1;
                s2.(fields1{i_field}) = subj2;
            end
        case 'numeric'
            diffs{end + 1} = fields1{i_field};
            s1.(fields1{i_field}) = subj1;
            s2.(fields1{i_field}) = subj2;
    end
end

end

function valtype = get_type(val)

valtype = class(val);

if strcmp(valtype, 'Macguffin')
    valtype = 'struct';
elseif strcmp(valtype, 'cell')
    if ~isempty(val) && strcmp(class(val{1}), 'char')
        valtype = 'cellstring';
    end
elseif isnumeric(val)
    valtype = 'numeric';
end

end
