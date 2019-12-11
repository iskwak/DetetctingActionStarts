function jab = fix_paths_jab(jab, root_dir)
% fix_paths_jab  Find and update jab exp dir paths.
%
% inputs
%  jab: The loaded jab file (jab = load(<some jab file>, '-mat'))
%  root_dir: The root directory to search for paths.
%
% outputs;
%  jab: jab file with updated paths.
%

% loop over the experiment directories and fix the directories
for i = 1:numel(jab.expDirNames)
    temp_path = jab.expDirNames{i}(3:end);
    temp_path = strrep(temp_path, '\', '/');
    [~, search_dir, ~] = fileparts(temp_path);

    disp(search_dir)
    found_path = find_paths(search_dir, root_dir);
    if strcmp(found_path, '') ~= 1
        disp(found_path)
        jab.expDirNames{i} = found_path;
    end
end