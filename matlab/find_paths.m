function found_path = find_paths(search_path, base_path)

    listing = dir(base_path);
    % remove the first two directories
    listing = listing(3:end);
    sub_dirs = listing(vertcat(listing.isdir));

    % first see if this level has the desired directory.
    if any(strcmp(search_path, {sub_dirs.name}))
        % return the reconstructed path
        found_path = fullfile(base_path, search_path);
        return
    end

    % otherwise... DFS
    for i = 1:numel(sub_dirs)
        % reconstruct the path
        sub_path = fullfile(base_path, sub_dirs(i).name);
        ret = find_paths(search_path, sub_path);
        if strcmp(ret, '') ~= 1
            % must have found the path, bail
            found_path = ret;
            return
        end
    end

    % if we get this far, then this tree was not fruitful
    found_path = '';
    return
end
