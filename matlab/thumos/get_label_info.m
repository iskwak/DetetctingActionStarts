
valid_data = load('/groups/branson/bransonlab/kwaki/data/thumos14/meta/valid/validation_set_meta/validation_set.mat');
valid_data = valid_data.validation_videos;

% valid classes:
base_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/meta/valid/annotation'
files = dir(base_dir);

% skip the first few files '.', '..', 'Ambiguous...'
label_names = {};

all_valid_videos = {};
for i = 4:length(files)
    label_names{end+1} = files(i).name(1:end-6);

    % load the text file and get the label locations
    fname = fullfile(base_dir, files(i).name);
    fid = fopen(fname, 'r');
    tline = fgetl(fid);
    while ischar(tline)
        video_name = strtok(tline);
        all_valid_videos{end+1} = video_name;
        %disp(video_name)
        tline = fgetl(fid);
    end
    fclose(fid);

end

all_valid_videos = unique(all_valid_videos);

% % write the label names to a file
% fid = fopen(fullfile(base_dir, 'labels.txt'), 'w');
% for i = 1:length(label_names)
%     fprintf(fid, '%s\n', label_names{i});
% end


% % check the frame rate for the movies
% % not all are 30, so need to write that info to file.
% meta_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/meta'
% fname = fullfile(meta_dir, 'fps.txt');
% fid = fopen(fname, 'w');
% for i = 1:length(valid_data)
%     if any(strcmp(all_valid_videos, valid_data(i).video_name))
%         fprintf(fid, '%s, %f\n', valid_data(i).video_name, ...
%             valid_data(i).frame_rate_FPS);
%     end
% end
% fclose(fid);


% get frame counts
meta_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/meta'
fname = fullfile(meta_dir, 'fps.txt');
frames = 0;
for i = 1:length(valid_data)
    if any(strcmp(all_valid_videos, valid_data(i).video_name))
        frames = frames + valid_data(i).number_of_frames;
        if valid_data(i).frame_rate_FPS ~= 30
            keyboard
        end
    end
end
