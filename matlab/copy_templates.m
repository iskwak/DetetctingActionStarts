function copy_templates(outdir)
    template_dir = '/groups/branson/home/kwaki/checkouts/QuackNN/templates';

    copyfile(fullfile(template_dir, 'require.js'), outdir);
    copyfile(fullfile(template_dir, 'predict_main.js'), outdir);
    % loop over the predict_* files
    predict_names = {'chew', 'grab', 'hand', 'lift', 'mouth', 'suppinate'};
    for i = 1:length(predict_names),
        copyfile(fullfile(template_dir, ['predict_', predict_names{i}, '.html']), ...
                 outdir);
    end
end