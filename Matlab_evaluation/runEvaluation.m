function [] = runEvaluation(executable, config, datasetDir)
    datasets = { ...
        struct( ...
            'name', 'MH_01_easy', ...
            'startFrame', 950, ...
            'endFrame', 3600 ...
        ), ...
        struct( ...
            'name', 'MH_02_easy', ...
            'startFrame', 800, ...
            'endFrame', 3000 ...
        ), ...
        struct( ...
            'name', 'MH_03_medium', ...
            'startFrame', 410, ...
            'endFrame', 2600 ...
        ), ...
        struct( ...
            'name', 'MH_04_difficult', ...
            'startFrame', 445, ...
            'endFrame', 1925 ...
        ), ...
        struct( ...
            'name', 'MH_05_difficult', ...
            'startFrame', 460, ...
            'endFrame', 2200 ...
        ), ...
        struct( ...
            'name', 'V1_01_easy', ...
            'startFrame', 0, ...
            'endFrame', 2800 ...
        % ), ...
        struct( ...
            'name', 'V1_02_medium', ...
            'startFrame', 115, ...
            'endFrame', 1600 ...
        ), ...
        struct( ...
            'name', 'V1_03_difficult', ...
            'startFrame', 250, ...
            'endFrame', 2020 ...
        ), ...
        struct( ...
            'name', 'V2_01_easy', ...
            'startFrame', 0, ...
            'endFrame', 2130 ...
        ), ...
        struct( ...
            'name', 'V2_02_medium', ...
            'startFrame', 100, ...
            'endFrame', 2230 ...
        ), ...
        struct( ...
            'name', 'V2_03_difficult', ...
            'startFrame', 115, ...
            'endFrame', 1880 ...
        ), ...
    };

    for datasetIdx = 1:size(datasets, 2)
        for iterIdx = 0:9
            while true
                arguments = [ ...
                    [' calib='  datasetDir '/camera.txt'], ...
                    [' config=' config ' '], ...
                    [' bag=' datasetDir '/' datasets{datasetIdx}.name '/' datasets{datasetIdx}.name '.bag'], ...
                    [' groundtruth=' datasetDir '/' datasets{datasetIdx}.name '/mav0/state_groundtruth_estimate0/data.csv'], ...
                    [' output=' 'mav_' datasets{datasetIdx}.name '_' num2str(iterIdx) '.txt'], ...
                    [' start_frame=' num2str(datasets{datasetIdx}.startFrame)], ...
                    [' end_frame=' num2str(datasets{datasetIdx}.endFrame)], ...
                    [' nogui=1']
                ];
                % [executable arguments]
                % return
                [status, cmdout] = system([executable arguments], '-echo')

                if status == 0
                    break;
                else
                    fileId = fopen('logs/failure.txt', 'a');
                    fprintf(fileId, [datasets{datasetIdx}.name '\n']);
                    fclose(fileId);
                end
            end
        end
    end
end