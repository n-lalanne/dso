LOG_DIR = '/home/rakesh/workspace/dso/Matlab_evaluation/logs';
files = dir(LOG_DIR);

for fileIdx = 1:size(files, 1)
	filename = files(fileIdx).name;
	split = strsplit(filename, '_');
	if (strcmp(split(1), 'mav') == 1)
		split(1) = strcat(split(1), '2');
		newFilename = strjoin(split, '_');
		system(['cp ' LOG_DIR '/' filename ' ' LOG_DIR '/' newFilename]);
	end
end
