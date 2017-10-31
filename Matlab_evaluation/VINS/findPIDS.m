function pids = findPIDS(processName)
	pids = [];
	output = evalc(['system(''ps -C '  processName ' '')']);
	lines = strsplit(output, '\n');
	for line = lines
		words = strsplit(line{1}, ' ');
		if length(words{1})
			[pid, status] = str2num(words{1});
			if status
				pids = [pids pid];
			end
		end
	end
end