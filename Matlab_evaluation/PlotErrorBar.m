addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);
dsoError = zeros(10, 22);
dsvioError = zeros(10, 22);

for seqNum = 1:22
	dsoError(:, seqNum) = DSO_MAV_FWD{seqNum}.allSegRMSE;
	dsvioError(:, seqNum) = DSVIO_MAV_FWD{seqNum}.allSegRMSE;
end

clipFun = @(x) (x<=1)*x + (x>1)*1;
% clip error to be atmost 0.5
% dsoError = arrayfun(clipFun, dsoError);
% dsvioError = arrayfun(clipFun, dsvioError);

meanDsoError = mean(dsoError);
meanDsvioError = mean(dsvioError);

maxDsoError = max(dsoError);
maxDsvioError = max(dsvioError);

minDsoError = min(dsoError);
minDsvioError = min(dsvioError);

% % --------------------------- clip error to be atmost 0.5 --------------------------- %
meanDsoError = arrayfun(clipFun, meanDsoError);
meanDsvioError = arrayfun(clipFun, meanDsvioError);

maxDsoError = arrayfun(clipFun, maxDsoError);
maxDsvioError = arrayfun(clipFun, maxDsvioError);

minDsoError = arrayfun(clipFun, minDsoError);
minDsvioError = arrayfun(clipFun, minDsvioError);

% calculate the difference from the mean
maxDsoError = maxDsoError - meanDsoError;
maxDsvioError = maxDsvioError - meanDsvioError;

minDsoError = meanDsoError - minDsoError;
minDsvioError = meanDsvioError - minDsvioError;

figure
hold on
grid on
plot1 = errorbar([1:22]-0.1, meanDsoError, minDsoError, maxDsoError, 'bo');
plot2 = errorbar([1:22]+0.1, meanDsvioError, minDsvioError, maxDsvioError, 'r*');
% baseline error (0.5)
% plot3 = plot(0:23, 0.5*ones(1,24), 'k--', 'LineWidth', 2);

sequences = { ...
	'MH01l', 'MH02l', 'MH03l', 'MH04l', 'MH05l', ...
	'V101l', 'V102l', 'V103l', ... 
	'V201l', 'V202l', 'V203l', ... 
	'MH01r', 'MH02r', 'MH03r', 'MH04r', 'MH05r', ...
	'V101r', 'V102r', 'V103r', ... 
	'V201r', 'V202r', 'V203r', ... 
};
set(gca, 'XTick', 1:22)
set(gca, 'XTickLabel', sequences)
xlabel('Sequence')
ylabel('ATE RMSE (m)')
legend([plot1, plot2], {'DSO', 'DSVIO'}, 'FontSize', 30)
title('EuRoC RMSE trajectory error')




