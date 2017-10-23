addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);
dsoError = zeros(10, 22);
dsvioError = zeros(10, 22);

for seqNum = 1:22
	dsoError(:, seqNum) = DSO_MAV_FWD{seqNum}.allSegRMSE(:, :);
	dsvioError(:, seqNum) = DSVIO_MAV_FWD{seqNum}.allSegRMSE(:, :);
end

figure
hold on
grid on
errorbar([1:22]-0.1, mean(dsoError), min(dsoError), max(dsoError), 'bo');
errorbar([1:22]+0.1, mean(dsvioError), min(dsvioError), max(dsvioError), 'r*');

sequences = { ...
	'MH\_01\_l', 'MH\_02\_l', 'MH\_03\_l', 'MH\_04\_l', 'MH\_05\_l', ...
	'V1\_01\_l', 'V1\_02\_l', 'V1\_03\_l', ... 
	'V2\_01\_l', 'V2\_02\_l', 'V2\_03\_l', ... 
	'MH\_01\_r', 'MH\_02\_r', 'MH\_03\_r', 'MH\_04\_r', 'MH\_05\_r', ...
	'V1\_01\_r', 'V1\_02\_r', 'V1\_03\_r', ... 
	'V2\_01\_r', 'V2\_02\_r', 'V2\_03\_r', ... 
};
set(gca, 'XTick', 1:22)
set(gca, 'XTickLabel', sequences)
xlabel('Time (s)')
ylabel('ATE (m)')
legend('DSO', 'DSVIO')
title('EuRoC RMSE trajectory error')

