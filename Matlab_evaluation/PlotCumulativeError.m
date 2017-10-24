addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);

seqNum = 8;
dsoIter = 1;
dsvioIter = 1;

intervalDuration = 10;

dsoError = DSO_MAV_FWD{seqNum}.allSegErrorCumulative(dsoIter,:);
dsvioError = DSVIO_MAV_FWD{seqNum}.allSegErrorCumulative(dsvioIter,:);

% remove the ones where the error has gone to zero
extra_idx = min([find(dsoError == 0), find(dsvioError == 0)]);
if (extra_idx)
	dsoError = dsoError(1:(extra_idx-1));
	dsvioError = dsvioError(1:(extra_idx-1));
end

figure
hold on
grid on

plot(intervalDuration:intervalDuration:(length(dsoError)*intervalDuration),		dsoError,	'b-o');
plot(intervalDuration:intervalDuration:(length(dsvioError)*intervalDuration),	dsvioError, 'r-*');

set(gca, 'XTick', intervalDuration:intervalDuration:(length(dsoError))*intervalDuration)
xlabel('Time (s)')
ylabel('ATE (m)')
legend('DSO', 'DSVIO')
title('EuRoC MAV V1\_03')