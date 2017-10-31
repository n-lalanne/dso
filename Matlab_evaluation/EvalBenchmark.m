% set this to the supplementary files folder.
FILEPATH = '/home/sicong/dso_bak/supp_v2';

DSO_EVAL_FILEPATH = '/home/sicong/dso/evaluate_ate_scale_matlab';
DSO_FILEPATH = FILEPATH;
DSVIO_FILEPATH = '/home/sicong/dso/build/bin/logs/';

addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);

options=0:9;

%% DSO
logPath_fwd = [DSO_FILEPATH '/DS-VO_Forward/'];
[ DSO_MAV_FWD ] = evalMAVdataset( logPath_fwd, options, errorPerSequenceMAV(FILEPATH) );

%% DSVIO
logPath_fwd = [DSVIO_FILEPATH];
[ DSVIO_MAV_FWD ] = evalMAVdataset( logPath_fwd, options, errorPerSequenceMAV(FILEPATH));
save('logs/DSVIO.mat', 'DSVIO_MAV_FWD')
Plot
PlotErrorBar