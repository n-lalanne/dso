% set this to the supplementary files folder.
FILEPATH = '/home/rakesh/SLAM/dso_supplementary/supp_v2'

DSO_EVAL_FILEPATH = '/home/rakesh/SLAM/dso_supplementary/Matlab_Evaluation';
DSO_FILEPATH = FILEPATH;
DSVIO_FILEPATH = '/home/rakesh/workspace/dso/Matlab_evaluation/logs/';

addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);

options=0:9;

%% DSO
logPath_fwd = [DSO_FILEPATH '/DS-VO_Forward/'];
[ DSO_MAV_FWD ] = evalMAVdataset( logPath_fwd, options, errorPerSequenceMAV(FILEPATH) );

%% DSVIO
logPath_fwd = [DSVIO_FILEPATH];
[ DSVIO_MAV_FWD ] = evalMAVdataset( logPath_fwd, options, errorPerSequenceMAV(FILEPATH) );
save('DSVIO.mat', 'DSVIO_MAV_FWD')