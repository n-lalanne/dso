DSO_EVAL_FILEPATH = '/home/rakesh/SLAM/dso_supplementary/Matlab_Evaluation';

addpath(DSO_EVAL_FILEPATH);
addpath([DSO_EVAL_FILEPATH '/MakePlots']);

%% Figure 10. Results on EuRoC MAV and ICL NUIM datasets.
figure % clf

max_error = 5.0

the = 0.8;
steps=0:0.05:0.5;
% subplot(1,2,1);

[ dso_mav_rmse, dso_mav_n] = getSortedMAVError( DSO_MAV_FWD, [0:9], the, 1:22 );
[ dsvio_mav_rmse, dsvio_mav_n] = getSortedMAVError( DSVIO_MAV_FWD, [0:9], the, 1:22 );

hold on
semilogx(dso_mav_rmse,220*(1:dso_mav_n)/dso_mav_n,'blue','LineWidth',2)
semilogx(dsvio_mav_rmse,220*(1:dsvio_mav_n)/dsvio_mav_n,'red','LineWidth',2)


legend('DSO', 'DSVIO', 'Location', 'eastoutside')
axis([0 0.5 0 220]);
grid on

set(gca, 'XTick',[0:0.1:0.5])
title('EuRoC MAV')


%% Figure 12. Full Evaluation Result (EuRoC MAV).
figure
% clf
DSVIO_MAV = nan(10,22);
DSO_MAV = nan(10,22);
for i=1:22
    DSVIO_MAV(1:10,i) = DSVIO_MAV_FWD{i}.allSegRMSE([1:10]);
    
    DSO_MAV(1:10,i) = DSO_MAV_FWD{i}.allSegRMSE(1:10);
end


subplot(1,2,1)
imagesc(imresize(DSVIO_MAV,20,'nearest') );

caxis([0 max_error]); %caxis([0 0.5]);
hold on
plot(20*[0 23],20*[10 10]+0.5,'black','LineWidth',2)
plot(20*[5 5]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[8 8]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[11 11]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[16 16]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[19 19]+0.5,20*[-100 100],'black','LineWidth',2)

set(gca, 'YTick',[5 15]*20-10)
set(gca, 'YTickLabel',{'Fwd', 'Bwd'})
set(gca, 'XTick',[3 7 10 14 18 21]*20-10)
set(gca, 'XTickLabel',{'MH_l', 'V1_l', 'V2_l', 'MH_r', 'V1_r', 'V2_r'})

title('DSVIO')




subplot(1,2,2)
imagesc(imresize(DSO_MAV,20,'nearest') );
caxis([0 max_error]);
hold on
plot(20*[0 23],20*[10 10]+0.5,'black','LineWidth',2)
plot(20*[5 5]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[8 8]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[11 11]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[16 16]+0.5,20*[-100 100],'black','LineWidth',2)
plot(20*[19 19]+0.5,20*[-100 100],'black','LineWidth',2)



set(gca, 'YTick',[5 15]*20-10)
set(gca, 'YTickLabel',{'Fwd', 'Bwd'})
set(gca, 'XTick',[3 7 10 14 18 21]*20-10)
set(gca, 'XTickLabel',{'MH_l', 'V1_l', 'V2_l', 'MH_r', 'V1_r', 'V2_r'})
colorbar
title('DSO')
