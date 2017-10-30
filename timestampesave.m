fileFolder=fullfile('/home/sicong/eurocdata/MH_01_easy/mav0/cam0/data');
dirOutput=dir(fullfile(fileFolder,'*.png'));
fileNames={dirOutput.name}';
fid=fopen('time.txt','w');



[n,m] = size(fileNames); 
fprintf(fid,'%d\n',n);
for i=1:n
    filename = fileNames{i};
    filename = filename(1:end-4);
    stamp = str2num(filename);
    fprintf(fid,'%d\n',stamp);
end
fclose(fid);