import os
import sys
from time import sleep
import subprocess

datasets = [
	# {
 #        'name': 'MH_01_easy',
 #        'startFrame': 950,
 #        'endFrame': 3600,
 #        'WINDOWSIZE': 120
 #    },
    # {
    #     'name': 'MH_02_easy',
    #     'startFrame': 800,
    #     'endFrame': 3000,
    #     'WINDOWSIZE': 120
    # },
    # {
    #     'name': 'MH_03_medium',
    #     'startFrame': 430,
    #     'endFrame': 2600,
    #     'WINDOWSIZE': 120
    # },
    # {
    #     'name': 'MH_04_difficult',
    #     'startFrame': 460,
    #     'endFrame': 1925,
    #     'WINDOWSIZE': 40
    # },
    # {
    #     'name': 'MH_05_difficult',
    #     'startFrame': 460,
    #     'endFrame': 2200,
    #     'WINDOWSIZE': 70
    # },
    {
        'name': 'V1_01_easy',
        'startFrame': 100,
        'endFrame': 2800,
        'WINDOWSIZE': 70
    },
    # {
    #     'name': 'V1_02_medium',
    #     'startFrame': 115,
    #     'endFrame': 1600,
    #     'WINDOWSIZE': 120
    # },
    # {
    #     'name': 'V1_03_difficult',
    #     'startFrame': 250,
    #     'endFrame': 2020,
    #     'WINDOWSIZE': 40
    # },
    # {
    #     'name': 'V2_01_easy',
    #     'startFrame': 80,
    #     'endFrame': 2130,
    #     'WINDOWSIZE': 120
    # },
    # {
    #     'name': 'V2_02_medium',
    #     'startFrame': 100,
    #     'endFrame': 2230,
    #     'WINDOWSIZE': 60
    # },
    # {
    #     'name': 'V2_03_difficult',
    #     'startFrame': 115,
    #     'endFrame': 1880,
    #     'WINDOWSIZE': 40
    # }
]

def findPIDS(process_name):
	p = os.popen('ps -C ' + process_name, "r")
	pids = []
	while 1:
	    line = p.readline()
	    if not line: break
	    words = line.split(' ')
	    if len(words):
	    	try:
	    		pid = int(words[0])
	    		pids.append(pid)
	    	except ValueError:
	    		pass
	    print line
	return pids

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python run_evaluation.py <dataset_dir>')
		exit(0)
	dataset_dir = sys.argv[1]

	for dataset in datasets:
		for iter_idx in range(2):
			os.system('killall roslaunch')
			os.system('killall rosbag')
			
			os.system('roslaunch vins_estimator euroc.launch &');
			os.system(
				'rosbag play ' + 
				dataset_dir + '/' + dataset['name'] + '.bag' +
				' -s ' + str( dataset['startFrame']/20.0 ) +
				' -u ' + str( (dataset['endFrame'] - dataset['startFrame'])/20.0 )
			)
			# wait for the files to be saved
			sleep(2)

			os.system(
				'cp /home/sicong/VIN_ws/src/VINS-Mono/config/euroc/vins_result.csv logs/' + 
				'mav' + dataset['name'] + '_' + str(iter_idx) + '.txt' 
			)

	os.system('killall roslaunch')
	os.system('killall rosbag')
			