#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt

PREFIX = 'E vs IMU_error is :'

if len(sys.argv) != 2:
    print 'Usage python visualize_imu_photo_error <log_file>'
    exit(0)

f = open(sys.argv[1], 'r')
lines = f.readlines()

list_errors = [[] for i in range(5)]

for line in lines:
    line_stripped = line.strip()
    if line_stripped.startswith(PREFIX):
        errors = line_stripped[len(PREFIX):].split(' ')
        if len(errors) != 4:
            continue
        errors = map(float, errors)
        list_errors[int(errors[3])].append(errors[:3])

for i in range(5):
    list_errors[i] = [ [list_errors[i][row][j] for row in range(len(list_errors[i]))] for j in range(3) ]


for i in range(5):
    plt.figure()
    plt.plot(list_errors[i][0], label="photometric")
    plt.hold(True)
    plt.plot(list_errors[i][1], label="IMU"+str(i))
    plt.hold(True)
    plt.plot(list_errors[i][2], label="angle_error")
    plt.legend()


plt.show()


