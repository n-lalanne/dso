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
        if len(errors) != 3:
            continue
        errors = map(float, errors)
        list_errors[int(errors[2])].append(errors[:2])

for i in range(5):
    list_errors[i] = [ [list_errors[i][row][j] for row in range(len(list_errors[i]))] for j in range(2) ]


for i in range(5):
    plt.plot(list_errors[i][0], label="photometric")
    plt.hold(True)
    plt.plot(list_errors[i][1], label="IMU")
    plt.legend()
    plt.figure()

plt.show()


