//
// Created by sicong on 19/10/17.
//

//
// Created by rakesh on 01/10/17.
//

#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <csignal>

#include "MsgSync/MsgSynchronizer.h"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>


std::string groundTruthFile = "";
std::string bagFile = "";

void parseArgument(char* arg)
{
    char buf[1000];

    if(1==sscanf(arg,"groundtruth=%s",buf))
    {
        groundTruthFile = buf;
        printf("loading groundTruth from %s!\n", groundTruthFile.c_str());
        return;
    }

    if(1==sscanf(arg,"bag=%s",buf))
    {
        bagFile = buf;
        printf("loading bag from %s!\n", bagFile.c_str());
        return;
    }

    printf("could not parse argument \"%s\"!!\n", arg);
}

int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parseArgument(argv[i]);

    if (groundTruthFile.empty())
    {
        printf("Groundtruth file location missing\n");
        return 1;
    }

    rosbag::Bag bag;
    rosbag::View *bagView = NULL;

    if (bagFile.empty())
    {
        std::cerr << "bagfile empty" << std::endl;
        exit(1);
    }
    else
    {
        ROS_INFO("Playing bagfile: %s", bagFile.c_str());
        bag.open(bagFile, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("/vicon/ueye/ueye"));

        bagView = new rosbag::View(bag, rosbag::TopicQuery(topics), ros::TIME_MIN, ros::TIME_MAX);

        ROS_INFO("BAG starts at: %f", bagView->getBeginTime().toSec());
    }


    FILE *groundtruthLog = fopen((std::string("logs/")+groundTruthFile).c_str(), "w");

    if (groundtruthLog)
    {
        std::cout << "Saving groundtruth" << std::endl;
        BOOST_FOREACH(rosbag::MessageInstance const m, *bagView)
        {
            geometry_msgs::TransformStampedConstPtr stransform = m.instantiate<geometry_msgs::TransformStamped>();
            if(stransform)
            {
                fprintf(
                        groundtruthLog, "%f %f %f %f %f %f %f %f\n",
                        stransform->header.stamp.toSec(),
                        stransform->transform.translation.x,
                        stransform->transform.translation.y,
                        stransform->transform.translation.z,
                        stransform->transform.rotation.x,
                        stransform->transform.rotation.y,
                        stransform->transform.rotation.z,
                        stransform->transform.rotation.w
                );
            }
        }
        fclose(groundtruthLog);
        std::cout << "groundtruth saved" << std::endl;
    }
    else
    {
        std::cout << "Could not log groundtruth" << std::endl;
    }

    delete bagView;

    return 0;
}