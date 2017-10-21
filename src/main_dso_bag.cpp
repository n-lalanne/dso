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

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include "GroundTruthIterator/GroundTruthIterator.h"

#include "IMU/configparam.h"
#include "IMU/imudata.h"

#include "MsgSync/MsgSynchronizer.h"

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

// GTSAM related includes.
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>


std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
std::string configFile = "";
std::string groundTruthFile = "";
std::string bagFile = "";
std::string outputFile = "";
int startFrame = 0;
int endFrame = std::numeric_limits<int>::max();

unsigned frameCount = 0;
bool addprior;

bool useSampleOutput=false;
bool is_exit = false;

std::ofstream angleComparisonFile;
gtsam::Pose3 relativePose;

using namespace dso;

void parseArgument(char* arg)
{
    int option;
    char buf[1000];

    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }


    if(1==sscanf(arg,"nolog=%d",&option))
    {
        if(option==1)
        {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }

    if(1==sscanf(arg,"nogui=%d",&option))
    {
        if(option==1)
        {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if(1==sscanf(arg,"nomt=%d",&option))
    {
        if(option==1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if(1==sscanf(arg,"calib=%s",buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }
    if(1==sscanf(arg,"vignette=%s",buf))
    {
        vignetteFile = buf;
        printf("loading vignette from %s!\n", vignetteFile.c_str());
        return;
    }

    if(1==sscanf(arg,"gamma=%s",buf))
    {
        gammaFile = buf;
        printf("loading gammaCalib from %s!\n", gammaFile.c_str());
        return;
    }

    if(1==sscanf(arg,"config=%s",buf))
    {
        configFile = buf;
        printf("loading config from %s!\n", configFile.c_str());
        return;
    }

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

    if(1==sscanf(arg,"output=%s",buf))
    {
        outputFile = buf;
        printf("loading output from %s!\n", outputFile.c_str());
        return;
    }

    if(1==sscanf(arg,"start_frame=%d",&startFrame))
    {
        printf("start frame %d\n", startFrame);
        return;
    }

    if(1==sscanf(arg,"end_frame=%d",&endFrame))
    {
        printf("end frame %d\n", endFrame);
        return;
    }


    printf("could not parse argument \"%s\"!!\n", arg);
}




FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;

sensor_msgs::ImageConstPtr imageMsg;
std::vector<sensor_msgs::ImuConstPtr> vimuMsg;

void track(const sensor_msgs::ImageConstPtr img, std::vector<dso_vi::IMUData> vimuData,
           dso_vi::ConfigParam &config, dso_vi::GroundTruthIterator::ground_truth_measurement_t groundtruth)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    assert(cv_ptr->image.type() == CV_8U);
    assert(cv_ptr->image.channels() == 1);


    if(setting_fullResetRequested)
    {
        std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
        delete fullSystem;
        for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
        fullSystem = new FullSystem();
        fullSystem->setTbc(config.GetEigTbc());
        fullSystem->linearizeOperation=false;
        fullSystem->outputWrapper = wraps;
        if(undistorter->photometricUndist != 0)
            fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
        setting_fullResetRequested=false;
    }

    MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
    ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
    fullSystem->addActiveFrame(undistImg, frameID, vimuData, img->header.stamp.toSec(), config, groundtruth);

    frameID++;

    //-------------------- Get relative pose -------------------- //
    std::vector<FrameShell*> allFrameHistory = fullSystem->getAllFrameHistory();
    if (!fullSystem->initialized || allFrameHistory.size() < 100)
    {
        return;
    }
    FrameShell* scurrent = allFrameHistory[allFrameHistory.size()-1];
    FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
    if (!scurrent || !slast || !scurrent->poseValid || !slast->poseValid)
    {
        return;
    }
    SE3 scurrent_2_slast = slast->camToWorld.inverse() * scurrent->camToWorld;
    // predicted by DSO
    Eigen::Quaternion<double> quaternionDSO = scurrent_2_slast.so3().unit_quaternion();
    // predicted by IMU
    gtsam::Vector3 gyroBias(-0.002153, 0.020744, 0.075806);
    gtsam::Vector3 acceleroBias(-0.013337, 0.103464, 0.093086);
    gtsam::imuBias::ConstantBias biasPrior(acceleroBias, gyroBias);
    PreintegrationType *imu_preintegrated = new PreintegratedImuMeasurements(
            dso_vi::getIMUParams(),
            biasPrior
    );

    Eigen::Matrix<double,3,3> Rbc = config.GetEigTbc().block<3,3>(0,0);



    double old_timestamp = slast->viTimestamp;
    for(dso_vi::IMUData  imudata: vimuData)
    {
        dso::Mat61 rawimudata;
        rawimudata <<   imudata._a(0), imudata._a(1), imudata._a(2),
                imudata._g(0), imudata._g(1), imudata._g(2);

//        rawimudata.head<3>() = Rbc * rawimudata.head<3>();
//        rawimudata.tail<3>() = Rbc * rawimudata.tail<3>();

        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "Data: ";
        for (int i = 0; i < 6; i++)
        {
            std::cout << rawimudata(i) << ", ";
        }
        std::cout << std::endl;
        std::cout << "Timestamp: " << std::fixed << imudata._t << ", " << old_timestamp << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        double dt = (imudata._t - old_timestamp);
        if (dt >= 0.0001) {
            imu_preintegrated->integrateMeasurement(
                    rawimudata.head<3>(),
                    rawimudata.tail<3>(),
                    dt
            );
        }
        old_timestamp = imudata._t;
    }

    gtsam::Rot3 gtsamRbc = gtsam::Rot3(Rbc);
    gtsam::Rot3 gtsamRcb = gtsamRbc.inverse();
    Eigen::Quaternion<double> quaternionIMU = gtsamRcb.compose( imu_preintegrated->deltaRij() ).compose(gtsamRbc).toQuaternion();

    // from groundtruth
    Eigen::Quaternion<double> quaternionGT = gtsamRcb.compose( relativePose.rotation() ).compose(gtsamRbc).toQuaternion();

    angleComparisonFile << quaternionDSO.x() << ", " << quaternionDSO.y() << ", " << quaternionDSO.z() << ", "
                        << quaternionIMU.x() << ", " << quaternionIMU.y() << ", " << quaternionIMU.z() << ", "
                        << quaternionGT.x() << ", " << quaternionGT.y() << ", " << quaternionGT.z()
                        << std::endl;

    delete undistImg;

}

int step(dso_vi::MsgSynchronizer &msgsync, dso_vi::ConfigParam &config, dso_vi::GroundTruthIterator &groundtruthIterator)
{
    // 3dm imu output per g. 1g=9.80665 according to datasheet
    const double g3dm = 9.80665;
    const bool bAccMultiply98 = config.GetAccMultiply9p8();
    const double nAccMultiplier = config.GetAccMultiply9p8() ? g3dm : 1;

    static double nPreviousImageTimestamp = -1;
    bool bdata = msgsync.getRecentMsgs(imageMsg, vimuMsg);

    if (bdata)
    {
        frameCount++;

        std::vector<dso_vi::IMUData> vimuData;
        vimuData.reserve(vimuMsg.size());

        for (sensor_msgs::ImuConstPtr &imuMsg: vimuMsg)
        {
            vimuData.push_back(
                    dso_vi::IMUData(
                            imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z,
                            imuMsg->linear_acceleration.x * nAccMultiplier,
                            imuMsg->linear_acceleration.y * nAccMultiplier,
                            imuMsg->linear_acceleration.z * nAccMultiplier,
                            imuMsg->header.stamp.toSec()
                    )
            );
        }
        ROS_INFO("time- %f, %ld IMU message between the images", imageMsg->header.stamp.toSec(), vimuData.size());
        ROS_INFO("Cam- %f. %f, IMU- %f, %f", nPreviousImageTimestamp, imageMsg->header.stamp.toSec(), vimuData[0]._t, vimuData.back()._t);
        if (nPreviousImageTimestamp > 0)
        {
            // read the groundtruth pose between the two camera poses
            dso_vi::GroundTruthIterator::ground_truth_measurement_t previousState;
            dso_vi::GroundTruthIterator::ground_truth_measurement_t currentState;

            try
            {
                relativePose = groundtruthIterator.getGroundTruthBetween(
                        nPreviousImageTimestamp, imageMsg->header.stamp.toSec(),
                        previousState, currentState
                );
            }
            catch (std::exception &e)
            {
                std::cerr << e.what() << std::endl;
                std::cout << "Ran out of groundtruth, not exiting though..." << std::endl;
//                return 1;
            }
            ROS_INFO("GT VS CAM, Start %f, End %f",
                     (previousState.timestamp - nPreviousImageTimestamp)*1e3,
                     (currentState.timestamp - imageMsg->header.stamp.toSec())*1e3
            );
//			ROS_INFO("%f - %f t: %f, %f, %f",
//				nPreviousImageTimestamp,
//				imageMsg->header.stamp.toSec(),
//				relativePose.translation().x(),
//				relativePose.translation().y(),
//				relativePose.translation().z()
//			);

            if (frameCount >= startFrame && frameCount <= endFrame)
            {
                track(imageMsg, vimuData, config, currentState);
            }
            else if (frameCount > endFrame)
            {
                is_exit = true;
            }
        }
        nPreviousImageTimestamp = imageMsg->header.stamp.toSec();
    }
    return 0;
}


int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parseArgument(argv[i]);

    if (configFile.empty())
    {
        printf("Config file location missing\n");
        return 1;
    }

    if (groundTruthFile.empty())
    {
        printf("Groundtruth file location missing\n");
        return 1;
    }

    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations=6;
    setting_minOptIterations=1;
    setting_logStuff = false;
    setting_kfGlobalWeight = 1;
    setting_keyframesPerSecond = 0.5;


    printf("MODE WITH CALIBRATION, but without exposure times!\n");
    setting_photometricCalibration = 2;
    setting_affineOptModeA = 0;
    setting_affineOptModeB = 0;



    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());

    // --------------------------------- Configs --------------------------------- //
    dso_vi::ConfigParam config(configFile);

    dso_vi::accel_noise_sigma = config.Getaccel_noise_sigma();
    dso_vi::gyro_noise_sigma = config.Getgyro_noise_sigma();
    dso_vi::accel_bias_rw_sigma = config.Getaccel_bias_rw_sigma();
    dso_vi::gyro_bias_rw_sigma = config.Getgyro_bias_rw_sigma();

    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=true;
    fullSystem->setTbc(config.GetEigTbc());
    fullSystem->setBiasEstimate(config.GetEigAccBias(), config.GetEigGyroBias());
    fullSystem->addprior = config.Getaddprior();
    fullSystem->addimu = config.Getaddimu();
    fullSystem->WINDOW_SIZE = 60;


    if(!disableAllDisplay)
        fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
                (int)undistorter->getSize()[0],
                (int)undistorter->getSize()[1]));


    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
        fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    dso_vi::MsgSynchronizer msgsync( config.GetImageDelayToIMU() );

    // logging
    angleComparisonFile.open("angle_comparison.txt");

    ros::Subscriber imgSub;
    ros::Subscriber imuSub;
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
        std::string imutopic = config._imuTopic;
        std::string imagetopic = config._imageTopic;
        topics.push_back(imagetopic);
        topics.push_back(imutopic);

        bagView = new rosbag::View(bag, rosbag::TopicQuery(topics), ros::TIME_MIN, ros::TIME_MAX);

        ROS_INFO("BAG starts at: %f", bagView->getBeginTime().toSec());
    }


    dso_vi::GroundTruthIterator groundtruthIterator(groundTruthFile);

    signal(SIGINT, [](int s) -> void {
        std::cout << "Caught signal " << s << std::endl;
        is_exit = true;
    });

    BOOST_FOREACH(rosbag::MessageInstance const m, *bagView)
    {
        sensor_msgs::ImuConstPtr simu = m.instantiate<sensor_msgs::Imu>();
        if(simu!=NULL)
        {
            msgsync.imuCallback(simu);
        }
        sensor_msgs::ImageConstPtr simage = m.instantiate<sensor_msgs::Image>();
        if(simage!=NULL)
        {
            msgsync.imageCallback(simage);
        }

        if (step(msgsync, config, groundtruthIterator) || is_exit)
        {
            break;
        }
    }

    FILE *trajectory_log = fopen((std::string("logs/")+outputFile).c_str(), "w");

    if (trajectory_log)
    {
        std::cout << "Saving trajectory" << std::endl;
        for(size_t fh_idx=0, len=fullSystem->getAllKeyFrameHistory().size(); fh_idx < len; fh_idx++)
        {
            FrameShell *fs = fullSystem->getAllKeyFrameHistory()[fh_idx];
            dso::SE3 pose = fs->camToWorld;
            fprintf(
                    trajectory_log, "%.5f %f %f %f %f %f %f %f\n",
                    fs->viTimestamp,
                    pose.translation()(0),
                    pose.translation()(1),
                    pose.translation()(2),
                    pose.unit_quaternion().x(),
                    pose.unit_quaternion().y(),
                    pose.unit_quaternion().z(),
                    pose.unit_quaternion().w()
            );
        }
        fclose(trajectory_log);
        std::cout << "Trajectory saved" << std::endl;
    }
    else
    {
        std::cout << "Could not log trajectory" << std::endl;
    }



    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }

    delete undistorter;
    delete fullSystem;

    delete bagView;

    angleComparisonFile.close();

    return 0;
}