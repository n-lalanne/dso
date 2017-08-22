#ifndef CONFIGPARAM_H
#define CONFIGPARAM_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#define TRACK_WITH_IMU

#define RUN_REALTIME

#define USE_BIAS_PRIOR 

namespace dso_vi
{

class ConfigParam
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConfigParam(std::string configfile);

    double _testDiscardTime;

    static Eigen::Matrix4d GetEigTbc();
    static cv::Mat GetMatTbc();
    static Eigen::Matrix4d GetEigT_cb();
    static cv::Mat GetMatT_cb();
    static int GetLocalWindowSize();
    static double GetImageDelayToIMU();
    static bool GetAccMultiply9p8();
    static bool Getaddprior();
    static double Getaccel_noise_sigma();
    static double Getgyro_noise_sigma();
    static double Getaccel_bias_rw_sigma();
    static double Getgyro_bias_rw_sigma();
    
    static cv::Mat GetMatAccBias();
    static Eigen::Vector3d GetEigAccBias();

    static cv::Mat GetMatGyroBias();
    static Eigen::Vector3d GetEigGyroBias();

    static cv::Mat _MatAccBias;
    static Eigen::Vector3d _EigAccBias;

    static cv::Mat _MatGyroBias;
    static Eigen::Vector3d _EigGyroBias;

    static double GetG(){return _g;}

    std::string _bagfile;
    std::string _imageTopic;
    std::string _imuTopic;

    static bool addprior;
    static double accel_noise_sigma;
    static double gyro_noise_sigma;
    static double accel_bias_rw_sigma;
    static double gyro_bias_rw_sigma;

    static double GetVINSInitTime(){return _nVINSInitTime;}

private:
    static Eigen::Matrix4d _EigTbc;
    static cv::Mat _MatTbc;
    static Eigen::Matrix4d _EigTcb;
    static cv::Mat _MatTcb;
    static int _LocalWindowSize;
    static double _ImageDelayToIMU;
    static bool _bAccMultiply9p8;


    static double _g;
    static double _nVINSInitTime;

};

}

#endif // CONFIGPARAM_H
