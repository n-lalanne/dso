#ifndef IMUDATA_H
#define IMUDATA_H

#include <Eigen/Dense>
#include "util/settings.h"

namespace dso_vi
{

using namespace Eigen;

class IMUData
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // covariance of measurement
    static Matrix3d _gyrMeasCov;
    static Matrix3d _accMeasCov;
    static Matrix3d getGyrMeasCov(void) {return _gyrMeasCov;}
    static Matrix3d getAccMeasCov(void) {return _accMeasCov;}

    // covariance of bias random walk
    static Matrix3d _gyrBiasRWCov;
    static Matrix3d _accBiasRWCov;
    static Matrix3d getGyrBiasRWCov(void) {return _gyrBiasRWCov;}
    static Matrix3d getAccBiasRWCov(void) {return _accBiasRWCov;}

    static double _gyrBiasRw2;
    static double _accBiasRw2;
    static double getGyrBiasRW2(void) {return _gyrBiasRw2;}
    static double getAccBiasRW2(void) {return _accBiasRw2;}


    IMUData(const double& gx, const double& gy, const double& gz,
            const double& ax, const double& ay, const double& az,
            const double& t);
    //IMUData(const IMUData& imu);

    // Raw data of imu's
    Vector3d _g;    //gyr data
    Vector3d _a;    //acc data
    double _t;      //timestamp

    static Matrix4d convertRelativeCamFrame2RelativeIMUFrame(Matrix4d camPose)
    {
        return dso_vi::Tbc.matrix() * camPose * dso_vi::Tcb.matrix();
    }

    static Matrix4d convertRelativeIMUFrame2RelativeCamFrame(Matrix4d imuPose)
    {
        return dso_vi::Tcb.matrix() * imuPose * dso_vi::Tbc.matrix();
    }

};

}

#endif // IMUDATA_H
