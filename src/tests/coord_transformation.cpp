#define MAX_TRANSLATION 0.05
#define MAX_ROTATION 0.05

#define MAX_COORDINATE_TRANSLATION 100
// (+/-)pi (allow full rotation)
#define MAX_COORDINATE_ROTATION 3.141592653589793

#define MIN_DEPTH 4.0
#define MAX_DEPTH 8.0

#define SCALE 100

#define NUM_POINTS 3

#define FOCAL_LENGTH 1.0

#define IMAGE_COORD_RANGE 100

// MRPT includes
#include <mrpt/poses/CPose3DPDF.h>

#include <iostream>
#include <random>
#include <ctime>
#include "sophus/se3.hpp"
#include "util/NumType.h"

// GTSAM related includes.
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Pose3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace dso;
using namespace gtsam;
typedef Sophus::SE3d SE3;

// camera to imu rotation
Mat44 Tbc = (Mat44() << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                        0.0, 0.0, 0.0, 1.0).finished();

bool doubleCompare(double a, double b, double epsilon=0.01)
{
    return fabs(a - b) < epsilon;
}

Mat33 getOrientationOfVector(Vec3 cameraZaxis)
{
    cameraZaxis.normalize();
    Vec3 cameraYaxis;
    if (!doubleCompare(-cameraZaxis(1), cameraZaxis(0)))
    {
        // rotate z-axis by 90 degree in z direction (arbitrarily) to get Y
        cameraYaxis(0) = -cameraZaxis(1);
        cameraYaxis(1) = cameraZaxis(0);
        cameraYaxis(2) = 0;
    }
    else if (!doubleCompare(-cameraZaxis(0), cameraZaxis(2)))
    {
        // rotate z-axis by 90 degree in y direction (arbitrarily) to get Y
        cameraYaxis(0) = cameraZaxis(2);
        cameraYaxis(2) = -cameraZaxis(0);
        cameraYaxis(1) = 0;
    }
    else if (!doubleCompare(-cameraZaxis(2), cameraZaxis(1)))
    {
        // rotate z-axis by 90 degree in y direction (arbitrarily) to get Y
        cameraYaxis(1) = -cameraZaxis(2);
        cameraYaxis(2) = cameraZaxis(1);
        cameraYaxis(0) = 0;
    }
    else
    {
        std::cerr << "Your camera direction is weird!!! Quitting..." << std::endl;
        return Mat33::Identity();
    }

    cameraYaxis.normalize();

    Vec3 cameraXaxis = cameraYaxis.cross(cameraZaxis);
    cameraXaxis.normalize();

    Mat33 rotationMatrix;
    rotationMatrix.col(0) = cameraXaxis;
    rotationMatrix.col(1) = cameraYaxis;
    rotationMatrix.col(2) = cameraZaxis;

    return rotationMatrix;
}

Mat33
euler2RotMat( const double roll,
                  const double pitch,
                  const double yaw )
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q.matrix();
}


class Reprojection
{
protected:
    Mat33 K_;
    Vec3 u_i_;
    double d_i_;
public:
    Reprojection(Mat33 _K, Vec3 _u_i, double _d_i)
            :   K_(_K),
                u_i_(_u_i),
                d_i_(_d_i)
    {

    }

    /**
     *
     * @brief reproject such that jacobian is with respect to only camera j (and assume camera i is at identity)
     */
    Vector2 reprojectLocal(const Pose3 _T_wj, const Pose3 _T_wi, OptionalJacobian<2, 6> J = boost::none)
    {
        Vec3 p_w = _T_wi * (K_.inverse() * u_i_ * d_i_);
        Vec3 p_j = _T_wj.inverse() * p_w;
        double d_j = p_j(2);
        Vec3 u_norm_j = p_j / d_j;
        Vec3 u_j = K_ * u_norm_j;

        if (J)
        {
            Matrix23 Maux;
            Maux.setZero();
            Maux(0,0) = K_(0, 0);
            Maux(0,1) = 0;
            Maux(0,2) = -u_j(0) * K_(0, 0);
            Maux(1,0) = 0;
            Maux(1,1) = K_(1, 1);
            Maux(1,2) = -u_j(1) * K_(1, 1);
            Maux /= d_j;


            (*J).leftCols(3).noalias() = Maux * dso::SO3::hat(p_j);
            (*J).rightCols(3) = -Maux;
        }

        return (Vector2() << u_j(0), u_j(1)).finished();
    }

    /**
     *
     * @brief reproject such that jacobian is with respect to only imu j
     */
    Vector2 reprojectLocalIMU(const Pose3 _T_w_bj, const Pose3 _T_w_bi, OptionalJacobian<2, 6> J = boost::none)
    {
        Pose3 T_wi = _T_w_bi.compose(Pose3(Tbc));
        Pose3 T_wj = _T_w_bj.compose(Pose3(Tbc));

        Vec3 p_w = T_wi * (K_.inverse() * u_i_ * d_i_);
        Vec3 p_j = T_wj.inverse() * p_w;

        double d_j = p_j(2);
        Vec3 u_norm_j = p_j / d_j;
        Vec3 u_j = K_ * u_norm_j;

        if (J)
        {
            Matrix23 Maux;
            Maux.setZero();
            Maux(0,0) = K_(0, 0);
            Maux(0,1) = 0;
            Maux(0,2) = -u_j(0) * K_(0, 0);
            Maux(1,0) = 0;
            Maux(1,1) = K_(1, 1);
            Maux(1,2) = -u_j(1) * K_(1, 1);
            Maux /= d_j;


            (*J).leftCols(3).noalias() = Maux * Tbc.topLeftCorner(3, 3).transpose() * dso::SO3::hat(Pose3(Tbc)*p_j);
            (*J).rightCols(3) = -Maux * Tbc.topLeftCorner(3, 3).transpose();
        }

        return (Vector2() << u_j(0), u_j(1)).finished();
    }

    /**
     *
     * @brief reproject with respect to both poses
     */
    Vector2 reprojectTwoPose(const Pose3 _T_wj, const Pose3 _T_wi, OptionalJacobian<2, 12> J = boost::none)
    {
        SE3 T_ji(_T_wj.inverse().compose(_T_wi).matrix());

        Vec3 p_i = (K_.inverse() * u_i_ * d_i_);
        Vec3 p_j = T_ji * p_i;
        double d_j = p_j(2);
        double id_j = 1/d_j;
        Vec3 u_norm_j = p_j / d_j;
        Vec3 u_j = K_ * u_norm_j;

        if (J)
        {
            Matrix23 Maux;
            Maux.setZero();
            Maux(0,0) = K_(0, 0);
            Maux(0,1) = 0;
            Maux(0,2) = -u_j(0) * K_(0, 0);
            Maux(1,0) = 0;
            Maux(1,1) = K_(1, 1);
            Maux(1,2) = -u_j(1) * K_(1, 1);
            Maux /= d_j;


            (*J).leftCols(3).noalias() = Maux * dso::SO3::hat(p_j);
            (*J).middleCols(3, 3) = -Maux;

            Matrix66 adjoint_sophus = -T_ji.Adj();
            Matrix26 Jj_sophus;
            Jj_sophus.leftCols(3) = (*J).middleCols(3, 3);
            Jj_sophus.rightCols(3) = (*J).leftCols(3);

            Matrix26 Ji_sophus;
            Ji_sophus.noalias() = Jj_sophus * adjoint_sophus;

            Matrix26 Ji_gtsam;
            Ji_gtsam.leftCols(3) = Ji_sophus.rightCols(3);
            Ji_gtsam.rightCols(3) = Ji_sophus.leftCols(3);

            (*J).rightCols(6) = Ji_gtsam;
        }

        return (Vector2() << u_j(0), u_j(1)).finished();
    }

    /**
     *
     * @brief reproject with respect to both poses
     */
    Vector2 reprojectTwoPoseIMU(const Pose3 _T_w_bj, const Pose3 _T_w_bi, OptionalJacobian<2, 12> J = boost::none)
    {
        Pose3 T_wi = _T_w_bi.compose(Pose3(Tbc));
        Pose3 T_wj = _T_w_bj.compose(Pose3(Tbc));
        SE3 T_ji(T_wj.inverse().compose(T_wi).matrix());
        SE3 T_bj_bi(_T_w_bj.inverse().compose(_T_w_bi).matrix());

        Vec3 p_i = (K_.inverse() * u_i_ * d_i_);
        Vec3 p_j = T_ji * p_i;
        double d_j = p_j(2);
        double id_j = 1/d_j;
        Vec3 u_norm_j = p_j / d_j;
        Vec3 u_j = K_ * u_norm_j;

        if (J)
        {
            Matrix23 Maux;
            Maux.setZero();
            Maux(0,0) = K_(0, 0);
            Maux(0,1) = 0;
            Maux(0,2) = -u_j(0) * K_(0, 0);
            Maux(1,0) = 0;
            Maux(1,1) = K_(1, 1);
            Maux(1,2) = -u_j(1) * K_(1, 1);
            Maux /= d_j;


            (*J).leftCols(3).noalias() = Maux * Tbc.topLeftCorner(3, 3).transpose() * dso::SO3::hat(Pose3(Tbc)*p_j);
            (*J).middleCols(3, 3) = -Maux * Tbc.topLeftCorner(3, 3).transpose();

            Matrix66 adjoint_sophus = -T_bj_bi.Adj();
            Matrix26 Jj_sophus;
            Jj_sophus.leftCols(3) = (*J).middleCols(3, 3);
            Jj_sophus.rightCols(3) = (*J).leftCols(3);

            Matrix26 Ji_sophus;
            Ji_sophus.noalias() = Jj_sophus * adjoint_sophus;

            Matrix26 Ji_gtsam;
            Ji_gtsam.leftCols(3) = Ji_sophus.rightCols(3);
            Ji_gtsam.rightCols(3) = Ji_sophus.leftCols(3);

            (*J).rightCols(6) = Ji_gtsam;
        }

        return (Vector2() << u_j(0), u_j(1)).finished();
    }

};

int main(int argc, char **argv)
{
    srand(time(NULL));

    Mat33 K = Mat33::Identity();
    K(0, 0) = FOCAL_LENGTH;
    K(1, 1) = FOCAL_LENGTH;

    // ----------------------------- Pose of frame i ----------------------------- //
    Mat33 R_i = getOrientationOfVector(
            (Vec3() << rand(), rand(), rand()).finished()
    );
    Vec3 t_i = (Vec3() <<   ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000
    ).finished();

    // ----------------------------- increment for frame j ----------------------------- //
    Vec6 xi = (Vec6() <<    ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION
    ).finished();

    // ----------------------------- coordinate frame transformation after initialization ----------------------------- //
    Mat44 coordinate_transformation = Mat44::Identity();
    // rotation
    coordinate_transformation.topLeftCorner(3,3) = euler2RotMat(
            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_ROTATION,
            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_ROTATION,
            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_ROTATION
    );
    // translation
    coordinate_transformation(0, 3) = ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_TRANSLATION;
    coordinate_transformation(1, 3) = ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_TRANSLATION;
    coordinate_transformation(2, 3) = ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_COORDINATE_TRANSLATION;

    SE3 T_wi(R_i, t_i);
    SE3 T_wj = T_wi * SE3::exp(xi);

    SE3 T_w_bi = T_wi * SE3(Tbc).inverse();
    SE3 T_w_bj = T_wj * SE3(Tbc).inverse();

    // ----------------------------- point in ith frame ----------------------------- //
    std::vector<Reprojection> reprojections_0;
    std::vector<Reprojection> reprojections_1;
    reprojections_0.reserve(NUM_POINTS);
    reprojections_1.reserve(NUM_POINTS);
    for (size_t pt_idx = 0; pt_idx < NUM_POINTS; pt_idx++)
    {
        double d_i = MIN_DEPTH + (MAX_DEPTH - MIN_DEPTH) * ((double) rand()) / RAND_MAX;
        Vec3 u_i = (Vec3() << (((double) rand()) / RAND_MAX - 0.5) * 2 * IMAGE_COORD_RANGE,
                (((double) rand()) / RAND_MAX - 0.5) * 2 * IMAGE_COORD_RANGE,
                1
        ).finished();

        reprojections_0.push_back(Reprojection(K, u_i, d_i));
        reprojections_1.push_back(Reprojection(K, u_i, d_i*SCALE));

//        Vec3 u_norm_i = K.inverse() * u_i;

//        Vec3 p_i = d_i * u_norm_i;
//        Vec3 p_w = T_wi * p_i;
//        Vec3 p_j = T_wj.inverse() * p_w;
//
//        Vec3 u_norm_j = p_j / p_j(2);
//        Vec3 u_j = K * u_norm_j;
    }


    // numerical jacobian
//    boost::function<Vector2(const Pose3&, const Pose3&)> boostReproject = boost::bind(&Reprojection::reprojectTwoPoseIMU, reprojection, _1, _2, boost::none);
//    Matrix26 J_numerical_j = numericalDerivative21(
//            boostReproject,
//            Pose3(T_w_bj.matrix()),
//            Pose3(T_w_bi.matrix())
//    );
//    Matrix26 J_numerical_i = numericalDerivative22(
//            boostReproject,
//            Pose3(T_w_bj.matrix()),
//            Pose3(T_w_bi.matrix())
//    );
//    Eigen::Matrix<double, 2, 12> J_numerical;
//    J_numerical.leftCols(6) = J_numerical_j;
//    J_numerical.rightCols(6) = J_numerical_i;

    // In original frame
    Eigen::Matrix<double, 2*NUM_POINTS, 12> J_0;
    // In transformed frame
    Eigen::Matrix<double, 2*NUM_POINTS, 12> J_1;

    // just scale
    Mat44 T_w_bj2_mat = T_w_bj.matrix();
    Mat44 T_w_bi2_mat = T_w_bi.matrix();
    T_w_bj2_mat.topRightCorner(3, 1) *= SCALE;
    T_w_bi2_mat.topRightCorner(3, 1) *= SCALE;

    for (size_t pt_idx = 0; pt_idx < NUM_POINTS; pt_idx++)
    {
        Eigen::Matrix<double, 2, 12> temp;
        // In original frame
        reprojections_0[pt_idx].reprojectTwoPoseIMU(Pose3(T_w_bj.matrix()), Pose3(T_w_bi.matrix()), &temp);
        J_0.block<2, 12>(pt_idx*2, 0) = temp;
        // In transformed frame
//        reprojections_1[pt_idx].reprojectTwoPoseIMU(
//                Pose3(coordinate_transformation * T_w_bj.matrix()),
//                Pose3(coordinate_transformation * T_w_bi.matrix()),
//                &temp
//        );
        reprojections_1[pt_idx].reprojectTwoPoseIMU(
                Pose3(T_w_bj2_mat),
                Pose3(T_w_bi2_mat),
                &temp
        );
        J_1.block<2, 12>(pt_idx*2, 0) = temp;
    }

//    mrpt::math::CMatrixDouble66 df_dx(mrpt::math::UNINITIALIZED_MATRIX);
//    mrpt::math::CMatrixDouble66 df_du(mrpt::math::UNINITIALIZED_MATRIX);
//
//    mrpt::poses::CPose3DPDF::jacobiansPoseComposition(
//            mrpt::poses::CPose3D(mrpt::math::CMatrixDouble44(T_w_bi.matrix())),
//            mrpt::poses::CPose3D(mrpt::math::CMatrixDouble44(coordinate_transformation.matrix())),
//            df_dx,
//            df_du
//    );

    Eigen::Matrix<double, 2*NUM_POINTS, 12> J_1_calc;
//    Mat66 adj;
//    adj = df_dx.transpose();
//    adj= SE3(coordinate_transformation).Adj().inverse().transpose();
//    J_1_calc.leftCols(6) = J_0.leftCols(6) * adj;
//    J_1_calc.rightCols(6) = J_0.rightCols(6) * adj;

    std::cout << "J_0: \n" << J_0 << std::endl;
    std::cout << "J_1: \n" << J_1 << std::endl;

    cv::Mat cv_J_0_j(2*NUM_POINTS, 6, CV_64F);
    cv::Mat cv_J_0_i(2*NUM_POINTS, 6, CV_64F);

    cv::Mat cv_J_0_j_inv(6, 2*NUM_POINTS, CV_64F);
    cv::Mat cv_J_0_i_inv(6, 2*NUM_POINTS, CV_64F);

    Eigen::Matrix<double, 2*NUM_POINTS, 6> eigen_J_0_j = J_0.leftCols(6);
    Eigen::Matrix<double, 2*NUM_POINTS, 6> eigen_J_0_i = J_0.rightCols(6);

    Eigen::Matrix<double, 6, 2*NUM_POINTS> eigen_J_0_j_inv;
    Eigen::Matrix<double, 6, 2*NUM_POINTS> eigen_J_0_i_inv;

    cv::eigen2cv(eigen_J_0_j, cv_J_0_j);
    cv::eigen2cv(eigen_J_0_i, cv_J_0_i);

    cv::invert(cv_J_0_j, cv_J_0_j_inv, cv::DECOMP_SVD);
    cv::invert(cv_J_0_i, cv_J_0_i_inv, cv::DECOMP_SVD);

    cv::cv2eigen(cv_J_0_j_inv, eigen_J_0_j_inv);
    cv::cv2eigen(cv_J_0_i_inv, eigen_J_0_i_inv);

    Mat66 transform_J_j = eigen_J_0_j_inv * J_1.leftCols(6);
    Mat66 transform_J_i = eigen_J_0_i_inv * J_1.rightCols(6);

    J_1_calc.leftCols(6) = J_0.leftCols(6) * transform_J_j;
    J_1_calc.rightCols(6) = J_0.rightCols(6) * transform_J_i;

    std::cout << "J_1 cal: \n" << J_1_calc << std::endl;


    std::cout << "transform for j: \n" << transform_J_j << std::endl;
    std::cout << "transform for i: \n" << transform_J_i << std::endl;
    std::cout << "transform diff: \n" << transform_J_j - transform_J_i << std::endl;
//    std::cout << "calculated: \n" << J_1_calc << std::endl;
//    std::cout << "err: \n" << J_1_calc - J_1 << std::endl;


//    std::cout << "jacobian err: \n" << J_0 - J_numerical << std::endl;
    return 0;
}