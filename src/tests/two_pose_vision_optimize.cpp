#define MAX_TRANSLATION 0.05
#define MAX_ROTATION 0.05

#define MIN_DEPTH 4.0
#define MAX_DEPTH 8.0

#define FOCAL_LENGTH 1.0

#define IMAGE_COORD_RANGE 100

#include <iostream>
#include <random>
#include <ctime>
#include "sophus/se3.hpp"
#include "util/NumType.h"

// GTSAM related includes.
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Pose3.h>

using namespace dso;
using namespace gtsam;
typedef Sophus::SE3d SE3;

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
//            (*J)(0, 0) = -u_j(0) * u_j(1) * K_(0, 0);
//            (*J)(0, 1) = (1 + u_j(0)*u_j(0))*K_(0, 0);
//            (*J)(0, 2) = -u_j(2) * K_(0, 0);
//            (*J)(0, 3) = id_j * K_(0, 0);
//            (*J)(0, 4) = 0;
//            (*J)(0, 5) = -id_j * u_j(0) * K_(0, 0);
//
//            (*J)(1, 0) = -(1 + u_j(1)*u_j(1))*K_(1, 1);
//            (*J)(1, 1) = u_j(0) * u_j(1) * K_(1, 1);
//            (*J)(1, 2) = u_j(1) * K_(1, 1);
//            (*J)(1, 3) = 0;
//            (*J)(1, 4) = id_j * K_(1, 1);
//            (*J)(1, 5) = -id_j * u_j(1) * K_(1, 1);

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

};

int main(int argc, char **argv)
{
    srand(time(NULL));

    Mat33 K = Mat33::Identity();
    K(0, 0) = FOCAL_LENGTH;
    K(1, 1) = FOCAL_LENGTH;

    // Pose of frame i
    Mat33 R_i = getOrientationOfVector(
            (Vec3() << rand(), rand(), rand()).finished()
    );
    Vec3 t_i = (Vec3() <<   ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * 5000
    ).finished();

    // increment for frame j
    Vec6 xi = (Vec6() <<    ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_TRANSLATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * MAX_ROTATION
    ).finished();

    // point in ith frame
    double d_i = MIN_DEPTH + (MAX_DEPTH - MIN_DEPTH) * ((double)rand()) / RAND_MAX;
    Vec3 u_i = (Vec3() <<   ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * IMAGE_COORD_RANGE,
                            ( ((double)rand()) / RAND_MAX - 0.5 ) * 2 * IMAGE_COORD_RANGE,
                            1
    ).finished();
    Vec3 u_norm_i = K.inverse() * u_i;

    SE3 T_wi(R_i, t_i);
    SE3 T_wj = T_wi * SE3::exp(xi);

    Vec3 p_i = d_i * u_norm_i;
    Vec3 p_w = T_wi * p_i;
    Vec3 p_j = T_wj.inverse() * p_w;

    Vec3 u_norm_j = p_j / p_j(2);
    Vec3 u_j = K * u_norm_j;

    Reprojection reprojection(K, u_i, d_i);

    // ------------------------- One pose (camera frame) ------------------------- //
    // numerical jacobian
//    boost::function<Vector2(const Pose3&)> boostReproject = boost::bind(&Reprojection::reprojectLocal, reprojection, _1, Pose3(T_wi.matrix()), boost::none);
//    Matrix26 J_numerical = numericalDerivative11(
//            boostReproject,
//            Pose3(T_wj.matrix())
//    );
//    // analytic jacobian
//    Matrix26 J_analytic;
//    reprojection.reprojectLocal(Pose3(T_wj.matrix()), Pose3(T_wi.matrix()), &J_analytic);
//
//    std::cout << "J_analytic: \n" << J_analytic << std::endl;
//    std::cout << "J_numerical: \n" << J_numerical << std::endl;

    // ------------------------- Two pose (camera frame) ------------------------- //
    boost::function<Vector2(const Pose3&, const Pose3&)> boostReproject = boost::bind(&Reprojection::reprojectTwoPose, reprojection, _1, _2, boost::none);
    Matrix26 J_numerical_j = numericalDerivative21(
            boostReproject,
            Pose3(T_wj.matrix()),
            Pose3(T_wi.matrix())
    );
    Matrix26 J_numerical_i = numericalDerivative22(
            boostReproject,
            Pose3(T_wj.matrix()),
            Pose3(T_wi.matrix())
    );
    Eigen::Matrix<double, 2, 12> J_numerical;
    J_numerical.leftCols(6) = J_numerical_j;
    J_numerical.rightCols(6) = J_numerical_i;
    // analytic jacobian
    Eigen::Matrix<double, 2, 12> J_analytic;
    reprojection.reprojectTwoPose(Pose3(T_wj.matrix()), Pose3(T_wi.matrix()), &J_analytic);

    std::cout << "J_analytic: \n" << J_analytic << std::endl;
    std::cout << "J_numerical: \n" << J_numerical << std::endl;
    std::cout << "error: \n" << J_analytic - J_numerical << std::endl;

    return 0;
}