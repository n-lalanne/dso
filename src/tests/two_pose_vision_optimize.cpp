#include <iostream>
#include <random>
#include <ctime>
#include "sophus/se3.hpp"
#include "util/NumType.h"

using namespace dso;
using namespace Sophus;

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

int main(int argc, char **argv)
{
    srand(time(NULL));
    Mat33 R = getOrientationOfVector(
            (Vec3() << rand(), rand(), rand()).finished()
    );

    std::cout << "R: \n" << R << std::endl;
    std::cout << "det: " << R.determinant() << std::endl;
    std::cout << "RT R: \n" << R.transpose() * R << std::endl;
    std::cout << "R RT: \n" << R * R.transpose() << std::endl;
    return 0;
}