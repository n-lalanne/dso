//
// Created by sicong, rakesh on 23/08/17.
//

#include "FullSystem/CoarseTracker.h"
void printMatrixIndices(MatXX H)
{
    for (size_t i = 0; i < H.rows(); i++)
    {
        for (size_t j = 0; j < H.cols(); j++)
        {
            printf("(%02d,%02d) ", (u_int32_t)H(i, j)/100, (u_int32_t)H(i, j) % 100);
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    CoarseTracker coarseTracker(100, 100, NULL);

    Mat3232 H;
    Vec32 b;
    for (size_t i = 0; i < 32; i++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            H(i, j) = i*100 + j;
        }
        b(i) = i;
    }

    coarseTracker.setUnscaledGaussNewtonSys(H, b);

    Mat1515 Hbb;
    Mat1517 Hbm;
    Mat1717 Hmm;
    Vec15 bb;
    Vec17 bm;

    coarseTracker.splitHessianForMarginalization(
            Hbb, Hbm, Hmm,
            bb, bm
    );

    std::cout << "Hbb: \n"; printMatrixIndices(Hbb); std::cout << std::endl << std::endl;
    std::cout << "Hbm: \n"; printMatrixIndices(Hbm); std::cout << std::endl << std::endl;
    std::cout << "Hmm: \n"; printMatrixIndices(Hmm); std::cout << std::endl << std::endl;
    std::cout << "bb: \n" << bb << std::endl << std::endl;
    std::cout << "bm: \n" << bm << std::endl << std::endl;

    return 0;
}