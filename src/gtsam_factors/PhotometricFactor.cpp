//
// Created by sicong on 08/08/17.
//

#include "PhotometricFactor.h"

PhotometricFactor::PhotometricFactor(   Key j, CoarseTracker *coarseTracker,
                                        int lvl, int pointIdx, AffLight aff_g2l, float cutoffTH,
                                        const SharedNoiseModel& model
                                    )
    : NoiseModelFactor1<Pose3>(model, j),
      coarseTracker_(coarseTracker),
      lvl_(lvl), pointIdx_(pointIdx), aff_g2l_(aff_g2l), cutoffThreshold_(cutoffTH)
{

}

Vector PhotometricFactor::evaluateError (    const Pose3& pose,
                                             boost::optional<Matrix&> H
                                        ) const
{
    coarseTracker_->calcPointResIMU(
            lvl_, pointIdx_,
            gtsam::NavState(pose, coarseTracker_->newFrame->shell->navstate.velocity()),
            aff_g2l_, cutoffThreshold_
    );

    // outlier, ignore
    if (coarseTracker_->buf_warped_weight[pointIdx_] == 0)
    {
        if (H)
        {
            *H = Vec6::Zero().transpose();
        }
        return (Vector(1) << 0).finished();
    }

    if (H)
    {
        SE3 Tib(pose.matrix());
        SE3 Tw_reffromNAV = SE3(coarseTracker_->lastRef->shell->navstate.pose().matrix()) *
                            coarseTracker_->imutocam().inverse();

        SE3 Tw_ref = coarseTracker_->lastRef->shell->camToWorld;
        Mat33 Rcb = coarseTracker_->imutocam().rotationMatrix();
        SE3 Trb = Tw_ref.inverse() * Tib;

        Vec3 Pr,PI;
        Vec6 Jab;
        Vec2 dxdy;
        dxdy(0) = *(coarseTracker_->buf_warped_dx + pointIdx_);
        dxdy(1)	= *(coarseTracker_->buf_warped_dy + pointIdx_);
//        float b0 = lastRef_aff_g2l.b;
//        float a = (float)(AffLight::fromToVecExposure(coarseTracker_->lastRef->ab_exposure, coarseTracker_->newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]);

        float id = *(coarseTracker_->buf_warped_idepth + pointIdx_);
        float u = *(coarseTracker_->buf_warped_u + pointIdx_);
        float v = *(coarseTracker_->buf_warped_v + pointIdx_);
        Pr(0) = *(coarseTracker_->buf_warped_rx + pointIdx_);
        Pr(1) = *(coarseTracker_->buf_warped_ry + pointIdx_);
        Pr(2) = *(coarseTracker_->buf_warped_rz + pointIdx_);
        PI = Tw_ref * Pr;
        // Jacobian of camera projection
        Matrix23 Maux;
        Maux.setZero();
        Maux(0,0) = coarseTracker_->fx[lvl_];
        Maux(0,1) = 0;
        Maux(0,2) = -u *coarseTracker_->fx[lvl_];
        Maux(1,0) = 0;
        Maux(1,1) = coarseTracker_->fy[lvl_];
        Maux(1,2) = -v * coarseTracker_->fy[lvl_];

        Matrix23 Jpi = Maux * id;

        Jab.head(3) = dxdy.transpose() * Jpi * Rcb * SO3::hat(Tib.inverse() * PI); //Rrb.transpose()*(Pr-Prb));
        Jab.tail(3) = dxdy.transpose() * Jpi * (-Rcb);

        *H = (gtsam::Vector6() << Jab(0), Jab(1), Jab(2), Jab(3), Jab(4), Jab(5)).finished().transpose();
//        *H = Jab.transpose();
//        if (pointIdx_ % 100 == 0)
//        {
//            std::cout << "Point idx: " << pointIdx_ << std::endl;
//            std::cout << "H: \n" << *H << std::endl;
//            std::cout << "E: " << (Vector(1) << coarseTracker_->buf_warped_residual[pointIdx_]).finished().transpose() << std::endl;
//        }
    }
    return (Vector(1) << (double)coarseTracker_->buf_warped_residual[pointIdx_]).finished();
}