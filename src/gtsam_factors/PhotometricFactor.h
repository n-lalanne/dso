//
// Created by sicong on 08/08/17.
//

#ifndef DSO_PHOTOMETRICFACTOR_H
#define DSO_PHOTOMETRICFACTOR_H

/* GTSAM includes */
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

#include "FullSystem/CoarseTracker.h"

namespace dso {

using namespace gtsam;
class GTSAM_EXPORT PhotometricFactor : public NoiseModelFactor1<Pose3> {

private:
    CoarseTracker *coarseTracker_;
    int lvl_;
    int pointIdx_;
    AffLight aff_g2l_;
    float cutoffThreshold_;

public:
    PhotometricFactor(Key j,
                      CoarseTracker *coarseTracker,
                      int lvl, int pointIdx, AffLight aff_g2l, float cutoffTH,
                      const SharedNoiseModel& model);

    virtual ~PhotometricFactor() {}

    Vector evaluateError(const Pose3& pose,
                         boost::optional<Matrix&> H = boost::none) const;

    virtual gtsam::NonlinearFactor::shared_ptr clone() const {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
                gtsam::NonlinearFactor::shared_ptr(new PhotometricFactor(*this)));
    }
};

} // namespace dso

#endif //DSO_PHOTOMETRICFACTOR_H
