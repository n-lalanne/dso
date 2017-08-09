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
class PhotometricFactor : public NoiseModelFactor1<Pose3> {

private:
    CoarseTracker *coarseTracker_;
    int lvl_;
    int pointIdx_;

public:
    PhotometricFactor(Key j,
                      CoarseTracker *coarseTracker,
                      int lvl, int pointIdx,
                      const SharedNoiseModel& model);

    Vector evaluateError(const Pose3& pose,
                         boost::optional<Matrix&> H = boost::none) const;
};

} // namespace dso

#endif //DSO_PHOTOMETRICFACTOR_H
