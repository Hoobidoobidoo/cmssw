#ifndef RecoTracker_LSTCore_interface_LSTInputHostCollection_h
#define RecoTracker_LSTCore_interface_LSTInputHostCollection_h

#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using LSTInputHostCollection = PortableHostMultiCollection<InputHitsSoA, InputPixelHitsSoA, InputPixelSeedsSoA>;
}  // namespace lst
#endif
