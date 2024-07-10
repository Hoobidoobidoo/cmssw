#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// LST includes
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTModulesDevESProducer : public ESProducer {
  public:
    LSTModulesDevESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig), ptCutLabel_(iConfig.getParameter<std::string>("ptCutLabel")) {
      setWhatProduced(this, &LSTModulesDevESProducer::produce, ptCutLabel_);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ptCutLabel", "0.8");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<SDL::LSTESData<SDL::DevHost>> produce(TrackerRecoGeometryRecord const& iRecord) {
      return SDL::loadAndFillESHost(ptCutLabel_);
    }

  private:
    std::string ptCutLabel_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
