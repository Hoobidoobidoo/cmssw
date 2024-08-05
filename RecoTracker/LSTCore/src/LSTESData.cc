#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ModuleConnectionMap.h"
#include "RecoTracker/LSTCore/interface/TiltedGeometry.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "ModuleMethods.h"

namespace {
  std::string trackLooperDir() {
    std::string path_str, path;
    const char* path_tracklooperdir = std::getenv("TRACKLOOPERDIR");
    std::stringstream search_path(std::getenv("CMSSW_SEARCH_PATH"));

    while (std::getline(search_path, path, ':')) {
      if (std::filesystem::exists(path + "/RecoTracker/LSTCore/data")) {
        path_str = path;
        break;
      }
    }

    if (path_str.empty()) {
      path_str = path_tracklooperdir;
      path_str += "/..";
    } else {
      path_str += "/RecoTracker/LSTCore";
    }

    return path_str;
  }

  std::string get_absolute_path_after_check_file_exists(std::string const& name) {
    std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
    if (not std::filesystem::exists(fullpath)) {
      throw std::runtime_error("Could not find the file = " + fullpath.string());
    }
    return fullpath.string();
  }

  void loadMapsHost(lst::MapPLStoLayer& pLStoLayer,
                    std::shared_ptr<lst::EndcapGeometry> endcapGeometry,
                    std::shared_ptr<lst::TiltedGeometry> tiltedGeometry,
                    std::shared_ptr<lst::ModuleConnectionMap> moduleConnectionMap,
                    std::string& ptCutLabel) {
    // Module orientation information (DrDz or phi angles)
    auto endcap_geom = get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt" +
                                                                 ptCutLabel + "/endcap_orientation.bin");
    auto tilted_geom = get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt" +
                                                                 ptCutLabel + "/tilted_barrel_orientation.bin");
    // Module connection map (for line segment building)
    auto mappath = get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt" + ptCutLabel +
                                                             "/module_connection_tracing_merged.bin");

    endcapGeometry->load(endcap_geom);
    tiltedGeometry->load(tilted_geom);
    moduleConnectionMap->load(mappath);

    auto pLSMapDir = trackLooperDir() + "/data/OT800_IT615_pt" + ptCutLabel + "/pixelmap/pLS_map";
    const std::array<std::string, 4> connects{
        {"_layer1_subdet5", "_layer2_subdet5", "_layer1_subdet4", "_layer2_subdet4"}};
    std::string path;

    static_assert(connects.size() == std::tuple_size<std::decay_t<decltype(pLStoLayer[0])>>{});
    for (unsigned int i = 0; i < connects.size(); i++) {
      auto connectData = connects[i].data();

      path = pLSMapDir + connectData + ".bin";
      pLStoLayer[0][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_pos" + connectData + ".bin";
      pLStoLayer[1][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));

      path = pLSMapDir + "_neg" + connectData + ".bin";
      pLStoLayer[2][i] = lst::ModuleConnectionMap(get_absolute_path_after_check_file_exists(path));
    }
  }
}  // namespace

std::unique_ptr<lst::LSTESData<alpaka_common::DevHost>> lst::loadAndFillESHost(std::string& ptCutLabel) {
  uint16_t nModules;
  uint16_t nLowerModules;
  unsigned int nPixels;
  std::shared_ptr<lst::ModulesBuffer<alpaka_common::DevHost>> modulesBuffers = nullptr;
  auto pLStoLayer = std::make_shared<MapPLStoLayer>();
  auto endcapGeometry = std::make_shared<EndcapGeometry>();
  auto tiltedGeometry = std::make_shared<TiltedGeometry>();
  auto pixelMapping = std::make_shared<PixelMap>();
  auto moduleConnectionMap = std::make_shared<ModuleConnectionMap>();
  ::loadMapsHost(*pLStoLayer, endcapGeometry, tiltedGeometry, moduleConnectionMap, ptCutLabel);

  auto endcapGeometryBuffers = std::make_shared<EndcapGeometryBuffer<alpaka_common::DevHost>>(
      cms::alpakatools::host(), endcapGeometry->nEndCapMap);
  alpaka::QueueCpuBlocking queue(cms::alpakatools::host());
  alpaka::memcpy(
      queue, endcapGeometryBuffers->geoMapDetId_buf, endcapGeometry->geoMapDetId_buf, endcapGeometry->nEndCapMap);
  alpaka::memcpy(
      queue, endcapGeometryBuffers->geoMapPhi_buf, endcapGeometry->geoMapPhi_buf, endcapGeometry->nEndCapMap);

  auto path = get_absolute_path_after_check_file_exists(trackLooperDir() + "/data/OT800_IT615_pt" + ptCutLabel +
                                                        "/sensor_centroids.bin");
  lst::loadModulesFromFile(pLStoLayer.get(),
                           path.c_str(),
                           nModules,
                           nLowerModules,
                           nPixels,
                           modulesBuffers,
                           pixelMapping.get(),
                           endcapGeometry.get(),
                           tiltedGeometry.get(),
                           moduleConnectionMap.get());
  return std::make_unique<LSTESData<alpaka_common::DevHost>>(
      nModules, nLowerModules, nPixels, endcapGeometry->nEndCapMap, modulesBuffers, endcapGeometryBuffers, pixelMapping);
}
