#ifndef RecoTracker_LST_LSTPixelSeedInput_h
#define RecoTracker_LST_LSTPixelSeedInput_h

#include <memory>
#include <vector>

class LSTPixelSeedInput {
public:
  LSTPixelSeedInput() = default;
  ~LSTPixelSeedInput() = default;

  void setLSTPixelSeedTraits(std::vector<float> px,
                             std::vector<float> py,
                             std::vector<float> pz,
                             std::vector<float> dxy,
                             std::vector<float> dz,
                             std::vector<float> ptErr,
                             std::vector<float> etaErr,
                             std::vector<float> stateTrajGlbX,
                             std::vector<float> stateTrajGlbY,
                             std::vector<float> stateTrajGlbZ,
                             std::vector<float> stateTrajGlbPx,
                             std::vector<float> stateTrajGlbPy,
                             std::vector<float> stateTrajGlbPz,
                             std::vector<int> q,
                             std::vector<unsigned int> algo,
                             std::vector<std::vector<int>> hitIdx) {
    px_ = px;
    py_ = py;
    pz_ = pz;
    dxy_ = dxy;
    dz_ = dz;
    ptErr_ = ptErr;
    etaErr_ = etaErr;
    stateTrajGlbX_ = stateTrajGlbX;
    stateTrajGlbY_ = stateTrajGlbY;
    stateTrajGlbZ_ = stateTrajGlbZ;
    stateTrajGlbPx_ = stateTrajGlbPx;
    stateTrajGlbPy_ = stateTrajGlbPy;
    stateTrajGlbPz_ = stateTrajGlbPz;
    q_ = q;
    algo_ = algo;
    hitIdx_ = hitIdx;
  }

  std::vector<float> const& px() const { return px_; }
  std::vector<float> const& py() const { return py_; }
  std::vector<float> const& pz() const { return pz_; }
  std::vector<float> const& dxy() const { return dxy_; }
  std::vector<float> const& dz() const { return dz_; }
  std::vector<float> const& ptErr() const { return ptErr_; }
  std::vector<float> const& etaErr() const { return etaErr_; }
  std::vector<float> const& stateTrajGlbX() const { return stateTrajGlbX_; }
  std::vector<float> const& stateTrajGlbY() const { return stateTrajGlbY_; }
  std::vector<float> const& stateTrajGlbZ() const { return stateTrajGlbZ_; }
  std::vector<float> const& stateTrajGlbPx() const { return stateTrajGlbPx_; }
  std::vector<float> const& stateTrajGlbPy() const { return stateTrajGlbPy_; }
  std::vector<float> const& stateTrajGlbPz() const { return stateTrajGlbPz_; }
  std::vector<int> const& q() const { return q_; }
  std::vector<unsigned int> const& algo() const { return algo_; }
  std::vector<std::vector<int>> const& hitIdx() const { return hitIdx_; }

private:
  std::vector<float> px_;
  std::vector<float> py_;
  std::vector<float> pz_;
  std::vector<float> dxy_;
  std::vector<float> dz_;
  std::vector<float> ptErr_;
  std::vector<float> etaErr_;
  std::vector<float> stateTrajGlbX_;
  std::vector<float> stateTrajGlbY_;
  std::vector<float> stateTrajGlbZ_;
  std::vector<float> stateTrajGlbPx_;
  std::vector<float> stateTrajGlbPy_;
  std::vector<float> stateTrajGlbPz_;
  std::vector<int> q_;
  std::vector<unsigned int> algo_;
  std::vector<std::vector<int>> hitIdx_;
};

#endif

