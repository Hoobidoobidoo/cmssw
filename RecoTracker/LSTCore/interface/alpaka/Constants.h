#ifndef RecoTracker_LSTCore_interface_alpaka_Constants_h
#define RecoTracker_LSTCore_interface_alpaka_Constants_h

#include "RecoTracker/LSTCore/interface/Constants.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

namespace lst {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Half precision wrapper functions.
#if defined(FP16_Base)
#define __F2H __float2half
#define __H2F __half2float
  typedef __half float FPX;
#else
#define __F2H
#define __H2F
  typedef float FPX;
#endif

  Vec3D constexpr elementsPerThread(Vec3D::all(static_cast<Idx>(1)));

// Needed for files that are compiled by g++ to not throw an error.
// uint4 is defined only for CUDA, so we will have to revisit this soon when running on other backends.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
#endif

  // Wrapper function to reduce code boilerplate for defining grid/block sizes.
  ALPAKA_FN_HOST ALPAKA_FN_INLINE Vec3D createVec(int x, int y, int z) {
    return Vec3D(static_cast<Idx>(x), static_cast<Idx>(y), static_cast<Idx>(z));
  }

  // Adjust grid and block sizes based on backend configuration
  template <typename Vec>
  ALPAKA_FN_HOST ALPAKA_FN_INLINE WorkDiv3D createWorkDiv(const Vec& blocksPerGrid,
                                                          const Vec& threadsPerBlock,
                                                          const Vec& elementsPerThreadArg) {
    Vec adjustedBlocks = blocksPerGrid;
    Vec adjustedThreads = threadsPerBlock;

    // Serial execution, so all launch parameters set to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    adjustedBlocks = Vec::all(static_cast<Idx>(1));
    adjustedThreads = Vec::all(static_cast<Idx>(1));
#endif

    // Threads enabled, set number of blocks to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    adjustedBlocks = Vec::all(static_cast<Idx>(1));
#endif

    return WorkDiv3D(adjustedBlocks, adjustedThreads, elementsPerThreadArg);
  }

  // 15 MeV constant from the approximate Bethe-Bloch formula
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMulsInGeV = 0.015;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float miniMulsPtScaleBarrel[6] = {
      0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float miniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float miniRminMeanBarrel[6] = {
      25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float miniRminMeanEndcap[5] = {
      130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR1GeVf = 1. / (2.99792458e-3 * 3.8);
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float sinAlphaMax = 0.95;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float ptCut = PT_CUT;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float deltaZLum = 15.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float pixelPSZpitch = 0.15;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float stripPSZpitch = 2.4;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float strip2SZpitch = 5.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float width2S = 0.009;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float widthPS = 0.01;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float pt_betaMax = 7.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float magnetic_field = 3.8112;
  // Since C++ can't represent infinity, lst_INF = 123456789 was used to represent infinity in the data table
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float lst_INF = 123456789.0;

  namespace t5dnn {

    // Working points matching LST fake rate (43.9%) or signal acceptance (82.0%)
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float lstwp1 = 0.3418833f;  // 94.0% TPR, 43.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float lstwp2 = 0.6177366f;  // 82.0% TPR, 20.0% FPR
    // Other working points
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp70 = 0.7776195f;    // 70.0% TPR, 10.0% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp75 = 0.7181118f;    // 75.0% TPR, 13.5% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp80 = 0.6492643f;    // 80.0% TPR, 17.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp85 = 0.5655319f;    // 85.0% TPR, 23.8% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp90 = 0.4592205f;    // 90.0% TPR, 32.6% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp95 = 0.3073708f;    // 95.0% TPR, 47.7% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp97p5 = 0.2001348f;  // 97.5% TPR, 61.2% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp99 = 0.1120605f;    // 99.0% TPR, 75.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float wp99p9 = 0.0218196f;  // 99.9% TPR, 95.4% FPR

  }  // namespace t5dnn

}  //namespace lst
#endif
