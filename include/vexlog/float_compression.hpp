#pragma once
#include <arm_neon.h>
#include <memory>
#include <vector>

namespace vexmaps {
namespace logger {
/**
 * @brief transforms floats within the range [a,b] into a list of
 * differences of unsigned within the range [0,2^13 / mod]
 *
 * @param data float data
 * @param result where the results get stored
 * @param len number of elements
 */
inline uint32_t compress_floats(float *data, int16_t *result, size_t len,
                                float a, float b, int mod = (1 << 13)) {
  //  a/2^15
  const float c0 = static_cast<float>(mod) / (b - a);
  // r = (x - a) * c0
  // r = x * c0 - a * c0
  // r = (-a * c0) + x * c0
  // r = c2 + x * c0
  const float c1 = (-a) * c0;

  const size_t remaining_floats = len - (len % 16);

  float32x4_t vc0 = vdupq_n_f32(c0);
  float32x4_t vc1 = vdupq_n_f32(c1);

  for (int i = 0; i < remaining_floats; i += 16) {
    float32x4_t v1 = vld1q_f32(&data[i]);
    float32x4_t v2 = vld1q_f32(&data[i + 4]);
    float32x4_t v3 = vld1q_f32(&data[i + 8]);
    float32x4_t v4 = vld1q_f32(&data[i + 12]);

    v1 = vmlaq_f32(vc1, v1, vc0);
    v2 = vmlaq_f32(vc1, v2, vc0);
    v3 = vmlaq_f32(vc1, v3, vc0);
    v4 = vmlaq_f32(vc1, v4, vc0);

    int32x4_t converted1 = vcvtq_s32_f32(v1);
    int32x4_t converted2 = vcvtq_s32_f32(v2);
    int32x4_t converted3 = vcvtq_s32_f32(v3);
    int32x4_t converted4 = vcvtq_s32_f32(v4);

    int16x4_t final1 = vmovn_s32(converted1);
    int16x4_t final2 = vmovn_s32(converted2);
    int16x4_t final3 = vmovn_s32(converted3);
    int16x4_t final4 = vmovn_s32(converted4);
    // store results
    vst1_s16(result + i, final1);
    vst1_s16(result + i + 4, final2);
    vst1_s16(result + i + 8, final3);
    vst1_s16(result + i + 12, final4);
  }
  // go through remaining particles if neccesary
  for (int i = remaining_floats; i < len; i++) {
    result[i] = static_cast<int16_t>(c1 + data[i] * c0);
  }

  // we assume particles will be vaguely near each other, so we can try and use
  // delta encoding to reduce their sizes

  // first element left alone
  int16_t last = result[0];

  // TODO: could vectorize if an intermediate array was used
  for (int i = 1; i < len; i++) {
    int16_t curr = result[i];
    // differences can be negative
    result[i] = curr - last;
    last = curr;
  }

  return mod;
}
} // namespace logger
} // namespace vexmaps
