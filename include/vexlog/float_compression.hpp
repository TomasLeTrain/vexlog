#pragma once
#include <arm_neon.h>
#include <memory>
#include <vector>

namespace vexmaps {
namespace logger {
/**
 * @brief transforms floats within the range [-1.78308,1.78308] into a list of
 * differences of unsigned within the range [0,65536]. Intended for floats
 * representing points on a vex field
 *
 * @param data float data
 * @param result where the results get stored
 * @param len number of elements
 */
inline void compress_floats(float *data, uint16_t *result, size_t len) {
  // 140.1(wall length) * 467.78 = 65536
  const float c0 = (467.780157031) / 0.0254;
  const float c1 = 35100;

  const size_t remaining_floats = len - (len % 16);

  float32x4_t vc0 = vdupq_n_f32(c0);
  float32x4_t vc1 = vdupq_n_f32(c1);

  for (int i = 0; i < remaining_floats; i += 16) {
    float32x4_t v1 = vld1q_f32(&data[i]);
    float32x4_t v2 = vld1q_f32(&data[i + 4]);
    float32x4_t v3 = vld1q_f32(&data[i + 8]);
    float32x4_t v4 = vld1q_f32(&data[i + 12]);

    v1 = vmlaq_f32(v1, vc0, vc1);
    v2 = vmlaq_f32(v2, vc0, vc1);
    v3 = vmlaq_f32(v3, vc0, vc1);
    v4 = vmlaq_f32(v4, vc0, vc1);

    int32x4_t converted1 = vcvtq_u32_f32(v1);
    int32x4_t converted2 = vcvtq_u32_f32(v2);
    int32x4_t converted3 = vcvtq_u32_f32(v3);
    int32x4_t converted4 = vcvtq_u32_f32(v4);

    uint16x4_t final1 = vreinterpret_u16_s16(vmovn_s32(converted1));
    uint16x4_t final2 = vreinterpret_u16_s16(vmovn_s32(converted2));
    uint16x4_t final3 = vreinterpret_u16_s16(vmovn_s32(converted3));
    uint16x4_t final4 = vreinterpret_u16_s16(vmovn_s32(converted4));
    // store results
    vst1_u16(result + i, final1);
    vst1_u16(result + i + 4, final1);
    vst1_u16(result + i + 8, final1);
    vst1_u16(result + i + 12, final1);
  }
  // go through remaining particles if neccesary
  for (int i = remaining_floats; i < len; i++) {
    result[i] = static_cast<uint16_t>(data[i] * c0 + c1);
  }

  // we assume particles will be vaguely near each other, so we can try and use
  // delta encoding to reduce their sizes

  // first element left alone
  // TODO: could vectorize if an intermediate array was used
  int last = result[0];
  for (int i = 1; i < len; i++) {
    int curr = result[i];
    result[i] = curr - last;
    last = curr;
  }
}
} // namespace logger
} // namespace vexmaps
