/**
 * @file
 * @brief Particle Filter specific messages
 */

#include "float_compression.hpp"
#include "logger.hpp"
#include <utility>

// TODO: put all magics in one place

namespace vexmaps {
namespace logger {

template <size_t N> class Float16ParticlesLogger : public BaseTypeLogger {
private:
  float16_t x[N];
  float16_t y[N];
  float16_t weights[N];

  static constexpr char particleLoggerMagic = 0x41;

public:
  char getMagic2() override { return particleLoggerMagic; }

  void addParticles(float *x, float *y, float *weights, const size_t len,
                    const size_t offset = 0) {
    // end index in our array
    const size_t n = len + offset;
    assert((n <= N) && "given more elements than length of logger");

    const size_t min_n = std::min(N, n);

    // offset should not matter
    const size_t remaining_particles = (min_n - (min_n % 8));

    for (size_t i = offset; i < remaining_particles; i += 8) {
      // load in values
      float32x4_t vx1 = vld1q_f32(&x[i]);
      float32x4_t vy1 = vld1q_f32(&y[i]);
      float32x4_t vweights1 = vld1q_f32(&weights[i]);

      float32x4_t vx2 = vld1q_f32(&x[i + 4]);
      float32x4_t vy2 = vld1q_f32(&y[i + 4]);
      float32x4_t vweights2 = vld1q_f32(&weights[i + 4]);

      // convert to float16
      float16x4_t v16x1 = vcvt_f16_f32(vx1);
      float16x4_t v16y1 = vcvt_f16_f32(vy1);
      float16x4_t v16weights1 = vcvt_f16_f32(vweights1);

      float16x4_t v16x2 = vcvt_f16_f32(vx2);
      float16x4_t v16y2 = vcvt_f16_f32(vy2);
      float16x4_t v16weights2 = vcvt_f16_f32(vweights2);

      // load to internal arrays
      vst1_f16(&(this->x[i]), v16x1);
      vst1_f16(&(this->y[i]), v16y1);
      vst1_f16(&(this->weights[i]), v16weights1);
      vst1_f16(&(this->x[i + 4]), v16x2);
      vst1_f16(&(this->y[i + 4]), v16y2);
      vst1_f16(&(this->weights[i + 4]), v16weights2);
    }
    for (size_t i = remaining_particles; i < min_n; i++) {
      this->x[i] = x[i];
      this->y[i] = y[i];
      this->weights[i] = weights[i];
    }
  }

  void setParticle(const size_t i, float x, float y, float weight) {
    this->x[i] = x;
    this->y[i] = y;
    this->weights[i] = weight;
  }

  size_t LogData(LogBuffer *buffer) override {
    // TODO: might be able to compress further if we remove the last two
    // bits of the mantissa from x/y

    // total len
    size_t misc_len = 0;

    misc_len += buffer->write(getMagic1());
    misc_len += buffer->write(getMagic2());

    size_t data_len_ind = buffer->getIndex();

    // leave space for len
    buffer->advanceIndex(4);
    misc_len += 4;

    size_t data_len = 0;
    for (int i = 0; i < N; i++) {
      data_len += buffer->write(x[i]);
      data_len += buffer->write(y[i]);
      data_len += buffer->write(weights[i]);
    }

    buffer->write_index(data_len_ind, data_len);

    return misc_len + data_len;
  }

  // two bytes per particle
  size_t maxSize() override {
    return 2 + sizeof(uint32_t) + 3 * N * sizeof(float16_t);
  }

  ~Float16ParticlesLogger() override = default;
};

class DistanceSensorLogger : public CategoryLogger {
private:
  static constexpr char distanceInfoMagic = 0x42;

  std::vector<BaseMessageLogger *> children{&identifier, &measured_distance,
                                            &confidence, &object_size, &exit};

public:
  // TODO: make a string/char class to make this more clear
  UIntLogger identifier;
  FloatLogger measured_distance;
  UIntLogger confidence;
  UIntLogger object_size;
  BoolLogger exit;

  void setData(int identifier, float measured_distance, int confidence,
               int object_size, int exit) {
    this->identifier.setData(identifier);
    this->measured_distance.setData(measured_distance);
    this->confidence.setData(confidence);
    this->object_size.setData(object_size);
    this->exit.setData(exit);
  }

  char getMagic2() override { return distanceInfoMagic; }

  std::vector<BaseMessageLogger *> getChildren() override { return children; };

  size_t maxSize() override {
    size_t len = 0;
    for (auto curr : children) {
      len += curr->maxSize();
    }
    return len;
  }

  ~DistanceSensorLogger() override = default;
};

// implements a different logger for particles that instead uses int
// representations for floats as well as varints
template <size_t N> class VarintParticlesLogger : public BaseTypeLogger {
private:
  int16_t x[N];
  int16_t y[N];
  int16_t weights[N];
  std::pair<float, float> x_bounds;
  std::pair<float, float> y_bounds;
  std::pair<float, float> weights_bounds;

  static constexpr char particleLoggerMagic = 0x49;

public:
  char getMagic2() override { return particleLoggerMagic; }

  // TODO: switch to maybe only doing the delta encoding when we are building
  // the message to enable point updates
  void addParticles(float *x, float *y, float *weights, const size_t len) {
    // since we rely on delta encoding we must have all the values right now
    assert((len == N) && "must give the same amount of particles");

    // calculate bounds
    x_bounds.first = x_bounds.second = x[0];
    y_bounds.first = y_bounds.second = y[0];
    weights_bounds.first = weights_bounds.second = weights[0];

    // TODO: vectorize
    for (int i = 1; i < N; i++) {
      x_bounds.first = std::min(x[i], x_bounds.first);
      x_bounds.second = std::max(x[i], x_bounds.second);

      y_bounds.first = std::min(y[i], y_bounds.first);
      y_bounds.second = std::max(y[i], y_bounds.second);

      weights_bounds.first = std::min(weights[i], weights_bounds.first);
      weights_bounds.second = std::max(weights[i], weights_bounds.second);
    }

    // 140.4/2^9 = 0.27 -> error of 0.27 inches is probably fine
    compress_floats(x, this->x, N, x_bounds.first, x_bounds.second, 1 << 6);
    compress_floats(y, this->y, N, y_bounds.first, y_bounds.second, 1 << 6);

    // weights have to be somewhat precise because they do have a high range
    // however it could greatly benefit from delta's since most weights will be
    // small
    compress_floats(weights, this->weights, N, weights_bounds.first,
                    weights_bounds.second);

    // compress_floats(x, this->x, N, -1.78308, 1.78308);
    // compress_floats(y, this->y, N, -1.78308, 1.78308);
    // compress_floats(weights, this->weights, N, 0,
    //                 200);
  }

  size_t LogData(LogBuffer *buffer) override {
    // total len
    size_t misc_len = 0;

    misc_len += buffer->write(getMagic1());
    misc_len += buffer->write(getMagic2());

    size_t data_len_ind = buffer->getIndex();

    // leave space for len
    buffer->advanceIndex(4);
    misc_len += 4;

    size_t data_len = 0;
    // write bounds for each category
    data_len = buffer->write(x_bounds.first);
    data_len = buffer->write(x_bounds.second);
    data_len = buffer->write(y_bounds.first);
    data_len = buffer->write(y_bounds.second);
    data_len = buffer->write(weights_bounds.first);
    data_len = buffer->write(weights_bounds.second);

    for (int i = 0; i < N; i++) {
      data_len += buffer->write_varint(x[i]);
    }
    for (int i = 0; i < N; i++) {
      data_len += buffer->write_varint(y[i]);
    }
    for (int i = 0; i < N; i++) {
      data_len += buffer->write_varint(weights[i]);
    }

    buffer->write_index(data_len_ind, data_len);

    return misc_len + data_len;
  }

  // at most three bytes per particle
  size_t maxSize() override {
    return 2 * sizeof(char) +     // magic
           1 * sizeof(uint32_t) + // len
           6 * sizeof(float) +    // bounds
           3 * N * 3;             // particles
  }

  ~VarintParticlesLogger() override = default;
};

// TODO: make it possible to dynamically add/remove distance sensors
// should not be too hard to implement
class GenerationInfoLogger : public CategoryLogger {
private:
  static constexpr char generationInfoMagic = 0x40;

  std::vector<BaseMessageLogger *> children{
      &timestamp, &time_taken, &prediction, &distance1,
      &distance2, &distance3,  &distance4};

public:
  UIntLogger timestamp;
  UIntLogger time_taken;
  PoseLogger prediction;
  DistanceSensorLogger distance1;
  DistanceSensorLogger distance2;
  DistanceSensorLogger distance3;
  DistanceSensorLogger distance4;

  char getMagic2() override { return generationInfoMagic; }

  void setData(int timestamp, int time_taken, float px, float py, float pz) {
    this->timestamp.setData(timestamp);
    this->time_taken.setData(time_taken);
    this->prediction.setData(px, py, pz);
  }

  std::vector<BaseMessageLogger *> getChildren() override { return children; };

  size_t maxSize() override {
    size_t len = 0;
    for (auto curr : children) {
      len += curr->maxSize();
    }
    return len;
  }

  ~GenerationInfoLogger() override = default;
};

/**
 * @brief Holds all the information being printed by the PF
 */
template <size_t N> class PFLogger : public CategoryLogger {
private:
  std::vector<BaseMessageLogger *> children{&generation_info, &particles};
  static constexpr char PFMagic = 0xaf;

public:
  GenerationInfoLogger generation_info;
  VarintParticlesLogger<N> particles;

  char getMagic2() override { return PFMagic; }

  std::vector<BaseMessageLogger *> getChildren() override { return children; }

  size_t maxSize() override {
    size_t len = 0;
    for (auto curr : children) {
      len += curr->maxSize();
    }
    return len;
  }

  ~PFLogger() override = default;
};
} // namespace logger
} // namespace vexmaps
