#include "pros/apix.h"
#include <arm_neon.h>
#include <cassert>
#include <cstring>
#include <vector>

namespace vexmaps {
namespace logger {

// message:
// we asssume both sides know the data types - magics
//
// each [] is a byte, each () is a varint
//
// [message magic 1][message magic 2](size in bytes)[child1][child2]...
// [children magic1][children magic2](size in bytes)[child1][child2]...
// ...

// vectors might still be slow
// TODO: add error checking
class LogBuffer {
private:
  std::vector<char> buf;
  // index must always be the size of the vector!
  size_t ind = 0;
  void setIndex(size_t i) { ind = i; }

public:
  LogBuffer(size_t len) : buf(len) {}

  std::vector<char> &getVector() { return buf; }

  inline void writeByte(char data) {
    buf[ind] = data;
    ind++;
  }

  void advanceIndex(size_t offset = 1) { ind += offset; }

  size_t getIndex() { return ind; }

  size_t write(char *data, int len) {
    for (int i = 0; i < len; i++)
      writeByte(data[i]);
    return len;
  }

  template <typename T> size_t write(T data) {
    char *char_data = reinterpret_cast<char *>(&data);
    for (int i = 0; i < sizeof(data); i++) {
      writeByte(char_data[i]);
    }
    return sizeof(data);
  }

  /**
   * @brief Writes data at specific index
   */
  template <typename... Args> void write_index(int i, Args... args) {
    size_t currentIndex = getIndex();
    setIndex(i);
    write(args...);
    setIndex(currentIndex);
  }
};

class BaseMessageLogger {
public:
  // this should be a general magic for the type of message (category, data
  // type, etc.)
  virtual char getMagic1() = 0;
  // this should be specific to the field
  virtual char getMagic2() = 0;

  virtual bool IsData() = 0;

  virtual size_t LogData(LogBuffer *buffer) = 0;
  virtual std::vector<BaseMessageLogger *> getChildren() = 0;

  virtual ~BaseMessageLogger() = default;
};

class CategoryLogger : public BaseMessageLogger {
private:
  static constexpr char basicDataTypeMagic = 0x70;

public:
  char getMagic1() override { return basicDataTypeMagic; }

  bool IsData() override { return false; }

  // since its a structure this should never get called
  size_t LogData(LogBuffer *buffer) override { return 0; }
};

class BaseTypeLogger : public BaseMessageLogger {
  static constexpr char basicDataTypeMagic = 0x71;

public:
  char getMagic1() override { return basicDataTypeMagic; }
  bool IsData() override { return true; }

  // this should also never get called
  std::vector<BaseMessageLogger *> getChildren() override { return {}; };
};

class BoolLogger : public BaseTypeLogger {
private:
  bool data;
  static constexpr char boolOffMagic = 0x14;
  static constexpr char boolOnMagic = 0x15;

public:
  BoolLogger() {}
  BoolLogger(bool data) : data(data) {}

  void setData(bool data) { this->data = data; }

  char getMagic2() override {
    if (data)
      return boolOnMagic;
    else
      return boolOffMagic;
  }

  // no data other than the magic
  size_t LogData(LogBuffer *buffer) override {
    return buffer->write(getMagic2());
  }

  ~BoolLogger() override = default;
};

class IntLogger : public BaseTypeLogger {
private:
  int data;
  static constexpr char intMagic = 0x11;

public:
  IntLogger() {}
  IntLogger(int data) : data(data) {}

  void setData(int data) { this->data = data; }

  char getMagic2() override { return intMagic; }

  size_t LogData(LogBuffer *buffer) override {
    // for now we dont add the first magic since this a basic type
    size_t len = 0;
    len += buffer->write(getMagic2());
    len += buffer->write(data);
    return len;
  }

  ~IntLogger() override = default;
};

class FloatLogger : public BaseTypeLogger {
private:
  float data;
  static constexpr char floatMagic = 0x12;

public:
  FloatLogger() {}
  FloatLogger(float data) : data(data) {}

  void setData(float data) { this->data = data; }

  char getMagic2() override { return floatMagic; }

  size_t LogData(LogBuffer *buffer) override {
    size_t len = 0;
    // for now we dont add the first magic since this a basic type
    len += buffer->write(getMagic2());
    len += buffer->write(data);
    return len;
  }

  ~FloatLogger() override = default;
};

class PoseLogger : public BaseTypeLogger {
private:
  float x;
  float y;
  float z;
  static constexpr char poseMagic = 0x13;

public:
  PoseLogger() {}
  PoseLogger(float x, float y, float z) : x(x), y(y), z(z) {}

  void setData(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  char getMagic2() override { return poseMagic; }

  size_t LogData(LogBuffer *buffer) override {
    size_t len = 0;
    // for now we dont add the first magic since this a basic type
    len += buffer->write(getMagic2());
    len += buffer->write(x);
    len += buffer->write(y);
    len += buffer->write(z);
    return len;
  }

  ~PoseLogger() override = default;
};

template <size_t N> class ParticlesLogger : public BaseTypeLogger {
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
    const size_t remaining_particles = (min_n - (min_n % 4));

    for (size_t i = offset; i < remaining_particles; i += 4) {
      // load in values
      float32x4_t vx = vld1q_f32(&x[i]);
      float32x4_t vy = vld1q_f32(&y[i]);
      float32x4_t vweights = vld1q_f32(&weights[i]);

      // convert to float16
      float16x4_t v16x = vcvt_f16_f32(vx);
      float16x4_t v16y = vcvt_f16_f32(vy);
      float16x4_t v16weights = vcvt_f16_f32(vweights);

      // load to internal arrays
      vst1_f16(&(this->x[i]), v16x);
      vst1_f16(&(this->y[i]), v16y);
      vst1_f16(&(this->weights[i]), v16weights);
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

  ~ParticlesLogger() override = default;
};

class DistanceSensorLogger : public CategoryLogger {
private:
  static constexpr char distanceInfoMagic = 0x42;

  std::vector<BaseMessageLogger *> children{&identifier, &measured_distance,
                                            &confidence, &object_size, &exit};

public:
  // TODO: make a string/char class to make this more clear
  IntLogger identifier;
  FloatLogger measured_distance;
  IntLogger confidence;
  IntLogger object_size;
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

  ~DistanceSensorLogger() override = default;
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
  IntLogger timestamp;
  IntLogger time_taken;
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
  ParticlesLogger<N> particles;

  char getMagic2() override { return PFMagic; }

  std::vector<BaseMessageLogger *> getChildren() override { return children; }

  ~PFLogger() override = default;
};

// traversing list in dfs order
// assumes list is getting copied
inline size_t buildData(BaseMessageLogger *current_message, LogBuffer *buffer) {
  auto children = current_message->getChildren();

  if (current_message->IsData()) {
    // not a structure, just data
    return current_message->LogData(buffer);
  }

  // size of magics and data len
  size_t misc_len = 0;

  misc_len += buffer->write(current_message->getMagic1());
  misc_len += buffer->write(current_message->getMagic2());

  size_t data_len_ind = buffer->getIndex();
  // leave space for length
  buffer->advanceIndex(4);
  misc_len += 4;

  size_t data_len = 0;

  for (auto curr : children) {
    data_len += buildData(curr, buffer);
  }

  // TODO: switch to varint's
  buffer->write_index(data_len_ind, data_len);

  // total size of the message incuding magics and len data
  return data_len + misc_len;
}

inline void sendData(BaseMessageLogger *message, size_t buffer_size) {
  auto start_time = pros::c::micros();
  LogBuffer buf(buffer_size);
  size_t final_size = buildData(message, &buf);
  auto end_time = pros::c::micros();

  // fine to use endl since we are sending all the data at once
  auto send_start_time = pros::c::micros();
  std::cout.write(buf.getVector().data(), final_size);
  std::cout << std::endl;
  auto send_end_time = pros::c::micros();

  std::cout << "total construction time: " << end_time - start_time
            << ", sending time: " << send_end_time - send_start_time
            << std::endl;
}

} // namespace logger
} // namespace vexmaps
