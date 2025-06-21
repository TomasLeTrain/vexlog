/**
 * @file
 * @brief Base classes and some basic type implementations for the serializer
 */

#pragma once

#include "pros/apix.h"
#include <arm_neon.h>
#include <cassert>
#include <cstring>
#include <type_traits>
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

  // code modified from https://github.com/tidwall/varint.c
  template <typename T>
    requires std::is_unsigned_v<T>
  size_t write_varint(T data) {
    if (data < 128) {
      writeByte(data);
      return 1;
    }
    int n = 0;
    do {
      writeByte((uint8_t)data | 128);
      n++;
      data >>= 7;
    } while (data >= 128);
    writeByte((uint8_t)data);
    n++;
    return n;
  }

  template <typename T>
    requires std::is_signed_v<T>
  size_t write_varint(T data) {
    uint64_t ux = (uint64_t)data << 1;
    ux = data < 0 ? ~ux : ux;
    return write_varint(ux);
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

  /**
   * @brief Returns max possible size of this message in bytes
   */
  virtual size_t maxSize() = 0;

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

  size_t maxSize() override { return 1; }

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
    len += buffer->write_varint(data);
    return len;
  }

  size_t maxSize() override { return 1 + sizeof(int32_t); }

  ~IntLogger() override = default;
};

class UIntLogger : public BaseTypeLogger {
private:
  uint32_t data;
  static constexpr char intMagic = 0x16;

public:
  UIntLogger() {}
  UIntLogger(uint32_t data) : data(data) {}

  void setData(int data) { this->data = data; }

  char getMagic2() override { return intMagic; }

  size_t LogData(LogBuffer *buffer) override {
    // for now we dont add the first magic since this a basic type
    size_t len = 0;
    len += buffer->write(getMagic2());
    len += buffer->write_varint(data);
    return len;
  }

  size_t maxSize() override { return 1 + sizeof(uint32_t); }

  ~UIntLogger() override = default;
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

  size_t maxSize() override { return 1 + sizeof(float); }

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

  size_t maxSize() override { return 1 + 3 * sizeof(float); }

  ~PoseLogger() override = default;
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
  LogBuffer buf(message->maxSize() + 200);

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
