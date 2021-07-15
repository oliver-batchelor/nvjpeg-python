#pragma once

#include <malloc.h>
#include <sys/types.h>

#include <iostream>

class Buffer {
  private:

    Buffer(Buffer const& buffer) {}
    void operator = (Buffer const& buffer) {}

    uint8_t *data;
    size_t size; 

  public:
    Buffer() : data(nullptr), size(0) { }

    Buffer(uint8_t *data, size_t size) 
      : data(data), size(size)
    { }



    Buffer(Buffer && buffer) 
      : data(buffer.data), size(buffer.size) {

      buffer.data = nullptr;
      buffer.size = 0;
    }

    Buffer(size_t size) : data(nullptr), size(size) {
      data = static_cast<uint8_t*>(malloc(size));
    }

    void operator = (Buffer && buffer) {
      if (nullptr != data) free(data);

      data = buffer.data;
      size = buffer.size;
    }


    ~Buffer() {
      if (nullptr != data) free(data);
    }

    uint8_t *release() {
      auto temp = data; 
      data = nullptr;
      size = 0;
      return temp;
    }

    uint8_t const *get_data() const { return data; }
    uint8_t *get_data() { return data; }


    size_t get_size() const { return size; }

   
};