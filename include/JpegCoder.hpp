#pragma once

#include <sys/types.h>
#include <memory>
#include <iostream>
#include <exception>
#include <vector>

#include <Buffer.hpp>

class JpegCoderError: public std::runtime_error{
protected:
    int _code;
public:
    JpegCoderError(int _code, const std::string& str) 
      : std::runtime_error(str), _code(_code) {
    }
    JpegCoderError(int _code, const char* str)
      : std::runtime_error(str), _code(_code) {
    }

    int code() {
        return _code;
    }
};

typedef enum
{
    JPEGCODER_CSS_444 = 0,
    JPEGCODER_CSS_422 = 1,
    JPEGCODER_CSS_420 = 2,
    JPEGCODER_CSS_440 = 3,
    JPEGCODER_CSS_411 = 4,
    JPEGCODER_CSS_410 = 5,
    JPEGCODER_CSS_GRAY = 6,
    JPEGCODER_CSS_UNKNOWN = -1
} JpegCoderChromaSubsampling;

typedef enum{
    JPEGCODER_PIXFMT_RGB         = 3,
    JPEGCODER_PIXFMT_BGR         = 4, 
    JPEGCODER_PIXFMT_RGBI        = 5, 
    JPEGCODER_PIXFMT_BGRI        = 6,
}JpegCoderColorFormat;



class JpegCoderImage {
public:

    virtual ~JpegCoderImage() {}
    
    virtual void fill(const uint8_t *data, size_t size) = 0;
    virtual Buffer buffer() const = 0;

    virtual size_t get_width() const = 0;
    virtual size_t get_height() const = 0;

    virtual size_t get_nChannels() const = 0;

};


class JpegCoder{
public:
    JpegCoder() {}
    virtual  ~JpegCoder() {}

    static std::shared_ptr<JpegCoder> create();

    virtual std::shared_ptr<JpegCoderImage> decode(uint8_t const *data, size_t size) = 0;
    virtual Buffer encode(std::shared_ptr<JpegCoderImage> img, int quality) = 0;

    virtual std::shared_ptr<JpegCoderImage> createImage(size_t width, size_t height, 
      short nChannel, JpegCoderChromaSubsampling subsampling) = 0;

};
