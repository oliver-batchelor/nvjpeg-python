#pragma once

#include <sys/types.h>
#include <malloc.h>
#include <memory.h>
#include <iostream>
#include <exception>
#include <vector>

class JpegCoderError: public std::runtime_error{
protected:
    int _code;
public:
    JpegCoderError(int code, const std::string& str):std::runtime_error(str){
        this->_code = code;
    }
    JpegCoderError(int code, const char* str):std::runtime_error(str){
        this->_code = code;
    }
    int code(){
        return this->_code;
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
    
    virtual void fill(const void* data) = 0;
    virtual void* buffer() = 0;

    virtual size_t get_width() = 0;
    virtual size_t get_height() = 0;

    virtual size_t get_nChannels() = 0;

};




class JpegCoder{
public:
    JpegCoder() {}
    virtual  ~JpegCoder() {}

    static JpegCoder *create();

    virtual void ensureThread(long threadIdent) = 0;
    virtual JpegCoderImage* decode(const void* jpegData, size_t length) = 0;
    virtual std::vector<unsigned char> encode(JpegCoderImage* img, int quality) = 0;

    virtual JpegCoderImage *createImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling) = 0;

};
