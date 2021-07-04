#include <JpegCoder.hpp>
#include <nvjpeg.h>
#include <memory>
#include <assert.h>

typedef struct
{
  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_statue;
  nvjpegEncoderState_t nv_enc_state;
  cudaStream_t stream;
} NvJpegContext;


class JpegCoderImageX86 : public JpegCoderImage {
  public:

    JpegCoderImageX86(size_t width, size_t height, short nChannel, 
      JpegCoderChromaSubsampling subsampling, cudaStream_t stream=NULL);

    virtual ~JpegCoderImageX86();

    void fill(uint8_t const *data, size_t size);
    Buffer buffer() const;

    size_t get_width() const { return width; }
    size_t get_height() const { return height; }
    size_t get_nChannels() const { return nChannel; }
    
  protected:
    nvjpegImage_t img;
    
    void *deviceBuffer;

    JpegCoderChromaSubsampling subsampling;
    size_t height;
    size_t width;
    short nChannel;

    cudaStream_t stream;

    friend class JpegCoderX86;
};

class JpegCoderX86 : public JpegCoder {
  public:
    JpegCoderX86();
    ~JpegCoderX86();
    
    std::shared_ptr<JpegCoderImage> decode(uint8_t const *jpegData, size_t length);
    Buffer encode(std::shared_ptr<JpegCoderImage> img, int quality);

    std::shared_ptr<JpegCoderImage> createImage(size_t width, size_t height, 
      short nChannel, JpegCoderChromaSubsampling subsampling = JPEGCODER_CSS_422);

  private:

    NvJpegContext context;
};


std::shared_ptr<JpegCoder> JpegCoder::create() {
  return std::shared_ptr<JpegCoder>(new JpegCoderX86());
}


inline void check_cuda(cudaError_t code) {
  if (cudaSuccess != code) {
    throw JpegCoderError(code, cudaGetErrorString(code));
  }
}

inline void check_nvjpeg(std::string const &message, nvjpegStatus_t code) {
  if (NVJPEG_STATUS_SUCCESS != code){
      throw JpegCoderError(code, message);
  }
}

JpegCoderImageX86::JpegCoderImageX86(size_t width, size_t height, short nChannel, 
  JpegCoderChromaSubsampling subsampling, cudaStream_t stream) 
  :  deviceBuffer(nullptr), subsampling(subsampling), 
     height(height), width(width),  nChannel(nChannel), stream(stream)
  {

    check_cuda(
      cudaMallocAsync(&deviceBuffer, width * height * nChannel, stream));
    size_t pitch = width * nChannel;

    // check_cuda(
    //   cudaMallocPitch(&deviceBuffer, &pitch, width * nChannel, height));

    for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
        img.channel[i] = nullptr;
        img.pitch[i] = 0;
    }

    img.pitch[0] = (unsigned int)pitch;
    img.channel[0] = (unsigned char*)deviceBuffer;
}


void JpegCoderImageX86::fill(uint8_t const *data, size_t data_size){
  size_t size = width * height * nChannel;
  assert(size == data_size);

  // check_cuda(
  //   cudaMemcpy2D(deviceBuffer, img.pitch[0], 
  //     data, width * nChannel, width * nChannel, height, cudaMemcpyHostToDevice));
  
  check_cuda(
    cudaMemcpyAsync(deviceBuffer, data, height * width * nChannel, cudaMemcpyHostToDevice));
}

Buffer JpegCoderImageX86::buffer() const {
    Buffer buffer(height * width * nChannel);

    check_cuda(
      cudaMemcpy(buffer.get_data(), deviceBuffer, buffer.get_size(), cudaMemcpyDeviceToHost));

    return buffer;
}

JpegCoderImageX86::~JpegCoderImageX86(){
    cudaFreeAsync(deviceBuffer, stream);
}

JpegCoderX86::JpegCoderX86() : JpegCoder(){
    nvjpegCreateSimple(&(context.nv_handle));
    nvjpegJpegStateCreate(context.nv_handle, &(context.nv_statue));
    nvjpegEncoderStateCreate(context.nv_handle, &(context.nv_enc_state), NULL);
    cudaStreamCreate(&(context.stream));
}

JpegCoderX86::~JpegCoderX86(){

    nvjpegJpegStateDestroy(context.nv_statue);
    nvjpegEncoderStateDestroy(context.nv_enc_state);
    nvjpegDestroy(context.nv_handle);

    cudaStreamDestroy(context.stream);
}



std::shared_ptr<JpegCoderImage> JpegCoderX86::decode(uint8_t const *jpegData, size_t length){
    nvjpegHandle_t nv_handle = context.nv_handle;
    nvjpegJpegState_t nv_statue = context.nv_statue;

    cudaStream_t stream = context.stream;

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];


    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegGetImageInfo(nv_handle, jpegData, length, 
      &nComponent, &subsampling, widths, heights);

    auto image = std::dynamic_pointer_cast<JpegCoderImageX86>(createImage(widths[0], heights[0], 
      nComponent, static_cast<JpegCoderChromaSubsampling>(subsampling)));

    check_nvjpeg("nvjpegDecode",
      nvjpegDecode(nv_handle, nv_statue, jpegData, 
        length, NVJPEG_OUTPUT_BGRI, &image->img, stream));


    return image;
}


std::shared_ptr<JpegCoderImage> JpegCoderX86::createImage(size_t width, size_t height, 
  short nChannel, JpegCoderChromaSubsampling subsampling) {

    return std::shared_ptr<JpegCoderImage>(
      new JpegCoderImageX86(width, height, nChannel, subsampling, context.stream));
}



Buffer JpegCoderX86::encode(std::shared_ptr<JpegCoderImage> _img, int quality){
    auto img = std::dynamic_pointer_cast<JpegCoderImageX86>(_img);

    nvjpegHandle_t nv_handle = context.nv_handle;
    nvjpegEncoderState_t nv_enc_state = context.nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream = context.stream;
    
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);

    nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
    nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, stream);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, 
      (nvjpegChromaSubsampling_t)img->subsampling, stream);

    check_nvjpeg("nvjpegEncodeImage", 
      nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, 
        &img->img, NVJPEG_INPUT_BGRI, img->width, img->height, stream));

    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
    
    Buffer jpegData(length);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, 
      jpegData.get_data(), &length, stream);

    cudaStreamSynchronize(context.stream);

    nvjpegEncoderParamsDestroy(nv_enc_params);
    return jpegData;
}

