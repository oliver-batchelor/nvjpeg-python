#include <JpegCoder.hpp>
#include <nvjpeg.h>

typedef struct
{
  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_statue;
  nvjpegEncoderState_t nv_enc_state;
  cudaStream_t stream;
} NvJpegContext;

#define ChromaSubsampling_Covert_JpegCoderToNvJpeg(subsampling) ((nvjpegChromaSubsampling_t)(subsampling))
#define ChromaSubsampling_Covert_NvJpegToJpegCoder(subsampling) ((JpegCoderChromaSubsampling)(subsampling))

class JpegCoderImageX86 : public JpegCoderImage {
  public:

  JpegCoderImageX86(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling, cudaStream_t stream=NULL);
  virtual ~JpegCoderImageX86();

  size_t get_width() { return height; }
  size_t get_height() { return width; }

  size_t get_nChannels() { return nChannel; }

  
  void fill(const unsigned char* data);
  unsigned char* buffer();

  nvjpegImage_t *img;
  JpegCoderChromaSubsampling subsampling;
  size_t height;
  size_t width;
  short nChannel;


  cudaStream_t stream;
};

class JpegCoderX86 : public JpegCoder {
  public:
  JpegCoderX86();
  ~JpegCoderX86();
  
  void ensureThread(long threadIdent);
  JpegCoderImage* decode(const unsigned char* jpegData, size_t length);
  JpegCoderBytes* encode(JpegCoderImage* img, int quality);

  JpegCoderImageX86 *createImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling);

  private:

  NvJpegContext context;
};




JpegCoder *JpegCoder::create() {
  return new JpegCoderX86();
}


JpegCoderImageX86::JpegCoderImageX86(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling, cudaStream_t stream){
    unsigned char * pBuffer = nullptr;

    cudaError_t eCopy = cudaMallocAsync((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT, stream);

    if (cudaSuccess != eCopy){
        throw JpegCoderError(eCopy, cudaGetErrorString(eCopy));
    }

    nvjpegImage_t *img = new nvjpegImage_t();
    for(int i = 0;i<NVJPEG_MAX_COMPONENT;i++){
        img->channel[i] = pBuffer + (width*height*i);
        img->pitch[i] = (unsigned int)width;
    }

    img->pitch[0] = (unsigned int)width*3;

    this->img = img;
    this->height = height;
    this->width = width;
    this->nChannel = nChannel;
    this->subsampling = subsampling;

    this->stream = stream;
}


void JpegCoderImageX86::fill(const unsigned char* data){
  cudaError_t eCopy = cudaMemcpyAsync(img->channel[0],  
    data, width * height * 3, cudaMemcpyHostToDevice, this->stream);
  
  if (cudaSuccess != eCopy){
      throw JpegCoderError(eCopy, cudaGetErrorString(eCopy));
  }
  
  this->subsampling = JPEGCODER_CSS_444;
}

unsigned char* JpegCoderImageX86::buffer(){
    nvjpegImage_t* img = this->img;
    size_t size = height*width*3;
    unsigned char* buffer = (unsigned char*)malloc(size);
    cudaMemcpy(buffer, img->channel[0], size, cudaMemcpyDeviceToHost);
    return buffer;
}

JpegCoderImageX86::~JpegCoderImageX86(){
    cudaFreeAsync(img->channel[0], this->stream);
    delete this->img;
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

void JpegCoderX86::ensureThread(long threadIdent){
    ;
}

JpegCoderImage* JpegCoderX86::decode(const unsigned char* jpegData, size_t length){
    nvjpegHandle_t nv_handle = context.nv_handle;
    nvjpegJpegState_t nv_statue = context.nv_statue;

    cudaStream_t stream = context.stream;

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegGetImageInfo(nv_handle, jpegData, length, &nComponent, &subsampling, widths, heights);

    JpegCoderImageX86* imgdesc = this->createImage(widths[0], heights[0], nComponent, ChromaSubsampling_Covert_NvJpegToJpegCoder(subsampling));
    int nReturnCode = nvjpegDecode(nv_handle, nv_statue, jpegData, length, NVJPEG_OUTPUT_BGRI, (nvjpegImage_t *)(imgdesc->img), stream);

    if (NVJPEG_STATUS_SUCCESS != nReturnCode){
        throw JpegCoderError(nReturnCode, "NvJpeg Decoder Error");
    }

    return imgdesc;
}


JpegCoderImageX86 *JpegCoderX86::createImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling) {
    return new JpegCoderImageX86(width, height, nChannel, subsampling, this->context.stream);
}



JpegCoderBytes* JpegCoderX86::encode(JpegCoderImage* _img, int quality){
    JpegCoderImageX86 *img = (JpegCoderImageX86*)_img;

    nvjpegHandle_t nv_handle = context.nv_handle;
    nvjpegEncoderState_t nv_enc_state = context.nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream = context.stream;
    
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);

    nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
    nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, stream);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, 
      ChromaSubsampling_Covert_JpegCoderToNvJpeg(img->subsampling), stream);

    int nReturnCode = nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, 
      (nvjpegImage_t*)(img->img), NVJPEG_INPUT_BGRI, (int)img->width, (int)img->height, stream);

    if (NVJPEG_STATUS_SUCCESS != nReturnCode){
        throw JpegCoderError(nReturnCode, "NvJpeg Encoder Error");
    }
    
    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
    
    JpegCoderBytes* jpegData = new JpegCoderBytes(length);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpegData->data, &(jpegData->size), stream);

    cudaStreamSynchronize(context.stream);

    nvjpegEncoderParamsDestroy(nv_enc_params);
    return jpegData;
}

