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

class JpegCoderX86 : public JpegCoder {
  public:
  JpegCoderX86();
  ~JpegCoderX86();
  
  void ensureThread(long threadIdent);
  JpegCoderImage* decode(const unsigned char* jpegData, size_t length);
  JpegCoderBytes* encode(JpegCoderImage* img, int quality);

  private:
  NvJpegContext context;
};


JpegCoder *JpegCoder::create() {
  return new JpegCoderX86();
}

JpegCoderImage::JpegCoderImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling){
    unsigned char * pBuffer = nullptr; 
    cudaError_t eCopy = cudaMalloc((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT);
    if (cudaSuccess != eCopy){
        throw JpegCoderError(eCopy, cudaGetErrorString(eCopy));
    }

    nvjpegImage_t *img = (nvjpegImage_t*)malloc(sizeof(nvjpegImage_t));
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
}

void JpegCoderImage::fill(const unsigned char* data){
    cudaError_t eCopy = cudaMemcpy(((nvjpegImage_t*)(this->img))->channel[0], data, width*height*3, cudaMemcpyHostToDevice);
    if (cudaSuccess != eCopy){
        throw JpegCoderError(eCopy, cudaGetErrorString(eCopy));
    }
    this->subsampling = JPEGCODER_CSS_444;
}

unsigned char* JpegCoderImage::buffer(){
    nvjpegImage_t* img = ((nvjpegImage_t*)(this->img));
    size_t size = height*width*3;
    unsigned char* buffer = (unsigned char*)malloc(size);
    cudaMemcpy(buffer, img->channel[0], size, cudaMemcpyDeviceToHost);
    return buffer;
}

JpegCoderImage::~JpegCoderImage(){
    cudaFree(((nvjpegImage_t *)(this->img))->channel[0]);
    free(this->img);
    this->img = nullptr;
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

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegGetImageInfo(nv_handle, jpegData, length, &nComponent, &subsampling, widths, heights);

    JpegCoderImage* imgdesc = new JpegCoderImage(widths[0], heights[0], nComponent, ChromaSubsampling_Covert_NvJpegToJpegCoder(subsampling));
    int nReturnCode = nvjpegDecode(nv_handle, nv_statue, jpegData, length, NVJPEG_OUTPUT_BGRI, (nvjpegImage_t *)(imgdesc->img), NULL);

    if (NVJPEG_STATUS_SUCCESS != nReturnCode){
        throw JpegCoderError(nReturnCode, "NvJpeg Decoder Error");
    }

    return imgdesc;
}

JpegCoderBytes* JpegCoderX86::encode(JpegCoderImage* img, int quality){
    nvjpegHandle_t nv_handle = context.nv_handle;
    nvjpegEncoderState_t nv_enc_state = context.nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream = context.stream;

    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, NULL);

    nvjpegEncoderParamsSetQuality(nv_enc_params, quality, NULL);
    nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, NULL);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, ChromaSubsampling_Covert_JpegCoderToNvJpeg(img->subsampling), NULL);

    int nReturnCode = nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, (nvjpegImage_t*)(img->img), NVJPEG_INPUT_BGRI, (int)img->width, (int)img->height, stream);
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

