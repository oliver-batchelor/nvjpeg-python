#include <stdio.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <Python.h>
#include <pythread.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "JpegCoder.hpp"

typedef struct
{
    PyObject_HEAD
    std::shared_ptr<JpegCoder> m_handle;
}NvJpeg;


static PyMemberDef NvJpeg_DataMembers[] =
{
        {"m_handle",   T_OBJECT, offsetof(NvJpeg, m_handle),   0, "NvJpeg handle ptr"},
        {NULL, 0, 0, 0, NULL}
};

int NvJpeg_init(PyObject *self, PyObject *args, PyObject *kwds) {
  ((NvJpeg*)self)->m_handle = JpegCoder::create();
  return 0;
}


static void NvJpeg_Destruct(PyObject* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* NvJpeg_Str(PyObject* self)
{
    return Py_BuildValue("s", "<nvjpeg-python.nvjpeg>");
}

static PyObject* NvJpeg_Repr(PyObject* self)
{
    return NvJpeg_Str(self);
}

static PyObject* NvJpeg_decode(NvJpeg* self, PyObject* Argvs)
{
    auto m_handle = self->m_handle;
    
    Py_buffer pyBuf;
    if(!PyArg_ParseTuple(Argvs, "y*", &pyBuf)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should jpegData byte string!");
        return NULL;
    }


    try{
        auto img = m_handle->decode((uint8_t*)pyBuf.buf, pyBuf.len);
        PyBuffer_Release(&pyBuf);

        auto image_data = img->buffer().release();

        npy_intp dims[3] = {(npy_intp)(img->get_height() ), (npy_intp)(img->get_width() ), 3};
        PyObject* image = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image_data);

        PyArray_ENABLEFLAGS((PyArrayObject*) image, NPY_ARRAY_OWNDATA);
        return image;

   }catch(JpegCoderError& e){
        PyBuffer_Release(&pyBuf);
        PyErr_Format(PyExc_ValueError, "%s, Code: %d", e.what(), e.code());
        return NULL;
    }


}

static PyObject* NvJpeg_encode(NvJpeg* self, PyObject* argv)
{
    PyArrayObject *image_in;
    unsigned int quality = 90;
    if (!PyArg_ParseTuple(argv, "O!|I", &PyArray_Type, &image_in, &quality)){
        PyErr_SetString(PyExc_ValueError, "NvJpeg_encode: expected ndarray, int");
        return NULL;
    }

    if (NULL == image_in){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(image_in) != 3){
        PyErr_SetString(PyExc_ValueError, "NvJpeg_encode: expected 3d ndarray of HxWx3");
        return NULL;
    }

    if(quality>100){
        quality = 100;
    }
    
    auto m_handle = self->m_handle;

    const void *buffer = PyArray_DATA(image_in);

    auto img = m_handle->createImage(PyArray_DIM(image_in, 1), PyArray_DIM(image_in, 0), 3, JPEGCODER_CSS_444);
    img->fill((uint8_t *)buffer, PyArray_SIZE(image_in));
    
    auto jpegData = m_handle->encode(img, quality);

    PyObject* rtn = PyBytes_FromStringAndSize((const char*)jpegData.get_data(), jpegData.get_size()); 
    return rtn;
}


static PyMethodDef NvJpeg_MethodMembers[] =
{
        {"encode",  (PyCFunction)NvJpeg_encode,  METH_VARARGS,  "encode jpge"},
        {"decode", (PyCFunction)NvJpeg_decode, METH_VARARGS,  "decode jpeg"},
        {NULL, NULL, 0, NULL}
};


static PyTypeObject NvJpeg_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)
        "nvjpeg.NvJpeg",
        sizeof(NvJpeg),
        0
};

void NvJpeg_module_destroy(void *_){
}

static PyModuleDef ModuleInfo =
{
        PyModuleDef_HEAD_INIT,
        "NvJpeg Module",
        "NvJpeg by Nvjpeg",
        -1,
        NULL, NULL, NULL, NULL,
        NvJpeg_module_destroy
};

PyMODINIT_FUNC
PyInit_nvjpeg(void) {
    PyObject * pReturn = NULL;

    NvJpeg_ClassInfo.tp_dealloc   = NvJpeg_Destruct;
    NvJpeg_ClassInfo.tp_repr      = NvJpeg_Repr;
    NvJpeg_ClassInfo.tp_str       = NvJpeg_Str;
    NvJpeg_ClassInfo.tp_flags     = Py_TPFLAGS_DEFAULT;
    NvJpeg_ClassInfo.tp_doc       = "NvJpeg Python Objects---Extensioned by nvjpeg";
    NvJpeg_ClassInfo.tp_weaklistoffset = 0;
    NvJpeg_ClassInfo.tp_methods   = NvJpeg_MethodMembers;
    NvJpeg_ClassInfo.tp_members   = NvJpeg_DataMembers;
    NvJpeg_ClassInfo.tp_dictoffset = 0;
    NvJpeg_ClassInfo.tp_init      = NvJpeg_init;
    NvJpeg_ClassInfo.tp_new = PyType_GenericNew;

    if(PyType_Ready(&NvJpeg_ClassInfo) < 0) 
        return NULL;

    pReturn = PyModule_Create(&ModuleInfo);
    if(pReturn == NULL)
        return NULL;

    Py_INCREF(&ModuleInfo);

    Py_INCREF(&NvJpeg_ClassInfo);
    if (PyModule_AddObject(pReturn, "NvJpeg", (PyObject*)&NvJpeg_ClassInfo) < 0) {
        Py_DECREF(&NvJpeg_ClassInfo);
        Py_DECREF(pReturn);
        return NULL;
    }

    import_array();
    return pReturn;
}
