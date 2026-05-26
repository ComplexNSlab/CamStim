/*
 * cgrabcallback.c
 *
 * Provides fast_memcpy(dst_addr, src_addr, nbytes) where dst_addr and src_addr
 * are integer values of raw C pointers.  This avoids ctypes object creation
 * overhead inside the camera GrabCallback hot path.
 *
 * Also provides fast_memcpy_from_buf(dst_addr, src_buffer, nbytes) as a
 * fallback that accepts a Python buffer (e.g. ctypes array) as the source.
 */
#include <Python.h>
#include <stdint.h>
#include <string.h>

/*
 * fast_memcpy(dst_addr: int, src_addr: int, nbytes: int) -> None
 *
 * Copies nbytes from the raw address src_addr to dst_addr.
 * Both addresses are passed as Python integers (e.g. from ctypes.addressof).
 */
static PyObject* fast_memcpy(PyObject* self, PyObject* args) {
    unsigned long long dst_addr, src_addr;
    Py_ssize_t nbytes;
    if (!PyArg_ParseTuple(args, "KKn", &dst_addr, &src_addr, &nbytes))
        return NULL;
    if (nbytes > 0)
        memcpy((void*)(uintptr_t)dst_addr, (const void*)(uintptr_t)src_addr, (size_t)nbytes);
    Py_RETURN_NONE;
}

/*
 * fast_memcpy_from_buf(dst_addr: int, src_buffer, nbytes: int) -> None
 *
 * Copies nbytes from a Python buffer object (e.g. ctypes array, bytearray)
 * into the raw address dst_addr.
 */
static PyObject* fast_memcpy_from_buf(PyObject* self, PyObject* args) {
    unsigned long long dst_addr;
    Py_buffer src;
    Py_ssize_t nbytes;
    if (!PyArg_ParseTuple(args, "Ky*n", &dst_addr, &src, &nbytes))
        return NULL;
    if (nbytes > src.len)
        nbytes = src.len;
    if (nbytes > 0)
        memcpy((void*)(uintptr_t)dst_addr, src.buf, (size_t)nbytes);
    PyBuffer_Release(&src);
    Py_RETURN_NONE;
}

/*
 * bin_u8_batch_sum_pow2(src_addr: int, n: int, in_h: int, in_w: int, bin_size: int, dst_addr: int) -> None
 *
 * Bins a batch of uint8 frames into uint16 output using sum pooling.
 * Input layout:  N x in_h x in_w (contiguous uint8)
 * Output layout: N x (in_h // bin_size) x (in_w // bin_size) (contiguous uint16)
 *
 * bin_size must be a power-of-two > 1.
 */
static PyObject* bin_u8_batch_sum_pow2(PyObject* self, PyObject* args) {
    unsigned long long src_addr, dst_addr;
    Py_ssize_t n, in_h, in_w, bin_size;
    if (!PyArg_ParseTuple(args, "KnnnnK", &src_addr, &n, &in_h, &in_w, &bin_size, &dst_addr))
        return NULL;

    if (n <= 0 || in_h <= 0 || in_w <= 0) {
        Py_RETURN_NONE;
    }

    if (bin_size <= 1 || (bin_size & (bin_size - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError, "bin_size must be a power-of-two greater than 1");
        return NULL;
    }

    const Py_ssize_t out_h = in_h / bin_size;
    const Py_ssize_t out_w = in_w / bin_size;
    if (out_h <= 0 || out_w <= 0) {
        PyErr_SetString(PyExc_ValueError, "Input frame too small for requested bin_size");
        return NULL;
    }

    const uint8_t* src_base = (const uint8_t*)(uintptr_t)src_addr;
    uint16_t* dst_base = (uint16_t*)(uintptr_t)dst_addr;
    const Py_ssize_t src_frame_stride = in_h * in_w;
    const Py_ssize_t dst_frame_stride = out_h * out_w;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t frame_idx = 0; frame_idx < n; ++frame_idx) {
        const uint8_t* src_frame = src_base + frame_idx * src_frame_stride;
        uint16_t* dst_frame = dst_base + frame_idx * dst_frame_stride;

        for (Py_ssize_t oy = 0; oy < out_h; ++oy) {
            const Py_ssize_t sy0 = oy * bin_size;
            uint16_t* dst_row = dst_frame + oy * out_w;
            for (Py_ssize_t ox = 0; ox < out_w; ++ox) {
                const Py_ssize_t sx0 = ox * bin_size;
                uint32_t accum = 0;
                for (Py_ssize_t ky = 0; ky < bin_size; ++ky) {
                    const uint8_t* src_row = src_frame + (sy0 + ky) * in_w + sx0;
                    for (Py_ssize_t kx = 0; kx < bin_size; ++kx) {
                        accum += (uint32_t)src_row[kx];
                    }
                }
                dst_row[ox] = (uint16_t)accum;
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef CGrabCallbackMethods[] = {
    {"fast_memcpy",          fast_memcpy,          METH_VARARGS,
     "fast_memcpy(dst_addr, src_addr, nbytes) -- raw-address memcpy, no Python objects"},
    {"fast_memcpy_from_buf", fast_memcpy_from_buf, METH_VARARGS,
     "fast_memcpy_from_buf(dst_addr, src_buffer, nbytes) -- copy from Python buffer to raw address"},
    {"bin_u8_batch_sum_pow2", bin_u8_batch_sum_pow2, METH_VARARGS,
     "bin_u8_batch_sum_pow2(src_addr, n, in_h, in_w, bin_size, dst_addr) -- native uint8->uint16 sum binning"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cgrabcallbackmodule = {
    PyModuleDef_HEAD_INIT,
    "cgrabcallback",
    "Fast memcpy helpers for camera GrabCallback hot path",
    -1,
    CGrabCallbackMethods
};

PyMODINIT_FUNC PyInit_cgrabcallback(void) {
    return PyModule_Create(&cgrabcallbackmodule);
}
