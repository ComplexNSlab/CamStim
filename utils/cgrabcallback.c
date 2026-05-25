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

static PyMethodDef CGrabCallbackMethods[] = {
    {"fast_memcpy",          fast_memcpy,          METH_VARARGS,
     "fast_memcpy(dst_addr, src_addr, nbytes) -- raw-address memcpy, no Python objects"},
    {"fast_memcpy_from_buf", fast_memcpy_from_buf, METH_VARARGS,
     "fast_memcpy_from_buf(dst_addr, src_buffer, nbytes) -- copy from Python buffer to raw address"},
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
