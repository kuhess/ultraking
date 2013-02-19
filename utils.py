import cv
import numpy


def array2img(ar, outSize):
    """Reshape a mono-dimensional array (a vector) into an IplImage."""
    ar = ar.reshape(outSize)
    ar = (ar - ar.min()) / (ar.max() - ar.min())
    return array2cv(ar)


def cv2array(im):
    """Transform an cv.IplImage to a numpy.array."""
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64'
    }
    a = numpy.fromstring(im.tostring(), dtype=depth2dtype[im.depth], count=im.width * im.height * im.nChannels)
    a.shape = (im.height, im.width, im.nChannels)
    return a


def array2cv(a):
    """Transform a numpy.array to an cv.IplImage ."""
    dtype2depth = {
        'uint8': cv.IPL_DEPTH_8U,
        'int8': cv.IPL_DEPTH_8S,
        'uint16': cv.IPL_DEPTH_16U,
        'int16': cv.IPL_DEPTH_16S,
        'int32': cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1], a.shape[0]), dtype2depth[str(a.dtype)], nChannels)
    cv.SetData(cv_im, a.tostring(), a.dtype.itemsize * nChannels * a.shape[1])
    return cv_im
