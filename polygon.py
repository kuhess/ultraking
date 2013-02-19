import cv
import numpy


class Polygon:
    """Polygon is a class for geometric extraction. In this implementation, the polygon is a parallelogram set by its center, its size, its rotation and its transvection."""
    def __init__(self, center, size, rotation=0, transvection=0, outSize=(32, 32)):
        """Create a Polygon."""
        self.center = center
        self.size = size
        self.rotation = rotation
        self.transvection = transvection
        self.outSize = outSize

    def corners(self):
        xy1 = numpy.ones((3, 4))
        xy1[0, 0] = -self.outSize[0] / 2
        xy1[0, 1] = -self.outSize[0] / 2
        xy1[0, 2] = self.outSize[0] / 2
        xy1[0, 3] = self.outSize[0] / 2
        xy1[1, 0] = -self.outSize[1] / 2
        xy1[1, 1] = self.outSize[1] / 2
        xy1[1, 2] = self.outSize[1] / 2
        xy1[1, 3] = -self.outSize[1] / 2

        mapMatrix = numpy.ones((2, 3))
        mapMatrix[0, 0] = self.mapMatrix()[0, 0]
        mapMatrix[0, 1] = self.mapMatrix()[0, 1]
        mapMatrix[0, 2] = self.mapMatrix()[0, 2]
        mapMatrix[1, 0] = self.mapMatrix()[1, 0]
        mapMatrix[1, 1] = self.mapMatrix()[1, 1]
        mapMatrix[1, 2] = self.mapMatrix()[1, 2]

        xy2 = numpy.dot(mapMatrix, xy1)
        polygonCorners = (
            (int(xy2[0, 0]), int(xy2[1, 0])),
            (int(xy2[0, 1]), int(xy2[1, 1])),
            (int(xy2[0, 2]), int(xy2[1, 2])),
            (int(xy2[0, 3]), int(xy2[1, 3]))
        )
        return polygonCorners

    def mapMatrix(self):
        """Get the mapping matrix to use in order to extract a the polygon area in an image."""
        mapMatrix = cv.CreateMat(2, 3, cv.CV_32FC1)
        sx = self.size[0] / float(self.outSize[0])
        sy = self.size[1] / float(self.outSize[1])
        mapMatrix[0, 0] = sx * numpy.cos(self.rotation)
        mapMatrix[0, 1] = sy * (numpy.cos(self.rotation) * numpy.sin(self.transvection) - numpy.sin(self.rotation))
        mapMatrix[1, 0] = sx * numpy.sin(self.rotation)
        mapMatrix[1, 1] = sy * (numpy.sin(self.rotation) * numpy.sin(self.transvection) + numpy.cos(self.rotation))
        mapMatrix[0, 2] = self.center[0]
        mapMatrix[1, 2] = self.center[1]
        return mapMatrix
