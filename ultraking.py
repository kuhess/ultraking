import cv
import numpy

from utils import array2cv, cv2array, array2img
from tracker import skl, warpImg, maxLikelihood
from polygon import Polygon

def onMouse(event, x, y, flags, polygon):
    if event == cv.CV_EVENT_LBUTTONDOWN:
        polygon.center = (x, y)
    elif event == cv.CV_EVENT_MOUSEMOVE and (flags & cv.CV_EVENT_FLAG_LBUTTON):
        polygon.size = (abs(polygon.center[0] - x) * 2, abs(polygon.center[1] - y) * 2)

if __name__ == "__main__":
    # Polygon parameters
    center = (0, 0)
    size = (0, 0)
    outSize = (32, 32)
    polygon = Polygon(center, size, outSize=outSize)
    
    cv.NamedWindow("Webcam Stream", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("Warped Stream", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("Mean Stream", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("1st Eigenface Stream", cv.CV_WINDOW_AUTOSIZE)
    cv.SetMouseCallback("Webcam Stream", onMouse, polygon)
    
    capture = cv.CreateCameraCapture(0)

    extract=False

    #SKL initialization
    actualisation = 10
    data = None
    U = None
    D = None
    mu = None
    n = None
    ff = 0.99
    K = 16
    i = 0
    
    # Dumb particles filter parameters
    num = 300
    sigmaCenterX = 20
    sigmaCenterY = 20
    sigmaSizeX = 5
    sigmaSizeY = 3
    sigmaRot = 10 * numpy.pi / 180.0
    sigmaIncl = 1 * numpy.pi / 180.0
    
    while True:
        # Get the image from the camera
        inImgRGB = cv.QueryFrame(capture)
        inImgGray = cv.CreateImage(cv.GetSize(inImgRGB), inImgRGB.depth, 1)
        cv.CvtColor(inImgRGB, inImgGray, cv.CV_RGB2GRAY)
        
        if extract:
            # Multiple extraction
            polygonTmp = []
            polygonTmp.append(polygon)
            subImgs = None
            for k in range(num):
                centerTmp = polygon.center[0] + numpy.random.randn() * sigmaCenterX, polygon.center[1] + numpy.random.randn() * sigmaCenterY
                sizeTmp = polygon.size[0] + numpy.random.randn() * sigmaSizeX, polygon.size[1] + numpy.random.randn() * sigmaSizeY
                rotationTmp = polygon.rotation + numpy.random.randn() * sigmaRot
                inclinaisonTmp = polygon.transvection + numpy.random.randn() * sigmaIncl
                polygonTmp.append(Polygon(centerTmp, sizeTmp, rotationTmp, inclinaisonTmp, outSize))
                
                subImgTmp = warpImg(inImgGray, polygonTmp[k])
                if subImgs == None:
                    subImgs = numpy.mat(cv2array(subImgTmp)).ravel().T
                else:
                    subImgs = numpy.hstack((subImgs, numpy.mat(cv2array(subImgTmp)).ravel().T))
            idx = maxLikelihood(subImgs, U, D, mu)
            polygon = polygonTmp[idx]
            
            # SKL!!!
            newData = subImgs[:,idx]
            if data == None:
                data = newData
            else:
                data = numpy.hstack((data, newData))
            if i % actualisation == 0:
                U, D, mu, n = skl(data=data, U0=U, D0=D, mu0=mu, n0=n, ff=ff, K=K)
                data = None
            
            cv.ShowImage("Warped Stream", array2img(subImgs[:, idx],outSize))
            cv.ShowImage("Mean Stream", array2img(mu, outSize))
            cv.ShowImage("1st Eigenface Stream", array2img(U[:, 0],outSize))
            i = i + 1
        
        # Draw the extraction polygon
        cv.PolyLine(inImgGray, (polygon.corners(),), True, cv.RGB(255, 255, 255))
        cv.ShowImage("Webcam Stream", inImgGray)
        
        keyPressed = cv.WaitKey(10)
        if keyPressed == 13:
            extract = not(extract)
        elif keyPressed != -1:
            break
