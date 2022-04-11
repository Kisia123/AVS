import cv2
import numpy as np


def toGray(IMG):
    return cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)


def erosion(IMG):
    return cv2.erode(IMG, np.ones((3, 3), np.uint8))


def dilation(IMG):
    return cv2.dilate(IMG, np.ones((3, 3), np.uint8))


def binarization(IMG):
    return cv2.threshold(IMG, 20, 255, cv2.THRESH_BINARY)[1]


def median(IMG):
    return cv2.medianBlur(IMG, 7)


if __name__ == '__main__':
    last = cv2.imread("pedestrian/input/in%06d.jpg" % 300)
    last = toGray(last)
    TP = TN = FP = FN = 0
    for i in range(301, 1100, 1):
        I = cv2.imread("pedestrian/input/in%06d.jpg" % i)
        current = toGray(I)
        DIFF = cv2.absdiff(last, current)
        MED = median(DIFF)
        BIN = binarization(MED)
        ER = erosion(BIN)
        DIL = dilation(ER)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(DIL)
        I_VIS = I
        if (stats.shape[0] > 1):
            tab = stats[1:, 4]
            pi = np.argmax(tab)
            pi = pi + 1
            cv2.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]),
                          (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
            cv2.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0))
            cv2.putText(I_VIS, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        B = DIL
        GTB = cv2.imread("pedestrian/groundtruth/gt%06d.png" % i)
        GTB = toGray(GTB)
        TP_M = np.logical_and((B == 255), (GTB == 255))
        TN_M = np.logical_and((B == 0), (GTB == 0))
        FP_M = np.logical_and((B == 255), (GTB == 0))
        FN_M = np.logical_and((B == 0), (GTB == 255))

        TP_S = np.sum(TP_M)
        TN_S = np.sum(TN_M)
        FP_S = np.sum(FP_M)
        FN_S = np.sum(FN_M)

        TP = TP + TP_S
        TN = TN + TN_S
        FP = FP + FP_S
        FN = FN + FN_S
        cv2.imshow("DIFF", I_VIS)
        cv2.waitKey(10)
        last = current
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print("P = %f, R = %f, F1 = %f" % (P, R, F1))
