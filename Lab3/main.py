import numpy as np
import cv2
import time


def frame(dir):
    f = open("%s/temporalROI.txt" % dir, "r")
    line = f.readline()
    start, end = line.split(" ")
    start = int(start)
    end = int(end)
    return start, end


def toGray(IMG):
    return cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)


def binarization(IMG):
    return cv2.threshold(IMG, 20, 255, cv2.THRESH_BINARY)[1]


def erosion(IMG):
    return cv2.erode(IMG, np.ones((3, 3), np.uint8))


def dilation(IMG):
    return cv2.dilate(IMG, np.ones((3, 3), np.uint8))


def median(IMG):
    return cv2.medianBlur(IMG, 7)


def subtraction(IMG, BG):
    # background subtraction and binarization
    return cv2.subtract(IMG, BG)


def doChanges(IMG):
    # median filtration
    BLURR = median(IMG)
    BLURR = np.uint8(BLURR)

    # morphological operations
    ERODE = erosion(BLURR)
    DIL = dilation(ERODE)

    # binarization for calc
    BIN = binarization(DIL)

    return BIN


def evaluation(I1, I2, TP, FP, TN, FN):
    TP_M = np.logical_and((I1 == 255), (I2 == 255))
    TN_M = np.logical_and((I1 == 0), (I2 == 0))
    FP_M = np.logical_and((I1 == 255), (I2 == 0))
    FN_M = np.logical_and((I1 == 0), (I2 == 255))

    TP_S = np.sum(TP_M)
    TN_S = np.sum(TN_M)
    FP_S = np.sum(FP_M)
    FN_S = np.sum(FN_M)

    TP = TP + TP_S
    TN = TN + TN_S
    FP = FP + FP_S
    FN = FN + FN_S

    return TP, FP, TN, FN


def calc(TP, FP, TN, FN):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return P, R, F1


def mean_approximation(frame, BG):
    alpha = 0.02
    BGN = alpha * np.float64(frame) + (1 - alpha) * np.float64(BG)
    return np.uint8(BGN)


def median_approximation(frame, BG):
    BGN = BG

    if BG.all() < frame.all():
        BGN = BG + 1
    elif BG.all() > frame.all():
        BGN = BG - 1

    return BGN


def ex1():
    TP_md = TN_md = FP_md = FN_md = 0
    TP_mn = TN_mn = FP_mn = FN_mn = 0
    # frame buffer
    N = 60
    IN = 0
    first = cv2.imread("%s/input/in%06d.jpg" % (DIR, first_fr))
    first = toGray(first)
    w, h = first.shape[:2]
    BUF = np.zeros((w, h, N), np.uint8)

    for i in range(first_fr, first_fr + 100):
        IG = cv2.imread("%s/input/in%06d.jpg" % (DIR, i))
        IG = toGray(IG)
        BG = cv2.imread('%s/groundtruth/gt%06d.png' % (DIR, i))
        BG = toGray(BG)

        # IN - current frame
        BUF[:, :, IN] = IG  # buffer for current frame

        # mean and median
        t1_mn = time.time()
        L = np.mean(BUF, 2)  # 2 - multiple axis
        MEAN = np.uint8(L)
        t2_mn = time.time()
        print("mean time in s:", (t2_mn - t1_mn))

        t1_md = time.time()
        M = np.median(BUF, 2)
        MEDIAN = np.uint8(M)
        t2_md = time.time()
        print("median time in s:", (t2_md - t1_md))
        MEDIAN = subtraction(MEDIAN, BG)
        MEDIAN = doChanges(MEDIAN)
        MEAN = subtraction(MEAN, BG)
        MEAN = doChanges(MEAN)

        # cv2.imshow("purpur", MEDIAN)
        # cv2.waitKey(10)

        TP_md, FP_md, TN_md, FN_md = evaluation(MEDIAN, BG, TP_md, FP_md, TN_md, FN_md)
        TP_mn, FP_mn, TN_mn, FN_mn = evaluation(MEAN, BG, TP_mn, FP_mn, TN_mn, FN_mn)
        break

        # check if IN reached buffer size (N)
        IN += 1
        if IN == N:
            IN = 0

    # print(TP_mn, FP_mn, TN_mn, FN_mn)
    calc_median = calc(TP_md, FP_md, TN_md, FN_md)
    calc_mean = calc(TP_mn, FP_mn, TN_mn, FN_mn)
    print("MEDIAN:")
    print("P = %f, R = %f, F1 = %f" % calc_median)
    print("MEAN:")
    print("P = %f, R = %f, F1 = %f" % calc_mean)


def ex2():
    TP_md = TN_md = FP_md = FN_md = 0
    TP_mn = TN_mn = FP_mn = FN_mn = 0

    first = cv2.imread("%s/input/in%06d.jpg" % (DIR, first_fr))
    prev_md = toGray(first)
    prev_mn = toGray(first)

    for i in range(first_fr + 1, last_fr):
        IG = cv2.imread("%s/input/in%06d.jpg" % (DIR, i))
        IG = toGray(IG)
        BG = cv2.imread('%s/groundtruth/gt%06d.png' % (DIR, i))
        BG = toGray(BG)

        # mean and median
        t1_mn = time.time()
        BGmn = mean_approximation(IG, prev_mn)
        t2_mn = time.time()
        print("mean time in s:", (t2_mn - t1_mn))
        t1_md = time.time()
        BGmd = median_approximation(IG, prev_md)
        t2_md = time.time()
        print("median time in s:", (t2_md - t1_md))

        MEAN = subtraction(BGmn, IG)
        MEAN = doChanges(BGmn)
        MEDIAN = subtraction(BGmd, IG)
        MEDIAN = doChanges(BGmd)

        prev_md = BGmd
        prev_mn = BGmn

        cv2.imshow("purpur", MEDIAN)
        cv2.waitKey(10)

        TP_md, FP_md, TN_md, FN_md = evaluation(MEDIAN, BG, TP_md, FP_md, TN_md, FN_md)
        TP_mn, FP_mn, TN_mn, FN_mn = evaluation(MEAN, BG, TP_mn, FP_mn, TN_mn, FN_mn)
        break

    # print(TP_mn, FP_mn, TN_mn, FN_mn)
    calc_median = calc(TP_md, FP_md, TN_md, FN_md)
    calc_mean = calc(TP_mn, FP_mn, TN_mn, FN_mn)
    print("MEDIAN:")
    print("P = %f, R = %f, F1 = %f" % calc_median)
    print("MEAN:")
    print("P = %f, R = %f, F1 = %f" % calc_mean)


def ex3():
    MOG2 = cv2.createBackgroundSubtractorMOG2(10, 4.8, 0)
    TP = TN = FP = FN = 0

    first = cv2.imread("%s/input/in%06d.jpg" % (DIR, first_fr))
    mask = MOG2.apply(first)

    for i in range(first_fr + 1, last_fr):
        IG = cv2.imread("%s/input/in%06d.jpg" % (DIR, i))
        IG = toGray(IG)
        BG = cv2.imread('%s/groundtruth/gt%06d.png' % (DIR, i))
        BG = toGray(BG)

        # foreground mask (image, fgmask, double learning rate)
        mask = MOG2.apply(IG, mask, 0.3)
        IG_mask = doChanges(mask)

        cv2.imshow("purpur", IG_mask)
        cv2.waitKey(10)

        TP, FP, TN, FN = evaluation(mask, BG, TP, FP, TN, FN)

    # print(TP_mn, FP_mn, TN_mn, FN_mn)
    calc_mask = calc(TP, FP, TN, FN)
    print("MASK:")
    print("P = %f, R = %f, F1 = %f" % calc_mask)


def ex4():
    KNN = cv2.createBackgroundSubtractorKNN(10, 6.1, 0)
    TP = TN = FP = FN = 0

    first = cv2.imread("%s/input/in%06d.jpg" % (DIR, first_fr))
    mask = KNN.apply(first)

    for i in range(first_fr + 1, last_fr):
        IG = cv2.imread("%s/input/in%06d.jpg" % (DIR, i))
        IG = toGray(IG)
        BG = cv2.imread('%s/groundtruth/gt%06d.png' % (DIR, i))
        BG = toGray(BG)

        # foreground mask (image, fgmask, double learning rate)
        mask = KNN.apply(IG, mask, 0.3)
        IG_mask = doChanges(mask)

        cv2.imshow("purpur", IG_mask)
        cv2.waitKey(10)

        TP, FP, TN, FN = evaluation(mask, BG, TP, FP, TN, FN)

    # print(TP_mn, FP_mn, TN_mn, FN_mn)
    calc_mask = calc(TP, FP, TN, FN)
    print("MASK:")
    print("P = %f, R = %f, F1 = %f" % calc_mask)


# def ex5():


if __name__ == '__main__':
    # frames
    # DIR = "./pedestrian"
    DIR = "./office"
    # DIR = "./highway"

    first_fr, last_fr = frame(DIR)
    # print("ex1:")
    # ex1()
    # print("ex2:")
    # # ex2()
    # print("ex3:")
    # ex3()
    print("ex4:")
    ex4()
