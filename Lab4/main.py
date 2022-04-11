import numpy as np
import cv2
import math


def toGray(IMG):
    return cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)


def HSVtoRGB(IMG):
    return cv2.cvtColor(np.uint8(IMG), cv2.COLOR_HSV2RGB)


def images():
    # images I and J
    I = cv2.imread("./I.jpg")
    I = toGray(I)
    J = cv2.imread("./J.jpg")
    J = toGray(J)
    return I, J


def ex_1():
    I, J = images()
    # difference with absdiff
    # J = cv2.resize(J, (w, h))
    # IMOD = I[:, :]
    # cv2.absdiff(I, J, IMOD)
    # cv2.imshow("Diff", IMOD)
    # cv2.waitKey(1000)

    W2 = 5  # half the size of the window
    dX = dY = 5  # size searched in two directions

    h, w = I.shape[:2]
    # coords of minima
    u = np.zeros((h, w), np.float32)
    v = np.zeros((h, w), np.float32)
    result = np.zeros((h, w), np.float32)

    # i - lines, j - columns
    for i in range(W2 + dX, h - W2 - dX):
        for j in range(W2 + dX, w - W2 - dY):
            #  no optical flow on edges

            IO = np.float32(I[
                            i - W2:i + W2 + 1,
                            j - W2:j + W2 + 1])  # cutting part of the I frame
            s_dis = math.inf
            c = (0, 0)
            # image patch 7 x 7
            for x in range(-dX, dX + 1):
                for y in range(-dY, dY + 1):
                    JO = np.float32(J[
                                    x + i - W2:x + i + W2 + 1,
                                    y + j - W2:y + j + W2 + 1])
                    dis = np.sum(np.sqrt(np.square(JO - IO)))

                    # smallest distance finder
                    if dis < s_dis:
                        s_dis = dis
                        c = (x, y)

                    # if c != (0, 0):
                    #     print(c)

            u[i, j] = c[0]
            v[i, j] = c[1]
    mag, angle = cv2.cartToPolar(v, u)  # mag - length of vector
    HSV = np.zeros((h, w, 3), np.float32)  # creating 3D table
    HSV[:, :, 0] = angle * 90 / np.pi
    HSV[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    HSV[:, :, 2] = 255
    OUT_IMAGE = HSVtoRGB(HSV)

    cv2.imwrite("image.png", OUT_IMAGE)
    cv2.imshow("opt", OUT_IMAGE)
    cv2.waitKey(0)


def im_pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k - 1], (0, 0), fx=0.5, fy=0.5))
    return images


def vis_flow(u, v, YX, name):
    # YX - im dim
    mag, angle = cv2.cartToPolar(v, u)  # mag - lenght of vec
    HSV = np.zeros((YX[0], YX[1], 3), np.float32)  # creating 3D table
    HSV[:, :, 0] = angle * 90 / np.pi
    HSV[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    HSV[:, :, 2] = 255
    OUT_IMAGE = HSVtoRGB(HSV)

    cv2.imwrite(f"{name}.jpg", OUT_IMAGE)
    cv2.imshow(name, OUT_IMAGE)
    cv2.waitKey(0)


def of(I, J, W2, dX, dY):
    # difference with absdiff
    # J = cv2.resize(J, (w, h))
    # IMOD = I[:, :]
    # cv2.absdiff(I, J, IMOD)
    # cv2.imshow("Diff", IMOD)
    # cv2.waitKey(1000)

    h, w = I.shape[:2]
    # coords of minima
    u = np.zeros((h, w), np.float32)
    v = np.zeros((h, w), np.float32)
    result = np.zeros((h, w), np.float32)

    # i - lines, j - columns
    for i in range(W2 + dX, h - W2 - dX):
        for j in range(W2 + dX, w - W2 - dY):
            #  no optical flow on edges

            IO = np.float32(I[
                            i - W2:i + W2 + 1,
                            j - W2:j + W2 + 1])  # cutting part of the I frame
            s_dis = math.inf
            c = (0, 0)
            # image patch 7 x 7
            for x in range(-dX, dX + 1):
                for y in range(-dY, dY + 1):
                    JO = np.float32(J[
                                    x + i - W2:x + i + W2 + 1,
                                    y + j - W2:y + j + W2 + 1])
                    dis = np.sum(np.sqrt(np.square(JO - IO)))

                    # smallest distance finder
                    if dis < s_dis:
                        s_dis = dis
                        c = (x, y)

                    # if c != (0, 0):
                    #     print(c)

            u[i, j] = c[0]
            v[i, j] = c[1]

    return u, v


def ex_2():
    I, J = images()
    I_org = I[:, :]  # original image saved
    scale = 2
    IP = im_pyramid(I, scale)  # scal
    IP.reverse()  # changing indexes
    JP = im_pyramid(J, scale)
    JP.reverse()
    I = IP[0]  #smallest image
    W2 = dX = dY = 5
    flow_array = []  #empty array
    for i in range(0, scale):
        J = JP[i]  # to have J ready
        u, v = of(I, J, W2, dX, dY)
        flow_array.append((u, v))
        I_new = I[:, :]
        h, w = I_new.shape[:2]
        if i != scale - 1:
            for x in range(0, h):
                for y in range(0, w):
                    flow_x = int(u[x, y])
                    flow_y = int(v[x, y])
                    I_new[x, y] = I[x + flow_x, y + flow_y]
        I = cv2.resize(I_new, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        print("I am calculating flows")
    h, w = I_org.shape[:2]
    total_flow_u = np.zeros((h, w), np.float32)
    total_flow_v = np.zeros((h, w), np.float32)
    for i in range(0, scale):
        u, v = flow_array[i]
        needed_scale = total_flow_u.shape[0] / u.shape[0]
        U = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        V = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
        U *= needed_scale
        V *= needed_scale

        total_flow_u += U
        total_flow_v += V
        print("I am calculating total flow")
        vis_flow(total_flow_u, total_flow_v, (h, w), "flowing")


if __name__ == '__main__':
    # ex_1()
    ex_2()
