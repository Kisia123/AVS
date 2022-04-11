import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle


def ex1_1():
    cv2.imshow("Mandrill", I)
    cv2.imwrite("myMandrill1.png", I)
    print(I.shape)
    print(I.size)
    print(I.dtype)


def ex1_2():
    # plt.figure(1)
    fig, ax = plt.subplots(1)
    rect = Rectangle((50, 50), 50, 100, fill=False, ec="r")
    ax.add_patch(rect)
    plt.imshow(I)
    plt.title("Mandrill")
    plt.imsave("myMandrill2.png", I)
    plt.axis("off")
    plt.show()
    x = [100, 150, 200, 250]
    y = [50, 100, 150, 200]
    plt.plot(x, y, "r.", markersize=10)
    plt.show()


def ex1_3():
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    IH = IHSV[:, :, 0]
    IS = IHSV[:, :, 1]
    IV = IHSV[:, :, 2]

    plt.imshow(IG)
    plt.gray()
    plt.title("MandrillGray")
    plt.axis("off")
    plt.show()

    plt.imshow(IH)
    plt.gray()
    plt.title("Mandrill_IH")
    plt.axis("off")
    plt.show()

    plt.imshow(IS)
    plt.gray()
    plt.title("Mandrill_IS")
    plt.axis("off")
    plt.show()

    plt.imshow(IV)
    plt.gray()
    plt.title("Mandrill_IV")
    plt.axis("off")
    plt.show()
    # cv2.imshow("Mandrill", IG)

    IMy = rgb2gray(I)
    plt.imshow(IMy)
    plt.title("Mandrill MyGray")
    plt.axis("off")
    plt.gray()
    plt.show()

    _HSV = matplotlib.colors.rgb_to_hsv(I)
    cv2.imshow("Mandrill HSV", _HSV)


def ex1_4():
    height, width = I.shape[:2]
    scale = 1.75
    Ix2 = cv2.resize(I, (int(scale * width), int(scale * height)))
    cv2.imshow("Mandrill Bigger", Ix2)


def rgb2gray(I):
    return 0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.144 * I[:, :, 2]


def ex1_5():
    GL = rgb2gray(L)
    GM = rgb2gray(I)
    h, w = GM.shape[:2]
    GL = cv2.resize(GL, (w, h))
    plt.imshow(GL + GM)
    plt.title("Mandrill+Lena=<3")
    plt.axis("off")
    plt.gray()
    plt.show()

    plt.imshow(GM - GL)
    plt.title("Mandrill-Lena=:(")
    plt.axis("off")
    plt.gray()
    plt.show()

    plt.imshow(GL * GM)
    plt.title("Mandrill*Lena=***")
    plt.axis("off")
    plt.gray()
    plt.show()

    ILinear = linear([(GM, 0.2), (GL, 1.0)])
    plt.imshow(ILinear)
    plt.title("Linear Image")
    plt.axis("off")
    plt.gray()
    plt.show()

    IMOD = GM[:, :]
    cv2.absdiff(GL, GM, IMOD)
    plt.imshow(IMOD)
    plt.title("MOD")
    plt.axis("off")
    plt.gray()
    plt.show()

    cv2.imshow("MODC", np.uint8(IMOD))


def ex1_6():
    IM = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    H = hist(IM)
    plt.title("H1")
    plt.hist(H)
    plt.show()

    H2 = cv2.calcHist([IM], [0], None, [256], [0, 256])
    plt.title("H2")
    plt.hist(H2)
    plt.show()

    IGE = cv2.equalizeHist(IM)
    plt.title("HIST - EQ")
    plt.hist(IGE)
    plt.show()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    I_CLAHE = clahe.apply(IM)
    plt.title("CLAHE")
    plt.hist(I_CLAHE)
    plt.show()


def ex1_7():
    GAUSS = cv2.GaussianBlur(I, (5, 5), 0, 0, cv2.BORDER_WRAP)
    cv2.imshow("GAUSS", GAUSS)

    SOBEL = cv2.Sobel(I, cv2.CV_32F, 1, 0)
    cv2.imshow("SOBEL", SOBEL)

    LAP = cv2.Laplacian(I, cv2.CV_32F)
    cv2.imshow("LAP", LAP)

    MED = cv2.medianBlur(I, 13)
    cv2.imshow("MED", MED)


def linear(arr):
    h, w = arr[0][0].shape[:2]
    empty = h * [[0] * w]
    for img, weight in arr:
        empty += img * weight
    return empty


def hist(img):
    hist = np.zeros((256, 1), np.float32)
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            hist[int(img[y][x])] += 1
    return hist


if __name__ == '__main__':
    I = cv2.imread("mandrill.jpg")
    L = cv2.imread("lena.png")
    # ex1_1()
    # I = plt.imread("mandrill.jpg")
    # ex1_2()
    # ex1_3()
    # ex1_4()
    # ex1_5()
    # ex1_6()
    ex1_7()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
