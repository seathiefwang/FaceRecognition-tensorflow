import cv2
import dlib
import os
import random
import imutil_class

output_dir = './my_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

# 改变图片尺寸
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
webcam = imutil_class.WebcamVideoStream(src=0).start()

index = 1
while True:
    if (index <= 10000):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
        img = webcam.read()
        img = resize(img, width=500)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            try:
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1, x2:y2]
                    # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                    face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                    cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                    index += 1
            except Exception:
                continue
            cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break

cv2.destroyAllWindows()
webcam.stop()
