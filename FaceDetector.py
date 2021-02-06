import cv2
import numpy as np
import os
import time

'''
Get the Face Image from Key Frame
Reference:https://blog.csdn.net/haohuajie1988/article/details/79163318?depth_1-utm_source=distribute.pc_relevant.none-task
Author:Zhutian Lin
Date:2020-3-27
'''
class FaceDetector:
    def __init__(self, target_path, classifier_path):
        '''
        initialize model params
        :param target_path: Target Path to Save Face Image,str
        :param classifier_path: The OpenCV Classifier Path,str
        '''
        self.image_path = []
        self.target_path = target_path
        self.classifier_path = classifier_path

    def load_file_folder(self, file_folder_path, *suffix):
        '''
        Load the Source Image Folder to The Classifier
        :param file_folder_path: Source Folder Path,str
        :param suffix: Image Suffix,str
        :return:
        '''
        for r, ds, fs in os.walk(file_folder_path):
            for fn in fs:
                if os.path.splitext(fn)[1] in suffix:
                    file_name = os.path.join(r, fn)
                    self.image_path.append(file_name)
        return

    def Extract_face_image(self):
        '''
        Extract Face Image and Save.
        :return:
        '''
        cnt = 1
        try:
            face_cascade = cv2.CascadeClassifier(self.classifier_path)
            face_cascade.load(self.classifier_path)
            for img_path in self.image_path:
                img = cv2.imread(img_path)
                faces = face_cascade.detectMultiScale(img, minNeighbors=5)
                for (x, y, w, h) in faces:
                    if w >= 128 and h >= 128:
                        listStr = [str(int(time.time())), str(cnt)]
                        file_name = '_'.join(listStr)

                        X = int(x * 0.5)
                        W = min(int((x + w) * 1.2), img.shape[1])
                        Y = int(y * 0.3)
                        H = min(int((y + h) * 1.4), img.shape[0])

                        f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                        cv2.imwrite(self.target_path + os.sep + '%s.jpg' % file_name, f)
                        cnt += 1
                        print(img_path + " have face")

        except Exception:
            print(Exception.__traceback__)


if __name__ == '__main__':
    fd = FaceDetector('face_img', 'D:\\anaconda\\pkgs\\libopencv-3.4.2-h20b85fd_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    fd.load_file_folder('img', '.jpg')
    fd.Extract_face_image()
