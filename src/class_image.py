import cv2
import matplotlib.pyplot as plt
import numpy as np

class Image:
    
    def __init__(self, path, extension, gamma, side_size):

        self.path = path
        self.extension = extension
        self.img = cv2.imread(path + '/img.' + extension)
        self.gamma = gamma
        self.side_size = side_size
        self.recognized = False
        self.num_triangles = -1
        self.ans = []

    def recognize_circles(self):

        self.recognized = True

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        GAMMA = self.gamma
        SIDE_SIZE = self.side_size
        LEFT_BOUNDARY = (SIDE_SIZE - 15) ** 2 * np.sqrt(3) / 4
        RIGHT_BOUNDARY = (SIDE_SIZE + 15) ** 2 * np.sqrt(3) / 4

        def gamma_pict(hsv):

            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, GAMMA) * 255.0, 0, 255)

            return np.where(cv2.LUT(hsv[:,:,0], lookUpTable)>12, 255,0).astype(np.uint8)

        def get_threshold(img):
    
            counts, bins = np.histogram(img.ravel(), bins=256, range=[0, 256], density=True)
            bins = bins[:-1]
            
            def smooth(y, box_pts):
                box = np.ones(box_pts) / box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            
            smooth_freq = smooth(counts, 39)
            
            st = 225
            while smooth_freq[st - 1] >= smooth_freq[st]:
                st -= 1
            while smooth_freq[st - 1] <= smooth_freq[st]:
                st -= 1
            return st - 7

        bin_img = np.where(hsv[:, :, 1] >= get_threshold(hsv[:,:,1]), 255, 0).astype(np.uint8)
        dst = cv2.medianBlur(bin_img, 7)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, np.ones((10, 10)))

        (numLabels, labels, stats, centroids) = output = cv2.connectedComponentsWithStats(dst, 8, cv2.CV_32S)

        def rotate_120(x, y):
            
            angle = 2 * np.pi / 3
            return np.cos(angle) * x - np.sin(angle) * y, np.sin(angle) * x + np.cos(angle) * y

        def detect_triangle(label):

            h, w = labels.shape
            cX, cY = centroids[label]

            max_ans = 0.
            max_i = -1
            max_j = -1

            for i in range(h):
                for j in range(w):

                    if labels[i][j] == label and max_ans < np.sqrt((j - cX) ** 2 + (i - cY) ** 2):
                        max_ans = np.sqrt((j - cX) ** 2 + (i - cY) ** 2)
                        max_i = i
                        max_j = j

            x1, y1 = max_j, max_i

            x2, y2 = rotate_120(x1 - cX, y1 - cY)
            x2, y2 = x2 + cX, y2 + cY

            x3, y3 = rotate_120(x2 - cX, y2 - cY)
            x3, y3 = x3 + cX, y3 + cY

            return x1, y1, x2, y2, x3, y3

        def crop_triangle(img, x1, y1, x2, y2, x3, y3):

            pts = np.array([[x1,y1],[x2,y2],[x3,y3]])

            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = img[y:y+h, x:x+w].copy()

            pts = pts - pts.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            return dst, (x1 - x, y1 - y, x2 - x, y2 - y, x3 - x, y3 - y)

        detected = 0

        cropped = []

        for i in range(0, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            if LEFT_BOUNDARY >= area or area >= RIGHT_BOUNDARY:
                continue
            detected += 1
            
            x1, y1, x2, y2, x3, y3 = detect_triangle(i)
            curr_cropped, pts = crop_triangle(hsv, x1, y1, int(x2), int(y2), int(x3), int(y3))
            
            center_coords = [int((x1 + x2 + x3) / 3), int((y1 + y2 + y3) / 3)]

            cropped.append((gamma_pict(curr_cropped), pts, center_coords))

        for i in range(detected):
            x1, y1, x2, y2, x3, y3 = cropped[i][1]
            
            blurred = cv2.medianBlur(cropped[i][0], 7)
            
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(blurred, 8, cv2.CV_32S)
            ans = [0, 0, 0]
            for j in range(numLabels):
                area = stats[j, cv2.CC_STAT_AREA]
                if area >= 10 and area <= 70:
                    (cX, cY) = centroids[j]
                    dists = np.array([(x1 - cX) ** 2 + (y1 - cY) ** 2,
                                      (x2 - cX) ** 2 + (y2 - cY) ** 2,
                                      (x3 - cX) ** 2 + (y3 - cY) ** 2])
                    ans[np.argmin(dists)] += 1

            self.ans.append(f'{cropped[i][2][0]}, {cropped[i][2][1]}: {ans[0]}, {ans[1]}, {ans[2]}')

        self.num_triangles = len(self.ans)
