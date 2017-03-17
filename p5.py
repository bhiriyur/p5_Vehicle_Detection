import glob
import os
import pickle
import time

import cv2
import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


#######################################################################################
# STATIC METHODS
#######################################################################################

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Otherwise call with one output
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    :rtype: list
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Define a function to draw bounding boxes"""
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def diag_screen(main_img, diag1=None, diag2=None, diag3=None, diag4=None):
    diag_window = np.zeros((960, 1280, 3), np.uint8)
    diag_window[:720, :1280] = main_img

    # Bottom screens
    if diag1 is not None:
        diag_window[720:960, 0:320] = cv2.resize(diag1, (320, 240), interpolation=cv2.INTER_AREA)

    if diag2 is not None:
        diag_window[720:960, 320:640] = cv2.resize(diag2, (320, 240), interpolation=cv2.INTER_AREA)

    if diag3 is not None:
        diag_window[720:960, 640:960] = cv2.resize(diag3, (320, 240), interpolation=cv2.INTER_AREA)

    if diag4 is not None:
        diag_window[720:960, 960:1280] = cv2.resize(diag4, (320, 240), interpolation=cv2.INTER_AREA)

    return diag_window


#######################################################################################
# PIPELINE CLASS
#######################################################################################
class VehicleDetector(object):
    def __init__(self, vidfile):
        """Vehicle detection pipeline"""
        self.vidfile = vidfile
        self.clip = None

        # Initialize class attributes (video processing)
        self.frameid = 0
        self.bboxes = []
        self.spatial_size = (16, 16)
        self.hist_bins = 16
        self.orient = 9
        self.pixels_per_cell = 8
        self.cells_per_block = 2
        self.colorspace = 'YCrCb'
        self.hog_channel = 'ALL'
        self.classifier = None
        self.x_scaler = None
        self.saved_bboxes = []
        self.num_heatmaps = 10
        self.saved_heatmaps = None
        self.heat_threshold = 1.0
        self.xy_window = (64, 64)
        self.xy_overlap = (0.5, 0.5)
        self.ystart = 400
        self.ystop = 700
        self.scale = 1.0
        self.cells_per_step = 2  # Instead of overlap, define how many cells to step
        self.algo = 'sliding'  # options are 'sliding' or 'subsample' or 'haar'
        self.car_cascade = None

        return

    def classify(self, load_file=None, C=1.0):

        if load_file is not None:
            if os.path.isfile(load_file):
                print("Loading Classifier...")
                data = pickle.load(open(load_file, 'rb'))
                self.classifier = data[0]
                self.x_scaler = data[1]
                return data

        print("Training Classifier...")
        # Prepare Data
        cars = glob.glob('datasets/vehicles/KITTI_extracted/*.png')
        cars += glob.glob('datasets/vehicles/GTI_Far/*.png')
        cars += glob.glob('datasets/vehicles/GTI_Left/*.png')
        cars += glob.glob('datasets/vehicles/GTI_MiddleClose/*.png')
        cars += glob.glob('datasets/vehicles/GTI_Right/*.png')

        notcars = glob.glob('datasets/non-vehicles/GTI/*.png')
        notcars += glob.glob('datasets/non-vehicles/Extras/*.png')

        print("{} Cars, {} notcars".format(len(cars), len(notcars)))

        t = time.time()
        car_features = self.extract_features(cars)
        notcar_features = self.extract_features(notcars)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        x = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        print("(Number-of-samples, feature-vec-length) = {}".format(x.shape))
        # Fit a per-column scaler
        x_scaler = StandardScaler().fit(x)
        # Apply the scaler to X
        scaled_x = x_scaler.transform(x)

        # Split to training/test sets
        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y, test_size=0.2, random_state=rand_state)

        # Initialize and train a classifier
        print('Using:', self.orient, 'orientations', self.pixels_per_cell,
              'pixels per cell and', self.cells_per_block, 'cells per block')
        print('Feature vector length:', len(x_train[0]))

        # Use a linear SVC
        svc = LinearSVC(C=C)

        # Check the training time for the SVC
        t = time.time()
        svc.fit(x_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = {}. Penalty = {}'.format(round(svc.score(x_test, y_test), 4), C))

        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts:     ', svc.predict(x_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

        # Save to pickle file
        pickle.dump([svc, x_scaler], open('classifier.pkl', 'wb'))

        self.classifier = svc
        self.x_scaler = x_scaler

        return svc

    def extract_features(self, imgs, files=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            if files:
                image = mpimg.imread(file)
            else:
                image = file
            # apply color conversion if other than 'RGB'
            if self.colorspace != 'RGB':
                if self.colorspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.colorspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif self.colorspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif self.colorspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif self.colorspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         self.orient, self.pixels_per_cell,
                                                         self.cells_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)

            else:
                hog_features = get_hog_features(feature_image[:, :, self.hog_channel],
                                                self.orient, self.pixels_per_cell,
                                                self.cells_per_block, vis=False,
                                                feature_vec=True)
            # Get color features
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            hist_features = color_hist(feature_image, nbins=self.hist_bins)

            all_features = np.concatenate((spatial_features, hist_features, hog_features))

            # Append the new feature vector to the features list
            features.append(all_features)

        # Return list of feature vectors
        return features

    def search_windows(self, img, windows):

        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        imgs = []
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            imgs.append(test_img)

        features = self.extract_features(imgs, files=False)
        # 5) Transform the features
        test_features = self.x_scaler.transform(features)

        # 6) Predict using your classifier
        predictions = self.classifier.predict(test_features)

        for i, window in enumerate(windows):
            # 7) If positive (prediction == 1) then save the window
            if predictions[i] == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def find_cars_sliding(self, img):
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        windows = slide_window(img, x_start_stop=(None, None), y_start_stop=(self.ystart, self.ystop),
                               xy_window=self.xy_window, xy_overlap=self.xy_overlap)

        bbox_list = self.search_windows(img, windows)

        # window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        diag_window = self.heatmap_filter(draw_img, bbox_list)

        return diag_window

    def find_cars_subsample_multiscale(self, img, scales=(1.0, 1.2, 1.5, 2.0)):
        bbox_list = []
        draw_img = np.copy(img)
        for scale in scales:
            bbox_list = self.find_cars_subsample(img, scale=scale, bbox_list=bbox_list)
        diag_window = self.heatmap_filter(draw_img, bbox_list)
        return diag_window

    def find_cars_subsample(self, img, scale=None, bbox_list=None):
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        if scale is None:
            scale = self.scale

        if bbox_list is None:
            bbox_list = []
            sendbbox = False
        else:
            sendbbox = True

        img_tosearch = img[self.ystart:self.ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, "RGB2{}".format(self.colorspace))
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale),
                                                           np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pixels_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.pixels_per_cell) - 1
        # nfeat_per_block = self.orient * self.cells_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pixels_per_cell) - 1
        nxsteps = (nxblocks - nblocks_per_window) // self.cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // self.cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * self.cells_per_step
                xpos = xb * self.cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                if self.hog_channel == 'ALL':
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                elif self.hog_channel == 0:
                    hog_features = hog_feat1
                elif self.hog_channel == 1:
                    hog_features = hog_feat2
                else:
                    hog_features = hog_feat3

                xleft = xpos * self.pixels_per_cell
                ytop = ypos * self.pixels_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.x_scaler.transform(np.concatenate((spatial_features, hist_features, hog_features)))

                test_prediction = self.classifier.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bbox_list.append([(xbox_left, ytop_draw + self.ystart),
                                      (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)])

        if sendbbox:
            return bbox_list
        else:
            diag_window = self.heatmap_filter(draw_img, bbox_list)
            return diag_window

    def heatmap_filter(self, img, bbox_list):

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        draw_img = np.copy(img)

        # Add heat to each box in box list
        heat = add_heat(heat, bbox_list)

        ####################################################
        # Apply temporal filter
        ####################################################
        if self.frameid == 0:
            shape = (heat.shape[0], heat.shape[1], self.num_heatmaps)
            self.saved_heatmaps = np.zeros(shape)

        idx = self.frameid % self.num_heatmaps
        self.frameid += 1
        self.saved_heatmaps[:, :, idx] = heat

        avg_heat = np.sum(self.saved_heatmaps, axis=2)

        # print("idx = {}, Heat range = {} {}".format(idx, np.min(avg_heat), np.max(avg_heat)))

        # Apply threshold to help remove false positives
        avg_heat_thres = apply_threshold(avg_heat, self.heat_threshold)

        # Visualize the heatmap when displaying
        # heatmap = np.clip(heat, 0, 255)
        heatmap_img = 255 * np.stack((avg_heat, 0 * avg_heat, 0 * avg_heat), axis=2) / np.max(avg_heat)

        # Find final boxes from heatmap using label function
        labels = label(avg_heat_thres)

        ####################################################

        draw_img = self.draw_labeled_bboxes(draw_img, labels)

        # Build diag-screen
        draw_img = diag_screen(draw_img, heatmap_img)

        return draw_img

    def bbox_filter(self, new):

        if len(self.saved_bboxes) == 0:
            return False

        x1n, y1n = new[0]
        x2n, y2n = new[1]

        for old in self.saved_bboxes:
            x1o, y1o = old[0]
            x2o, y2o = old[1]

            # Check if bounding boxes intersect
            if x2n < x1o or x1n > x2o or y2n < y1o or y1n > y2o:
                # Look for next
                continue
            else:
                return True

        return False

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        new_bboxes = []

        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            new_bboxes.append(bbox)

            if self.bbox_filter(bbox):
                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

        self.saved_bboxes = new_bboxes
        # Return the image
        return img

    def haar_classifier(self, img):

        draw_img = np.copy(img)

        if self.car_cascade is None:
            cascade_src = 'cars.xml'
            self.car_cascade = cv2.CascadeClassifier(cascade_src)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray[:self.ystart] = 0
        gray[self.ystop:] = 0

        cars = self.car_cascade.detectMultiScale(gray, 1.1, 1)

        for (x, y, w, h) in cars:
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return draw_img

    def video_process(self, start_stop=None, preview=True, save_output=False):
        """Video pipeline"""
        self.clip = VideoFileClip(self.vidfile)
        if start_stop is not None:
            self.clip = self.clip.subclip(start_stop[0], start_stop[1])

        # Process the pipeline
        if self.algo == 'sliding':
            func = self.find_cars_sliding
        elif self.algo == 'subsample':
            func = self.find_cars_subsample_multiscale
        elif self.algo == 'haar':
            func = self.haar_classifier
        else:
            raise ValueError("Invalid algo = {}".format(self.algo))
        white_clip = self.clip.fl_image(func)

        if preview:
            white_clip.preview(fps=25)

        if save_output:
            outfile = self.vidfile.split('.')[0] + '_out.mp4'
            white_clip.write_videofile(outfile, audio=False)


#######################################################################################
# MAIN THREAD
#######################################################################################
if __name__ == '__main__':
    # Load video and call pipeline
    V = VehicleDetector('project_video.mp4')
    V.classify(load_file='classifier.pkl', C=0.001) #'classifier.pkl'

    V.algo = 'subsample'
    V.heat_threshold = 15
    # Below options only matters if V.algo == 'subsample'
    V.cells_per_step = 2
    # V.scale = 1.2

    V.video_process(start_stop=None, preview=False, save_output=True)
    # V.video_process(start_stop=(6, 10), preview=True, save_output=False)

    # for test_img in glob.glob('test_images/*.jpg'):
    #     img = mpimg.imread(test_img)
    #     print(test_img)
    #     out = V.find_cars(img, scale=1.0)
    #     plt.imshow(out)
    #     plt.show()
