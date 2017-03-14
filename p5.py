import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import cv2
import glob
from moviepy.editor import VideoFileClip

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label


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


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img
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
        self.hog_channel = 2
        self.classifier = None
        self.X_scaler = None
        self.heatmap = []

        return

    def classify(self, load_file=None):

        if load_file != None:
            print("Loading Classifier...")
            data = pickle.load(open(load_file, 'rb'))
            self.svc = data[0]
            self.X_scaler = data[1]
            return data

        print("Training Classifier...")
        # Prepare Data
        cars = glob.glob('datasets/vehicles/KITTI_extracted/*.png')
        notcars = glob.glob('datasets/non-vehicles/GTI/*.png')
        #notcars += glob.glob('datasets/non-vehicles/Extras/*.png')

        print("{} Cars, {} notcars".format(len(cars),len(notcars)))

        t = time.time()
        car_features = self.extract_features(cars)
        notcar_features = self.extract_features(notcars)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        print ("New-shape = {}".format(X.shape))
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Split to training/test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # Initialize and train a classifier
        print('Using:', self.orient, 'orientations', self.pixels_per_cell,
              'pixels per cell and', self.cells_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC
        svc = LinearSVC()

        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

        # Save to pickle file
        pickle.dump([svc, X_scaler], open('classifier.pkl', 'wb'))

        self.classifier = svc
        self.X_scaler = X_scaler

        return svc

    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if self.colorspace != 'RGB':
                if self.colorspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.colorspace== 'LUV':
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

    def find_cars(self, img, ystart=400, ystop=700, scale=1.0):
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, "RGB2{}".format(self.colorspace))
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pixels_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.pixels_per_cell) - 1
        nfeat_per_block = self.orient * self.cells_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pixels_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pixels_per_cell, self.cells_per_block, feature_vec=False)

        bbox_list = []
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

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
                elif self.hog_channel == 2:
                    hog_features = hog_feat3

                xleft = xpos * self.pixels_per_cell
                ytop = ypos * self.pixels_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.concatenate((spatial_features, hist_features, hog_features)))

                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                    bbox_list.append([(xbox_left, ytop_draw + ystart),
                                      (xbox_left + win_draw, ytop_draw + win_draw + ystart)])

        # Add heat to each box in box list
        heat = add_heat(heat, bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(draw_img, labels)

        return draw_img

    def video_process(self, start_stop=None, preview=True, save_output=False):
        """Video pipeline"""
        self.clip = VideoFileClip(self.vidfile)
        if start_stop is not None:
            self.clip = self.clip.subclip(start_stop[0], start_stop[1])

        # Process the pipeline
        white_clip = self.clip.fl_image(self.find_cars)

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
    V.classify(load_file='classifier.pkl')

    V.video_process(preview=False, save_output=True)

    # for test_img in glob.glob('test_images/*.jpg'):
    #     img = mpimg.imread(test_img)
    #     print(test_img)
    #     out = V.find_cars(img, scale=1.0)
    #     plt.imshow(out)
    #     plt.show()