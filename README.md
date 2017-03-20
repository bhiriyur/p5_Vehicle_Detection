##Project 5 - Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (TODO: Update Images) 

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Submission Summary
* The file **p5.py** contains the main class **VehicleDetector** and various supporting functions implemented in this project. 
* The folder **datasets** contains all the training images used by the classifier
* **README.md** - This write-up report. Links the images in **output_images** folder.
    
Further details of the project implementation are provided below (following the rubric points). The final project video is linked at the end.

This project is hosted On [Github](https://github.com/bhiriyur/p5_Vehicle_Detection).


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I am using the write-up template provided. Here it is!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The full feature list that is used by the classification algorithm uses all of the following:
1. HOG features (extracted from a static method ```get_hog_featres``` reproduced below:

```python
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
```

2. A histogram of spatial features extracted from static method ```bin_spatial``` reproduced below:
```python
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

```

3. And finally, a histogram of color features extracted from the static method ```color_hist``` reproduced below:

```python
def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

```

All of the above were called by a single method called ```extract_features``` (of class ```VehicleDetector```) to build the full feature list for classification. 

####2. Explain how you settled on your final choice of HOG parameters.

After much experimentation, the following parameter values were chosen for all of the above feature selection and were found to work best:

```python
    self.spatial_size = (16, 16)
    self.hist_bins = 16
    self.orient = 9
    self.pixels_per_cell = 8
    self.cells_per_block = 2
    self.colorspace = 'YCrCb'
    self.hog_channel = 'ALL'
```

As seen above, the colorspace **YCrCb** was used with all three channels. This of course made the feature vector of length **6108** and thereby increased the processing time for each frame by a significant amount, however the classification test accuracy seemed to be quite high using all of these parameters and so they were all retained for the final pipeline.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The ```classify``` method of the ```VehicleDetector``` class implements the classification pipeline. A linear support vector machine is used with the aforementioned feature vectors. The training set consists of 8792 car images and 8968 non car images. The entire dataset was divided into training set (80%) and test set (20%) and the results of the classifier training and validation are shown below:

```text
Training Classifier...
8792 Cars, 8968 notcars
171.59 Seconds to extract HOG features...
(Number-of-samples, feature-vec-length) = (17760, 6108)
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
14.2 Seconds to train SVC...
Test Accuracy of SVC = 0.9932. Penalty = 0.001
My SVC predicts:      [ 0.  1.  1.  0.  1.  1.  1.  1.  1.  0.]
For these 10 labels:  [ 0.  1.  1.  0.  1.  1.  1.  1.  1.  0.]
0.22301 Seconds to predict 10 labels with SVC
```

As noted above, the test accuracy is extremely high at 99.32% and I used a low value of 0.001 for the penalty parameter **C** of the LinearSVC to keep the decision boundary relatively smooth. The classifier and the normalizer were stored in a pickle file when used for the actual pipeline.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
A HOG subsampling approach was used to find the location of the cars within an image. This approach was implemented in the method ```find_cars_subsample```. In this method, the hog features for the entire image was extracted once for a given scale. From these features, subsampling was done for windows of size 64 x 64 (size of the training images for classifier) and passed through to the classifier for prediction.

```python
    def find_cars_subsample(self, img, scale=None, bbox_list=None):
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255


        bbox_list = []

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

        diag_window = self.heatmap_filter(draw_img, bbox_list)
        return diag_window

```

The relevant parameter values chosen for the final video pipeline (after some trial and error) were as follows:
```text
self.pixels_per_cell = 8
self.cells_per_block = 2
self.cells_per_step = 2  # Instead of overlap, define how many cells to step
```

For the final video pipeline, a multiscale approach using the method reproduced below was adopted with the following scales: ```scales=(1.0, 1.2, 1.5, 2.0)``` 
```python
    def find_cars_subsample_multiscale(self, img, scales=(1.0, 1.2, 1.5, 2.0)):
        bbox_list = []
        draw_img = np.copy(img)
        for scale in scales:
            bbox_list = self.find_cars_subsample(img, scale=scale, bbox_list=bbox_list)
        diag_window = self.heatmap_filter(draw_img, bbox_list)
        return diag_window
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is a link to my video result for the project video.

[![Video](http://img.youtube.com/vi/h0HUkiVbQqw/0.jpg)](http://www.youtube.com/watch?v=h0HUkiVbQqw "Advanced Lane Finding")


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

