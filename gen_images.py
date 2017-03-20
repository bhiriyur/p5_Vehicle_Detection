from p5 import *


V = VehicleDetector('project_video.mp4')
V.classify(load_file='classifier.pkl', C=0.001)


# Classifier Performance Images
cars, noncars = [], []
cars.append('datasets/vehicles/GTI_MiddleClose/image0009.png')
cars.append('datasets/vehicles/KITTI_extracted/18.png')
cars.append('datasets/vehicles/KITTI_extracted/32.png')

noncars.append('datasets/non-vehicles/Extras/extra1.png')
noncars.append('datasets/non-vehicles/GTI/image42.png')
noncars.append('datasets/non-vehicles/GTI/image8.png')

cars_feat = V.extract_features(cars)
noncars_feat = V.extract_features(noncars)

cars_feat = V.x_scaler.transform(cars_feat)
noncars_feat = V.x_scaler.transform(noncars_feat)

car_predictions = V.classifier.predict(cars_feat)
noncar_predictions = V.classifier.predict(noncars_feat)

fig = plt.figure(figsize=(4, 6))
fig.canvas.set_window_title('Car/Noncar predictions')
for i in range(3):

    car_img = plt.imread(cars[i])
    noncar_img = plt.imread(noncars[i])

    plt.subplot(3,2,2*i+1)
    plt.imshow(car_img)
    plt.xticks([])
    plt.yticks([])
    plt.title("Prediction = {:.0f}".format(car_predictions[i]))

    plt.subplot(3,2,2*i+2)
    plt.imshow(noncar_img)
    plt.xticks([])
    plt.yticks([])
    plt.title("Prediction = {:.0f}".format(noncar_predictions[i]))

plt.tight_layout()
fig.savefig('output_images/figure_1.png')

fig = plt.figure()
plt.bar(range(len(cars_feat[0])), cars_feat[0],color='b')
plt.title('Example Feature Vector (after normalization)')
fig.savefig('output_images/figure_2.png')

img = plt.imread('test_images/test5.jpg')
fig=V.draw_subsample_windows(img, scale=1.0)
plt.title('Scale = 1.0')
fig.savefig('output_images/figure_3.png')

fig=V.draw_subsample_windows(img, scale=1.5)
plt.title('Scale = 1.5')
fig.savefig('output_images/figure_4.png')

fig=V.draw_subsample_windows(img, scale=2.0)
plt.title('Scale = 2.0')
fig.savefig('output_images/figure_5.png')

scale = 2.0
bbox_list = V.find_cars_subsample(img, scale=scale, bbox_list=[], heat_filter=False)
draw_img = np.copy(img)
heat = np.zeros_like(img[:, :, 0]).astype(np.float)
heat = add_heat(heat, bbox_list)
heatmap_img = (255 * np.stack((heat, 0*heat, 0*heat), axis=2) / np.max(heat)).astype(np.uint8)
labels = label(heat)
out = V.draw_labeled_bboxes(img, labels=labels)

fig = plt.figure()
plt.subplot(211)
plt.imshow(heatmap_img)
plt.title('Heatmap - Scale = {}'.format(scale))
plt.xticks([])
plt.yticks([])

plt.subplot(212)
plt.imshow(out)
plt.title('Boxes - Scale = {}'.format(scale))
plt.xticks([])
plt.yticks([])

fig.savefig('output_images/figure_6.png')

