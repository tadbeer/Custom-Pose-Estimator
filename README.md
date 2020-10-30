# Custom-Pose-Estimator
To determine the three key points of a horizontal shape in the image.

Input shall be an image with a horizontal object in it. The object shall be the whole image across long. It wouldn't necessarily be a straight line.

Representative image for this task
![alt text]()

The objective was to determine the keypoint of the horizontal object. These keypoints shall be used to obtain an approximate segmentation of the horizontal object, which shall be used for subsequent analysis of the horizontal object.
Three keypoints were determined to be sufficient for keypoint estimation. (marked with 'x' in above image.)

The given code builds a model that extracts features from an image using a resent backbone. The feature map is then processed by splitting into three and iteratively spatially compressed to a 6 length vector. The spatial compression takes into account the visual properties of the horizontal object and convolutional pooling of features along desired axes.

The script pose_by3.py builds and trains the model.

The script mark_pose.py can be used to build dataset for this model, which would iterate over a set of images and taking inputs from user as three clicks on the image, at the three keypoints.
The script pose_by3_data.py consists of helper function to preprocess such data for the training.

The script pose_by3_modellers.py consists of the custom Keras layers used in pose_by3.py.
