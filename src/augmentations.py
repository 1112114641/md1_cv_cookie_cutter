"""
Some augmentations for image data.

Individual transformations can all be called using ImageAugment given a configuration
yaml file similar to `artefacts/configs/data_loader.yml`.

TODO: create augment class, read in parameters, create list of augmentation functions
TODO: make class callable for `DataLoader`
"""
# from typing import Tuple, List
# import tensorflow as tf


# def sample_beta_distribution(
#     size: int, concentration_0: float = 0.2, concentration_1: float = 0.2
# ) -> float:
#     gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# @tf.function
# def cutmix(
#     train_ds_one: tf.data.Dataset, train_ds_two: tf.data.Dataset
# ) -> Tuple[tf.data.Dataset]:
#     (image1, label1), (image2, label2) = train_ds_one, train_ds_two

#     alpha, beta = [0.25], [0.25]
#     lambda_value = sample_beta_distribution(1, alpha, beta)
#     lambda_value = lambda_value[0][0]

#     # Get the bounding box offsets, heights and widths
#     boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

#     # Get a patch from the second image (`image2`)
#     crop2 = tf.image.crop_to_bounding_box(
#         image2, boundaryy1, boundaryx1, target_h, target_w
#     )
#     # Pad the `image2` patch (`crop2`) with the same offset
#     image2 = tf.image.pad_to_bounding_box(
#         crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
#     )
#     # Get a patch from the first image (`image1`)
#     crop1 = tf.image.crop_to_bounding_box(
#         image1, boundaryy1, boundaryx1, target_h, target_w
#     )
#     # Pad the `image1` patch (`crop1`) with the same offset
#     img1 = tf.image.pad_to_bounding_box(
#         crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
#     )

#     Modify the first image by subtracting the patch from `image1`
#     (before applying the `image2` patch)
#     image1 = image1 - img1
#     Add the modified `image1` and `image2`  together to get the CutMix image
#     image = image1 + image2

#     # Adjust Lambda in accordance to the pixel ration
#     # lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
#     lambda_value = tf.cast(lambda_value, tf.float32)

#     # Combine the labels of both images
#     label = lambda_value * label1 + (1 - lambda_value) * label2
#     return image, label


# @tf.function
# def mixup(ds_one: tf.data.Dataset, ds_two: tf.data.Dataset, alpha=0.2):
#     # Unpack two datasets
#     images_one, labels_one = ds_one
#     images_two, labels_two = ds_two
#     batch_size = tf.shape(images_one)[0]

#     # Sample lambda and reshape it to do the mixup
#     lambda_ = sample_beta_distribution(batch_size, alpha, alpha)
#     x_l = tf.reshape(lambda_, (batch_size, 1, 1, 1))
#     y_l = tf.reshape(lambda_, (batch_size, 1))

#     # Perform mixup on both images and labels by combining a pair of images/labels
#     # (one from each dataset) into one image/label
#     images = images_one * x_l + images_two * (1 - x_l)
#     labels = labels_one * y_l + labels_two * (1 - y_l)
#     return (images, labels)


# def dropout_area(ds_one: tf.data.Dataset, size: float, fill_value: float):
#     images_one, labels = ds_one
#     # random generate small numbers within x,y<size
#     x1, x2, y1, y2 = 1, 1, 1, 1
#     holes = [x1, x2, y1, y2]
#     for x1, y1, x2, y2 in holes:
#         images_one[y1:y2, x1:x2] = fill_value
#     # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/dropout/grid_dropout.py#L47  # noqa: E501
#     # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/dropout/coarse_dropout.py#L48  # noqa: E501
#     images = 1  # black_out_randm_patch
#     return (images, labels)


# @tf.function
# def random_center_crop(image):
#     crop_image = image
#     return crop_image


# @tf.function
# def dropout_pixel_grid(
#     ds_one: tf.data.Dataset,
#     size: int,
#     holes: List[int],
#     frequency: float,
#     fill_value: float,
# ):
#     images_one, labels = ds_one
#     # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/dropout/grid_dropout.py#L47  # noqa: E501
#     # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/dropout/coarse_dropout.py#L48  # noqa: E501
#     for x1, y1, x2, y2 in holes:
#         images = dropout_area(images_one, size, fill_value)  # black_out_random_patch
#     return (images, labels)


# for i in range(21):
#     print(
#         i,
#         "l" if (i - 1) % 5 == 0 else "",
#         "h" if (i - 2) % 5 == 0 else "",
#         "w" if (i - 3) % 5 == 0 else "",
#         "x" if (i - 4) % 5 == 0 else "",
#         "y" if ((i - 5) % 5 == 0) & (i > 2) else "",
#     )


###
#
# data augmentation calls -
#
###
# if self.config.augment_params["brightness"]:
#     image = tf.image.random_brightness(
#         image, max_delta=0.2, seed=self.config.augment_params["random_seed"]
#     )
# if self.config.augment_params["flip_updown"]:
#     image = tf.image.random_flip_up_down(  # FIXME: fix for object detection
#         image, seed=self.config.augment_params["random_seed"]
#     )
# if self.config.augment_params["flip_leftright"]:
#     image = (
#         tf.image.random_flip_left_right(  # FIXME: fix for object detection
#             image, seed=self.config.augment_params["random_seed"]
#         )
#     )
# if self.config.augment_params["crop"]:
#     image = random_center_crop(  # FIXME: fix for object detection
#         image,
#         offset=self.config.augment_params["crop"],
#         seed=self.config.augment_params["random_seed"],
#     )
# if self.config.augment_params["contrast"]:
#     image = tf.image.random_contrast(
#         image,
#         lower=0.2,
#         upper=1.0,
#         seed=self.config.augment_params["random_seed"],
#     )
# if self.config.augment_params["quality"]:
#     image = tf.image.random_jpeg_quality(
#         image,
#         min_jpeg_quality=80,
#         max_jpeg_quality=100,
#         seed=self.config.augment_params["random_seed"],
#     )
# if self.config.augment_params["saturation"]:
#     image = tf.image.random_saturation(
#         image,
#         lower=80,
#         upper=100,
#         seed=self.config.augment_params["random_seed"],
#     )
# if self.config.augment_params["dropout_area"]:
#     image = dropout_area(  # FIXME: fix for object detection
#         image,
#         lower=5,
#         upper=15,
#         seed=self.config.augment_params["random_seed"],
#     )
