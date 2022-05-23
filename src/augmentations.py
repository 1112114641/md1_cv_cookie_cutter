"""
Some augmentations for image data.

Individual transformations can all be called using ImageAugment given a configuration
yaml file similar to `artefacts/configs/data_loader.yml`.


structure:
augmentation functions
random generator
class Augment

TODO: create augment class, read in parameters, create list of augmentation functions
TODO: make class callable for `DataLoader`
"""
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from src.abstracts import AugmentABC

# import tensorflow_probability as tfp
# import tensorflow_models.vision as tmv


AVAILABLE_OPS = [
    tf.image.adjust_brightness,  # https://www.tensorflow.org/api_docs/python/tf/image/random_brightness
    tf.image.flip_up_down,  # https://www.tensorflow.org/api_docs/python/tf/image/random_flip_up_down
    tf.image.flip_left_right,  # https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right
    # random_central_crop,  # https://www.tensorflow.org/api_docs/python/tf/image/central_crop
    tf.image.adjust_contrast,  # https://www.tensorflow.org/api_docs/python/tf/image/random_contrast
    tf.image.adjust_jpeg_quality,  # https://www.tensorflow.org/api_docs/python/tf/image/random_jpeg_quality
    tf.image.adjust_saturation,  # https://www.tensorflow.org/api_docs/python/tf/image/random_saturation
    tfa.image.cutout,  # https://www.tensorflow.org/addons/api_docs/python/tfa/image/random_cutout
    # dropout_grid,  # inspired by https://arxiv.org/abs/2001.04086
    # mixup,  # https://www.tensorflow.org/api_docs/python/tf/image/random_flip_up_down
    # cutmix,  # https://www.tensorflow.org/api_docs/python/tf/image/random_flip_up_down
    # random_rotation, # cf
]


class AugmentClassic(AugmentABC):
    def __init__(self, cfg_loc: str) -> None:
        super().__init__(cfg_loc)
        self.param_list = [
            getattr(self.config, s)
            for s in dir(self.config)
            if s.startswith("augment_params_")
        ]
        self.aug_list = self._choose_augs()

    def _choose_augs(
        self,
    ) -> List:
        perform_aug = np.array([True if prm else False for prm in self.param_list])
        return np.array(AVAILABLE_OPS)[perform_aug]

    def run_augmentations(self, image: tf.Tensor) -> tf.Tensor:
        # get number N of augmentations
        # create N rand uniform numbers
        # apply all fct from AVAILABLE_OPS if available
        rnd = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
        rnd_max_index = tf.math.argmax(rnd)
        if rnd[rnd_max_index] >= 0.5:
            image = self.aug_list[rnd_max_index](image, self.param_list[rnd_max_index])
            # apply_grid_mask(image, (*IMG_DIM,3))
        return image

    def run_AugMix_augmentations(self, image: tf.Tensor) -> tf.Tensor:
        """Following AugMix as introduced in https://arxiv.org/pdf/1912.02781.pdf.

        Args:
            image (tf.Tensor)

        Returns:
            tf.Tensor: Augmented image
        """
        (
            strength,
            width,
            depth,
            alpha,
        ) = self.config.augment_params_AugMix.values()
        dirichlet_sample = np.random.dirichlet([alpha] * width).astype(np.float32)
        m = np.float32(np.random.beta(alpha, alpha))
        mix = np.zeros_like(image).astype(np.float32)
        for i in range(width):
            image_aug = image.copy()
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op, strength)
                mix += dirichlet_sample[i] * image_aug

        mixed = (1 - m) * image + m * mix
        return mixed


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


# def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

#     h, w = image_height, image_width
#     hh = int(np.ceil(np.sqrt(h*h+w*w)))
#     hh = hh+1 if hh%2==1 else hh
#     d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
#     l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

#     st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
#     st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

#     y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
#     x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

#     for i in range(0, hh//d+1):
#         s1 = i * d + st_h
#         s2 = i * d + st_w
#         y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
#         x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

#     x_clip_mask = tf.logical_or(x_ranges <0 , x_ranges > hh-1)
#     y_clip_mask = tf.logical_or(y_ranges <0 , y_ranges > hh-1)
#     clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

#     x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
#     y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

#     hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
#     x_ranges = tf.repeat(x_ranges, hh)
#     y_ranges = tf.repeat(y_ranges, hh)

#     y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
#     x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

#     y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
#     y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

#     x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
#     x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

#     mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

#     mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
#     mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)

#     return mask

# def apply_grid_mask(image, image_shape):
#     mask = GridMask(image_shape[0],
#                     image_shape[1],
#                     AugParams['d1'],
#                     AugParams['d2'],
#                     AugParams['rotate'],
#                     AugParams['ratio'])

#     if image_shape[-1] == 3:
#         mask = tf.concat([mask, mask, mask], axis=-1)

#     return image * tf.cast(mask, tf.uint8)
