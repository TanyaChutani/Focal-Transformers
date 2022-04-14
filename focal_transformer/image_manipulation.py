import tensorflow as tf


class Image:
    def __init__(self, patch_width, patch_height, window_size):

        self.patch_width = patch_width
        self.patch_height = patch_height
        self.window_size = window_size

    def extract_patches_from_image(self, image):
        batch_size = tf.shape(image)[0]
        patches = tf.image.extract_patches(
            images=image,
            sizes=(1, self.patch_width, self.patch_height, 1),
            strides=(1, self.patch_width, self.patch_height, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

    def window_partition(self, image):
        batch_size, height, width, channels = image.shape
        image = tf.reshape(
            image,
            (
                batch_size,
                height // self.window_size,
                self.window_size,
                width // self.window_size,
                self.window_size,
                channels,
            ),
        )
        windows = tf.reshape(
            tf.transpose(image, (0, 1, 3, 2, 4, 5)),
            shape=(-1, self.window_size, self.window_size, channels),
        )
        return windows
