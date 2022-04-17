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
        if ((height % self.window_size) ! = 0) or ((width %self.window_size) !=0):
                pad_left = pad_top = 0
                pad_right = (self.window_size - width % self.window_size) % self.window_size
                pad_bottom = (self.window_size - height % self.window_size) % self.window_size
                x = tf.pad(image, (0, 0, pad_left, pad_right, pad_top, pad_bottom))

        new_batch_size, new_height, new_width, new_channels = image.shape

        image = tf.reshape(
            image,
            (
                new_batch_size,
                new_height // self.window_size,
                self.window_size,
                new_width // self.window_size,
                self.window_size,
                new_channels,
            ),
        )
        windows = tf.reshape(
            tf.transpose(image, (0, 1, 3, 2, 4, 5)),
            shape=(-1, self.window_size, self.window_size, new_channels),
        )
        return windows

    def window_reverse(self, windows, image_height, image_width):
            B = int(windows.shape[0] / (image_height * image_width / self.window_size / self.window_size))
            x = tf.reshape(B, image_height // self.window_size, image_width // self.window_size, self.window_size, self.window_size, -1)
            x = tf.reshape(x.permute(0, 1, 3, 2, 4, 5), B, image_height, image_width, -1)
            return x
