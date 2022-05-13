import tensorflow as tf

class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, focal_window, focal_level, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout()
        self.softmax_layer = tf.keras.layers.Softmax()

    def build(self, input_shape):

        coords_h = tf.range(self.window_size[0]) - self.window_size[0] // 2
        coords_w = tf.range(self.window_size[1]) - self.window_size[1] // 2        
        coords_window = tf.stack(tf.meshgrid([coords_h, coords_w]), dim=-1)
        coords_flatten = coords_window.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.transpose([1, 2, 0]) 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_idx = relative_coords.sum(-1)  
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_idx), trainable=False
        )        
        
