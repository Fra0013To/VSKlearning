import tensorflow as tf

optimizers_dict = {
    'adam_lr2e-3': tf.keras.optimizers.Adam(learning_rate=0.002),
    'adam_lr1e-3': tf.keras.optimizers.Adam(learning_rate=0.001),
    'adam_lr5e-4': tf.keras.optimizers.Adam(learning_rate=0.0005),
    'adam_lr1e-4': tf.keras.optimizers.Adam(learning_rate=0.0001)
}

