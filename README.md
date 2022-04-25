# Jupyter Vertex AI Magic

A Jupyter Notebook %%magic for training ML Models with Vertex AI training

Installation
------------

    $ pip install git+https://github.com/mblanc/vertex-magic
    
    

Usage
------

Load extension inside a Jupyter notebook:

```
%load_ext vertex-magic
```

Add code with Cell magic:

```
%%vertex

# code to run

```


Examples
--------

```
%%vertex --region europe-west4

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.save("<Google Cloud Storage Path>")
```

