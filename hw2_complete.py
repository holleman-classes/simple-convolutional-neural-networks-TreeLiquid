import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, SeparableConv2D, Dropout, Add
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split


def build_model1():
    model1 = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        # First Layer
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),  # Second Layer
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),  # Third Layer
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),  # Pooling
        Flatten(),  # Flatten
        Dense(128, activation='relu'),  # Dense layers
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

    model1.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model1.summary()
    return model1


def build_model2():
    model2 = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        # First Layer
        BatchNormalization(),
        SeparableConv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),  # Second Layer
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),  # Third Layer
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),  # Pooling
        Flatten(),  # Flatten
        Dense(128, activation='relu'),  # Dense layers
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

    model2.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model2.summary()
    return model2


def build_model3():
    inputs = Input(shape=(32, 32, 3))
    # Layer 1
    res = Conv2D(32, (3, 3), strides=(2, 2), name='conv1', activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(res)
    conv1 = Dropout(0.4)(conv1)
    # Layer 2
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), name='conv2', activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.4)(conv2)
    # Layer 3
    conv3 = Conv2D(128, (3, 3), strides=(2, 2), name='conv3', activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.4)(conv3)

    # Skip Layer 1
    sk1 = Conv2D(128, (3, 3), strides=(2, 2), name='skip1', activation='relu', padding='same')(conv2)
    sk1 = Add()([sk1, conv3])

    # Layer 4
    conv4 = Conv2D(128, (3, 3), strides=(2, 2), name='conv4', activation='relu', padding='same')(sk1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.4)(conv4)
    # Layer 5
    conv5 = Conv2D(128, (3, 3), strides=(2, 2), name='conv5', activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.4)(conv5)

    # Skip Layer 2
    sk2 = Add()([sk1, conv5])

    # Layer 6
    conv6 = Conv2D(128, (3, 3), strides=(2, 2), name='conv6', activation='relu', padding='same')(sk2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.4)(conv6)
    # Layer 7
    conv7 = Conv2D(128, (3, 3), strides=(2, 2), name='conv7', activation='relu', padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.4)(conv7)

    # Skip Layer 3
    sk3 = Add()([sk1, conv7])

    # Pooling
    pool1 = MaxPooling2D((4, 4), strides=(4, 4))(sk3)
    flat1 = Flatten()(pool1)

    # Dense Layers
    dense = Dense(128, activation='relu')(flat1)
    dense = BatchNormalization()(dense)

    # output
    output = Dense(10, activation='softmax')(dense)

    model3 = Model(
        inputs=inputs,
        outputs=output
    )

    model3.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model3.summary()
    return model3


def build_model50k():
    model50k = Sequential([
        Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model50k.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model50k.summary()
    return model50k


# no training or dataset construction should happen above this line
if __name__ == '__main__':
    ########################################
    (train_images_all, train_labels_all), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Train test Split
    train_images, val_images, train_labels, val_labels = train_test_split(train_images_all, train_labels_all,
                                                                          test_size=0.1, random_state=1)

    train_images = train_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    ########################################
    # Build and train model 1
    model1 = build_model1()
    model1.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #history1 = model1.fit(
    #    train_images,
    #    train_labels,
    #    epochs=50,
    #    validation_data=(val_images, val_labels)
    #)
    #test_loss, test_acc = model1.evaluate(test_images, test_labels)
    #train_acc = history1.history['accuracy']
    #val_acc = history1.history['val_accuracy']

    #print("Test Acc:", test_acc)
    #print("Final Training Accuracy:", train_acc[-1])
    #print("Final Validation Accuracy:", val_acc[-1])
    #model1.save('model1.h5')
    #model1 = tf.keras.models.load_model('model1.h5')
    #model1.summary()

    #Load image
    #image1 = 'dog.jpg'
    #img = image.load_img(image1, target_size= (32,32))
    #im_arr = image.img_to_array(img)
    #im_arr = np.expand_dims(im_arr, axis=0)
    #im_arr = im_arr / 255.0

    #Predictions
    #pred = model1.predict(im_arr)

    #class_pred = np.argmax(pred)
    #classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #predicted_class = classes[class_pred]
    #print("Predicted Class:", predicted_class)


    # =======================================================#
    # Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()

    model2.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    #
    #history2 = model2.fit(
    #    train_images,
    #    train_labels,
    #    epochs=50,
    #    validation_data=(val_images, val_labels)
    #)
    #test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
    #train_acc2 = history2.history['accuracy']
    #val_acc2 = history2.history['val_accuracy']

    #print("Test Acc:", test_acc2)
    #print("Final Training Accuracy:", train_acc2[-1])
    #print("Final Validation Accuracy:", val_acc2[-1])
    #model2.save('model2.h5')
    model2 = tf.keras.models.load_model('model2.h5')
    model2.summary()

    # =============================================================#
    # Model 3
    model3 = build_model3()

    model3.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    #
    #history3 = model3.fit(
    #    train_images,
    #    train_labels,
    #    epochs=50,
    #    validation_data=(val_images, val_labels)
    #)
    #test_loss3, test_acc3 = model3.evaluate(test_images, test_labels)
    #train_acc3 = history3.history['accuracy']
    #val_acc3 = history3.history['val_accuracy']

    #print("Test Acc:", test_acc3)
    #print("Final Training Accuracy:", train_acc3[-1])
    #print("Final Validation Accuracy:", val_acc3[-1])
    #model3.save('model3.h5')
    model3 = tf.keras.models.load_model('model3.h5')
    model3.summary()

    # =================================#
    # Model4/50k
    model50k = build_model50k()

    model50k.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history50k = model50k.fit(
        train_images,
        train_labels,
        epochs=50,
        validation_data=(val_images, val_labels)
    )
    test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels)
    train_acc50k = history50k.history['accuracy']
    val_acc50k = history50k.history['val_accuracy']

    print("Test Acc:", test_acc50k)
    print("Final Training Accuracy:", train_acc50k[-1])
    print("Final Validation Accuracy:", val_acc50k[-1])
    model50k.save('best_model.h5')
