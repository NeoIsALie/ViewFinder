from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


def generate_data(path):
    train = path + '/Train'
    img_width, img_height, channels = 150, 150, 3  # you can try different sizes
    batch_size = 15
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=90,
                                       vertical_flip=True,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(train,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   subset='training')

    val_data = val_datagen.flow_from_directory(train,
                                               target_size=(img_width, img_height),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False,
                                               subset='validation')
    return train_data, val_data


def train_model(model, path: str):
    batch_size = 15
    nb_train_samples = 179
    train_data, val_data = generate_data(path)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
    model.fit(
        train_data,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=20,
        validation_data=val_data,
        callbacks=[es],
    )
    model.save(path + '/my_model.h5')
    model.save(path + '/my_model.tf')
