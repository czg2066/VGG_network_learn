from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks as kc
import keras.utils as ku
import params
import network


param = params.net_param()

training_generator = ku.image_dataset_from_directory(
    param.Train_pth,
    image_size=(param.Width, param.Height),
    batch_size=param.Batch_size)
validation_generator = ku.image_dataset_from_directory(
    param.Vali_pth,
    image_size=(param.Width, param.Height),
    batch_size=param.Batch_size)
test_generator = ku.image_dataset_from_directory(
    param.Test_pth,
    image_size=(param.Width, param.Height),
    batch_size=1,
    shuffle=False)

mymodel = network.my_model()
model = mymodel.model
mymodel.show()

callbacks = [
    # 当验证集的损失低于1e-5时，训练自动停止
    kc.EarlyStopping(monitor="loss",
                     patience=1e5,
                     min_delta=0,
                     verbose=1,
                     baseline=1e-4,
                     mode="min")
]
history = model.fit(training_generator,
            epochs=param.Epochs,
            batch_size=param.Batch_size,
            callbacks=callbacks,
            validation_data=validation_generator)

model.save(param.Model_save_pth+'four.h5')

score = model.evaluate(test_generator,
                       verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])