# NN-MushroomRecognition-Py-TensorFlow-SSTU

Implementation of a neural network for solving the problem of mushroom recognition based on the TensorFlow library and the pre-trained efficientnet/b0 model. The Tkinter framework is used to draw the interface.

![image](https://github.com/ruddanil/NN-MushroomRecognition-Py-TensorFlow-SSTU/assets/25799951/c698a75d-d6ef-49fa-a3fc-b58ea7996b8d)

## Using the program to solve other recognition tasks

### Option 1. You want to train/retrain the model from scratch on your dataset.

The program implements the neural network training functionality on any image dataset. The only condition is that the data in the dataset must be divided into folders containing images of a certain category.

To train on your data, you need to do the following:
1. In the source code of the program, replace the category headers with those contained in your dataset.
```
dataset_labels = ['Альбатреллус Овечий', 'Аурикулярия', 'Белый Гриб', 'Боровик', 'Вешенка',
                  'Волнушка', 'Гигрофор', 'Головач', 'Груздь', 'Зимний Гриб', 'Зонтик',
                  'Лиофиллум Ильмовый', 'Лисичка', 'Масленок', 'Мокруха', 'Моховик',
                  'Подберезовик', 'Подосиновик', 'Рыжик', 'Спарассис Курчавый']
```
![image](https://github.com/ruddanil/NN-MushroomRecognition-Py-TensorFlow-SSTU/assets/25799951/e6904f11-1611-46da-9868-4b9dbe9a975b)

3. Clear or copy the contents of the "saved_models" folder to another directory.
4. Click on the "Dataset" button and select the directory where the dataset is located.
5. Specify the number of training epochs.
6. Press the "Train" button. The trained model will be saved in the saved_models folder.
7. You can start the recognition. Click the "Choose Image" button, select your image and click the "Recognize Image" button.

As a result, the top 3 most likely recognized categories are displayed at the bottom of the screen, indicating confidence in %.

Optional: You can replace the pre-trained TensorFlow model by specifying a link to it in the source code.
```
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
                       output_shape=[1280],
                       trainable=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
```
### Note: This program is an educational project. It is convenient to use it to demonstrate the work of a pre-trained model. To solve more serious problems, it is more convenient and correct to use tools like "Jupiter Notebook", where the learning progress is displayed and additional settings are possible.

### Option 2. You have a pre-trained model.

1. Replace dataset_labels in the source code (by analogy with option 1.1).
2. Click the "Load Model" button and specify the directory with your pre-trained model.
3. Proceed with recognition.


