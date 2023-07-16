import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFile

data_dir = None
input_value = 1
model = None
selected_image = None
dataset_labels = ['Альбатреллус Овечий', 'Аурикулярия', 'Белый Гриб', 'Боровик', 'Вешенка',
                  'Волнушка', 'Гигрофор', 'Головач', 'Груздь', 'Зимний Гриб', 'Зонтик',
                  'Лиофиллум Ильмовый', 'Лисичка', 'Масленок', 'Мокруха', 'Моховик',
                  'Подберезовик', 'Подосиновик', 'Рыжик', 'Спарассис Курчавый']


def choose_data_dir():
    global data_dir
    data_dir = filedialog.askdirectory()


def train_model():
    global data_dir
    global input_value
    global dataset_labels
    global model

    if data_dir is None:
        messagebox.showinfo("Error", "Please select the dataset directory first.")
        return

    IMAGE_RESOL = (224, 224)
    DATA_DIR = str(data_dir)
    datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        subset="validation",
        shuffle=True,
        target_size=IMAGE_RESOL
    )
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        subset="training",
        shuffle=True,
        target_size=IMAGE_RESOL)

    dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair: pair[1])
    dataset_labels = np.array([key.title() for key, value in dataset_labels])
    print("Dataset labels:", dataset_labels)

    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
                       output_shape=[1280],
                       trainable=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.build([None, 224, 224, 3])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)
    val_steps_per_epoch = np.ceil(valid_generator.samples / valid_generator.batch_size)

    if input_entry.get():
        input_value = int(input_entry.get())
    else:
        input_value = 1

    model.fit(
        train_generator,
        epochs=input_value,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=val_steps_per_epoch)

    model.save("./saved_models")


def choose_image():
    global selected_image
    global image_label

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        selected_image = Image.open(file_path)
        resized_image = selected_image.resize((300, 300))
        photo = ImageTk.PhotoImage(resized_image)

        image_label.configure(image=None)

        image_label.image = photo
        image_label.configure(image=photo)


def load_model_from_disk():
    global model
    model = tf.keras.models.load_model(filedialog.askdirectory())


def recognize_image():
    global selected_image
    global model
    global dataset_labels

    if selected_image is None:
        messagebox.showinfo("Error", "Please choose an image first.")
        return

    if model is None:
        messagebox.showinfo("Error", "Please choose trained model first.")
        return

    resized_image = selected_image.resize((224, 224))
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

    top_k = tf.math.top_k(predictions[0], k=4)
    top_k_indices = top_k.indices.numpy()
    top_k_values = top_k.values.numpy()
    top_k_labels = [dataset_labels[i] for i in top_k_indices]

    result_text = ""
    for i in range(3):
        result_text += f"{top_k_labels[i]} ({round(top_k_values[i] * 100, 2)}%)\n"

    result_label.configure(text=result_text)


window = tk.Tk()
window.title("Image Recognition")
window.geometry("310x515")
window.configure(bg="white")

label_train = tk.Label(window, text="Neural network training", font=("Arial", 12, "bold"), bg="white")
label_train.grid(row=0, column=0, columnspan=4, sticky="ew", pady=5)

data_dir_button = tk.Button(window, text="Dataset", command=choose_data_dir)
data_dir_button.grid(row=1, column=0, sticky="ew", padx=(5, 0))

input_label = tk.Label(window, text="Epochs:", bg="white")
input_label.grid(row=1, column=1, sticky="e")

input_entry = tk.Entry(window, width=3)
input_entry.grid(row=1, column=2, sticky="w")

train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.grid(row=1, column=3, sticky="ew", pady=5, padx=(0, 5))

label_recognize = tk.Label(window, text="Recognition", font=("Arial", 12, "bold"), bg="white")
label_recognize.grid(row=2, column=0, columnspan=4, sticky="ew")

load_model_button = tk.Button(window, text="Load Model", command=load_model_from_disk)
load_model_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(5, 0))

choose_image_button = tk.Button(window, text="Choose Image", command=choose_image)
choose_image_button.grid(row=3, column=2, columnspan=2, sticky="ew", padx=(0, 5))

image_label = tk.Label(window, text="Place for the image", bg="white")
image_label.grid(row=4, column=0, columnspan=4, pady=5, padx=5, sticky="ew")

recognize_button = tk.Button(window, text="Recognize Image", command=recognize_image)
recognize_button.grid(row=5, column=0, columnspan=4, sticky="ew", padx=5)

result_label = tk.Label(window, bg="white")
result_label.grid(row=6, column=0, columnspan=4, sticky="ew")

window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=1)
window.grid_columnconfigure(3, weight=1)
window.grid_rowconfigure(4, minsize=300)

window.mainloop()
