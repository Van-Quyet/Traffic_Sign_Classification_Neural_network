import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import sys

# Load your trained model
model = load_model('Traffic Sign.h5')

# Define your class names
class_names = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
    'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h',
    'Speed Limit 100 km/h', 'Speed Limit 120 km/h', 'No passing',
    'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road',
    'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Create the main window with a custom size (width x height)
root = tk.Tk()
root.title("Traffic Sign Classification")
root.geometry("500x500")  # Set the window size here

# Create a label for instructions
instruction_label = tk.Label(root, text="Upload an image for classification:")
instruction_label.pack(pady=10)

# Create a label to display the result with green text color
result_label = tk.Label(root, text="", font=('Helvetica', 14), fg="green")
result_label.pack(pady=10)

# Create a label to display the uploaded image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Function to preprocess and classify an image
def classify_image():
    global result_label
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = np.expand_dims(img, axis=-1)
        img = np.array(img)
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        probability_value = np.max(predictions)
        class_name = class_names[class_index]
        
        # Update the text and color of the "Class" and "Probability" labels
        result_label.config(text=f'Class: {class_name}\nProbability: {probability_value:.2%}', fg="green")
        
        # Display the uploaded image on the GUI
        image = Image.open(file_path)
        image.thumbnail((200, 200))
        image = ImageTk.PhotoImage(image)
        image_label.config(image=image)
        image_label.image = image

# Function to preprocess an image
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Function to exit the application
def exit_application():
    root.destroy()
    sys.exit()

# Create a button to upload an image
upload_button = tk.Button(root, text="UPLOAD IMAGE", command=classify_image, width=15, height=1,font=('Helvetica', 10),fg="blue",bg="gray")
upload_button.pack(pady=10)

# Create an "Exit" button with a custom size
exit_button = tk.Button(root, text="EXIT", command=exit_application, width=15, height=1,font=('Helvetica', 10),fg="red",bg="gray")
exit_button.pack(pady=10)

# Run the GUI main loop
root.mainloop()
