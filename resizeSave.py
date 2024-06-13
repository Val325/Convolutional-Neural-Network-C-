import os
from PIL import Image

train_cat = "dataset/training_set/training_set/cats"
train_dog = "dataset/training_set/training_set/dogs"
train_cat_data = os.listdir(train_cat) 
train_dog_data = os.listdir(train_dog) 
print("train_cat: ",len(train_cat_data))
print("train_dog: ",len(train_dog_data))

train_cat_processing = "datasetpreprocessing/train/cats"
train_dog_processing = "datasetpreprocessing/train/cats"

test_cat = "dataset/test_set/test_set/cats"
test_dog = "dataset/test_set/test_set/dogs"

test_cat_data = os.listdir(test_cat) 
test_dog_data = os.listdir(test_dog) 

print("test_cat: ", len(os.listdir(test_cat)))
print("test_dog: ", len(os.listdir(test_dog)))

test_cat_processing = "datasetpreprocessing/test/cats"
test_dog_processing = "datasetpreprocessing/test/dogs"

cropimagesize = (150, 150)
total_size = len(train_cat_data) + len(train_dog_data) + len(os.listdir(test_cat)) + len(os.listdir(test_dog)) 
print("total size:", total_size)
current_size_processed = 0

def Resize_Save(array_image, path, prefix, pathoutput):
    global current_size_processed 
    for i in range(len(array_image)):
        current_size_processed += 1
        size_proc =  (total_size / current_size_processed) * 100  
        print("process: ", size_proc)
        os.system('clear')
        image = Image.open(path + "/" + array_image[i]) 
        image = image.resize(cropimagesize)
        nameimage = pathoutput + "/" + str(prefix) + "." + str(i) + ".png" 
        image.save(nameimage)


Resize_Save(train_cat_data, train_cat, "cat", train_cat_processing)
Resize_Save(train_dog_data, train_dog, "dog", train_dog_processing)

Resize_Save(test_cat_data, test_cat, "cat", test_cat_processing)
Resize_Save(test_dog_data, test_dog, "dog", test_dog_processing)


