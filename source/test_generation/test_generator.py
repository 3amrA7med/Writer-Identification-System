import os
from pathlib import Path
import shutil
import random

           
def generate_tests(M = 100):            # M is the number of randomly generated test cases
    f_dataset = open('./forms.txt', 'r')
    Lines = f_dataset.readlines()
    writers = []
    for i in range(672):
        writers.append(list())

    for line in Lines:
        if line[0] != '#':
            writers[int(line.split()[1])].append(line.split()[0])

    f_dataset.close()
    tests_writers = []                              # Array to keep the writer of the test image
    path = "./test/"
    number_of_writers_per_testcase = 3
    number_of_images_per_writer = 2
    # If folder does not exist create it
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    # If folder already exists delete it to generate new test cases
    else:
        shutil.rmtree(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    # Generate M test cases from the dataset
    for i in range(M):
        path = "./test/"
        if i < 9:
            path = path + str(0) + str(i+1)
        else:
            path = path + str(i+1)

        if i + 1 % 50:
            print(str(i + 1) + " test generated")
        # Create folder for each test case
        Path(path).mkdir(parents=True, exist_ok=True)

        # Pick 3 random writers for each test case
        random_writers = []                                                 # Picked writers for a test case
        while len(random_writers) < number_of_writers_per_testcase:
            W = random.randint(0,len(writers) - 1)                          # Pick random writer
            if not W in random_writers:                                     # Check that writer is not already picked
                if len(writers[W]) > number_of_images_per_writer:           # Check that writer has at least 3 images
                    random_writers.append(W)
        
        random_images = []                                                  # All the picked images in the testset
        for j in range(number_of_writers_per_testcase):
            writer_path = path + "/" + str(j + 1)
            Path(writer_path).mkdir(parents=True, exist_ok=True)            # Create folder for each writer
            
            # Pick 2 random images for each writer
            random_images_per_writer = []                                   # Picked images per writer
            while len(random_images_per_writer) < 2:
                I = random.randint(0,len(writers[random_writers[j]]) - 1)           # Pick a random image by the writer
                # Check the image is not already picked
                if not writers[random_writers[j]][I] in random_images_per_writer:
                    random_images_per_writer.append(writers[random_writers[j]][I])
                    random_images.append(writers[random_writers[j]][I])
            
            # Copy the picked images from the dataset folder to testcases
            shutil.copy2("./dataset/" + random_images_per_writer[0] + ".png", writer_path + "/1.png")
            shutil.copy2("./dataset/" + random_images_per_writer[1] + ".png", writer_path + "/2.png")

        # Pick test image from the dataset
        T = random.randint(0, len(random_writers) - 1)                          # Pick the writer of the test image
        while True:
            T_img = random.randint(0, len(writers[random_writers[T]]) - 1)      # Pick the test image
            # Check the test image is not in training images
            if not writers[random_writers[T]][T_img] in random_images:
                shutil.copy2("./dataset/" + writers[random_writers[T]][T_img] + ".png", path + "/test.png")
                tests_writers.append(T+1)
                break
    print(tests_writers)
    f = open("test_results.txt", "w")
    for test in tests_writers:
        f.write(str(test) + '\n')
    f.close()
