#!/usr/bin/python

import sys
import os
import time
from pathlib import Path
import cv2
from source.preprocessing.preprocessing import preprocessing
from source.features.feature_extraction import *
from source.classifier.classifier import classifier, vote_result
from source.test_generation.test_generator import generate_tests
import click


@click.command()
@click.option('-g', '--generate', type=bool,
              help='Generate test cases.', required=True, default=False)
@click.option('-n', '--number_of_test_cases',  type=click.types.INT,
              help='Number of test cases to generate.', default=100)
def writer_identification(generate, number_of_test_cases):
    """
    Writer identification function
    :param generate: boolean to generate test cases
    :param number_of_test_cases: number of test case to generate
    """

    f_time = open('./output/time.txt', 'w')
    f_results = open('./output/results.txt', 'w')
    # Generate tests
    if generate:
        generate_tests(number_of_test_cases)
    # open tests results
    f = open("test_results.txt", "r")
    test_results = f.readlines()

    for i in range(0, len(test_results)):
        test_results[i] = int(test_results[i])

    # program main loop
    total = 0
    accurate = 0
    for i in range(20):
        path = "./test/"

        if i < 9:
            path = path + str(0) + str(i+1) + "/"
        else:
            path = path + str(i+1) + "/"

        w11 = cv2.imread(path + str(1) + "/1.PNG")
        sentences11 = preprocessing(w11)
        w12 = cv2.imread(path + str(1) + "/2.png")
        sentences12 = preprocessing(w12)
        w21 = cv2.imread(path + str(2) + "/1.png")
        sentences21 = preprocessing(w21)
        w22 = cv2.imread(path + str(2) + "/2.png")
        sentences22 = preprocessing(w22)
        w31 = cv2.imread(path + str(3) + "/1.png")
        sentences31 = preprocessing(w31)
        w32 = cv2.imread(path + str(3) + "/2.png")
        sentences32 = preprocessing(w32)

        test_im = cv2.imread(path + "test.png")
        sentences_test = preprocessing(test_im)

        imgs_1 = sentences11 + sentences12
        imgs_2 = sentences21 + sentences22
        imgs_3 = sentences31 + sentences32

        start_time = time.time()
        # program logic here
        # Extract features from the three writers
        data, labels, desc = feature_extractor(imgs_1,imgs_2,imgs_3)

        # Train a SVM Classifier
        model = classifier(data, labels)

        # Classify the test
        results = test(model, sentences_test, desc)

        # Vote on classification
        winner = vote_result(results)
        print("Winner is",winner,"test result is",test_results[i], "for test number",i+1)
        total += 1
        if winner == test_results[i]:
            accurate += 1
        end_time = time.time()

        f_time.write(str(end_time - start_time) + "\n")

    if total > 0:
        print("Total is",total,"of which",accurate,"are accurate")
        print("Accuracy is", float(accurate)/total*100,"%")

    f_time.close()
    f_results.close()


if __name__ == '__main__':
    writer_identification()
