import time
import cv2
from source.preprocessing.preprocessing import preprocessing
from source.features.feature_extraction import *
from source.classifier.classifier import classifier, vote_result
from source.test_generation.test_generator import generate_tests
import click
from threading import Thread


@click.command()
@click.option('-g', '--generate', type=bool,
              help='Generate test cases.', required=True, default=False)
@click.option('-n', '--number_of_test_cases', required=True,  type=click.types.INT,
              help='Number of test cases to generate.', default=100)
@click.option('-a', '--accuracy', required=False,  type=click.types.INT, default=1,
              help='Accuracy vs Performance. Values: [ 1, 2, 3]. 3 being highest accuracy, 1 being highest performance.')
def writer_identification(generate, number_of_test_cases, accuracy):
    """
    Writer identification function
    :param generate: boolean to generate test cases
    :param number_of_test_cases: number of test case to generate
    :param accuracy: determine accuracy level to be used by writer identification system classifier.
    """

    # Generate test cases
    if generate:
        generate_tests(number_of_test_cases)
        return
    # Open files used to save the output and read test results
    f_time = open('./output/time.txt', 'w')
    f_results = open('./output/results.txt', 'w')
    f = open("test_results.txt", "r")
    test_results = f.readlines()

    for i in range(0, len(test_results)):
        test_results[i] = int(test_results[i])

    total = 0
    accurate = 0
    avg_time = []
    file_names = ["1/1.PNG",
                  "1/2.PNG",
                  "2/1.PNG",
                  "2/2.PNG",
                  "3/1.PNG",
                  "3/2.PNG",
                  "test.png"]
    # program main loop
    for i in range(number_of_test_cases):
        path = "./test/"
        images = [None] * 7
        files_loading_threads = [None] * 7
        
        if i < 9:
            path = path + str(0) + str(i+1) + "/"
        else:
            path = path + str(i+1) + "/"
        # Ope threads to read test images
        for j in range(len(files_loading_threads)):
            files_loading_threads[j] = Thread(target=read_image, args=(path+file_names[j], images, j))
            files_loading_threads[j].start()
        for j in range(len(files_loading_threads)):
            files_loading_threads[j].join()

        preprocessing_threads = [None] * 7
        sentences = [None] * 7
        total_sentences = [None] * 3
        start_time = time.time()
        # Open threads to pre-process images
        for j in range(len(preprocessing_threads)):
            preprocessing_threads[j] = Thread(target=preprocess_image, args=(images, sentences, j))
            preprocessing_threads[j].start()
        for j in range(len(preprocessing_threads)):
            preprocessing_threads[j].join()

        total_sentences[0] = sentences[0] + sentences[1]
        total_sentences[1] = sentences[2] + sentences[3]
        total_sentences[2] = sentences[4] + sentences[5]

        # Extract features from the three writers
        data, labels, desc = feature_extractor(total_sentences, accuracy)

        # Train a SVM Classifier
        model = classifier(data, labels)

        # Classify the test
        results = test(model, sentences[6], desc)

        # Calculate time for this test case
        end_time = time.time()
        print("Test#" + str(i+1) + ", time: " + str(end_time - start_time))
        f_time.write(str(end_time - start_time) + "\n")
        avg_time.append(end_time - start_time)

        # Vote on classification
        winner = vote_result(results)
        f_results.write(str(winner) + "\n")
        if winner == test_results[i]:
            print("Test#" + str(i+1) + " succeeded")
            accurate += 1
        else:
            print("Test#" + str(i+1) + " failed")
        total += 1

    if total > 0:
        print("Total is", total, "of which", accurate, "are accurate")
        print("Accuracy is", float(accurate)/total*100, "%")
        print("Average Time:" + str(float(sum(avg_time))/len(avg_time)))
    f_time.close()
    f_results.close()


def read_image(path, images, index):
    """
    Read grayscale image
    :param path: path to the image
    :param images: array holds test case images
    :param index: index of the image to be read
    """
    images[index] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[100:3100, 150:-50]


def preprocess_image(images, sentences, index):
    """
    Pre-process images
    :param images: array holding test case images
    :param sentences: array of arrays holding sentences for each image
    :param index: index of the image to pre-process it.
    """
    sentences[index] = preprocessing(images[index])


if __name__ == '__main__':
    writer_identification()
