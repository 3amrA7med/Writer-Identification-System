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
def writer_identification(generate, number_of_test_cases):
    """
    Writer identification function
    :param generate: boolean to generate test cases
    :param number_of_test_cases: number of test case to generate
    """
    # Generate tests
    if generate:
        generate_tests(number_of_test_cases)
        return

    f_time = open('./output/time.txt', 'w')
    f_results = open('./output/results.txt', 'w')
    # open tests results
    f = open("test_results.txt", "r")
    test_results = f.readlines()

    for i in range(0, len(test_results)):
        test_results[i] = int(test_results[i])

    # program main loop
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
    for i in range(number_of_test_cases):
        path = "./test/"
        if i < 9:
            path = path + str(0) + str(i+1) + "/"
        else:
            path = path + str(i+1) + "/"
        preprocessing_threads = [None] * 7
        sentences = [None] * 7
        images = [None] * 3
        start_time = time.time()
        start = time.time()
        # Open thread to pre-process images
        for j in range(len(preprocessing_threads)):
            preprocessing_threads[j] = Thread(target=read_and_preprocess_image, args=(path+file_names[j], sentences, j))
            preprocessing_threads[j].start()

        for j in range(len(preprocessing_threads)):
            preprocessing_threads[j].join()

        end = time.time()
        print("Preprocessing time:" + str(end - start))

        images[0] = sentences[0] + sentences[1]
        images[1] = sentences[2] + sentences[3]
        images[2] = sentences[4] + sentences[5]

        # program logic here
        # Extract features from the three writers
        data, labels, desc = feature_extractor(images)

        # Train a SVM Classifier
        model = classifier(data, labels)

        # Classify the test
        results = test(model, sentences[6], desc)
        end_time = time.time()
        print("Test#" + str(i+1) + ", time: " + str(end_time - start_time))
        f_time.write(str(end_time - start_time) + "\n")
        avg_time.append(end_time - start_time)
        # Vote on classification
        winner = vote_result(results)
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


def read_and_preprocess_image(path, sentences, index):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sentences[index] = preprocessing(image)


if __name__ == '__main__':
    writer_identification()
