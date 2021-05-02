from source.features.LBP import LocalBinaryPatterns
from threading import Thread


def feature_extractor(images, accuracy):
    """
    This function reads returns histograms accompanied with their labels, also returns descriptor used for feature extraction.
    """

    desc = LocalBinaryPatterns(accuracy)
    data_arr_1 = []
    labels_arr_1 = []
    data_arr_2 = []
    labels_arr_2 = []
    data_arr_3 = []
    labels_arr_3 = []
    feature_extraction_threads = [None] * 3

    feature_extraction_threads[0] = Thread(target=writer_feature_extraction, args=(images[0], desc,
                                                                                   labels_arr_1, data_arr_1, 1))
    feature_extraction_threads[0].start()
    feature_extraction_threads[1] = Thread(target=writer_feature_extraction, args=(images[1], desc,
                                                                                   labels_arr_2, data_arr_2, 2))
    feature_extraction_threads[1].start()
    feature_extraction_threads[2] = Thread(target=writer_feature_extraction, args=(images[2], desc,
                                                                                   labels_arr_3, data_arr_3, 3))
    feature_extraction_threads[2].start()
    for j in range(len(feature_extraction_threads)):
        feature_extraction_threads[j].join()

    data = data_arr_1 + data_arr_2 + data_arr_3
    labels = labels_arr_1 + labels_arr_2 + labels_arr_3

    return data, labels, desc


def writer_feature_extraction(images, desc, labels, data, index):
    """
    This function loops through images within the thread and modifies their labels and data.
    """
    for img in images:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append(str(index))
        data.append(hist)


def test(model, imgs, desc):
    """
    This function runs classification on test image in multiple threads and returns their results
    """
    
    # loop over the testing images
    testing_threads = [None] * len(imgs)
    results = [None] * len(imgs)
    for j in range(len(testing_threads)):
        testing_threads[j] = Thread(target=test_image, args=(model, imgs, desc, results, j))
        testing_threads[j].start()
    for j in range(len(testing_threads)):
        testing_threads[j].join()
    return results


def test_image(model, imgs, desc, results, j):
    hist = desc.describe(imgs[j])
    prediction = model.predict(hist.reshape(1, -1))
    results[j] = prediction[0]
