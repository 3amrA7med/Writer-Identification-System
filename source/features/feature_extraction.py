import time
from source.features.LBP import LocalBinaryPatterns
from threading import Thread


def feature_extractor(images):
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    # start = time.time()
    desc = LocalBinaryPatterns(8, 3)
    """
    data_arr = [[]] * 3
    labels_arr = [[]] * 3
    feature_extraction_threads = [None] * 3

    for j in range(len(feature_extraction_threads)):
        feature_extraction_threads[j] = Thread(target=writer_feature_extraction, args=(images[j], desc,
                                                                                       labels_arr, data_arr, j))
        feature_extraction_threads[j].start()
    """
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
    # print(labels)
    """
    # loop over the training images
    for img in imgs_1:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("1")
        data.append(hist)
    for img in imgs_2:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("2")
        data.append(hist)
    for img in imgs_3:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("3")
        data.append(hist)
    """
    # end = time.time()
    # print("Writers feature extraction time:" + str(end - start))
    return data, labels, desc


def writer_feature_extraction(images, desc, labels, data, index):
    for img in images:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append(str(index))
        data.append(hist)


def test(model, imgs, desc):
    # loop over the testing images
    results = []
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