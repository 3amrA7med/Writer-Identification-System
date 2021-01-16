import time
from source.features.LBP import LocalBinaryPatterns
from threading import Thread


def feature_extractor(images):
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    start = time.time()
    desc = LocalBinaryPatterns(8, 3)
    data_arr = [[]] * 3
    labels_arr = [[]] * 3
    feature_extraction_threads = [None] * 3

    for j in range(len(feature_extraction_threads)):
        feature_extraction_threads[j] = Thread(target=writer_feature_extraction, args=(images[j], desc,
                                                                                       labels_arr, data_arr, j))
        feature_extraction_threads[j].start()

    for j in range(len(feature_extraction_threads)):
        feature_extraction_threads[j].join()

    data = data_arr[0] + data_arr[1] + data_arr[2]
    labels = labels_arr[0] + labels_arr[1] + labels_arr[2]
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
    end = time.time()
    print("Writers feature extraction time:" + str(end - start))
    return data, labels, desc


def writer_feature_extraction(images, desc, labels, data, index):
    for img in images:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels[index].append(str(index + 1))
        data[index].append(hist)


def test(model, imgs, desc):
    # loop over the testing images
    start = time.time()
    results = []
    for image in imgs:
        hist = desc.describe(image)
        prediction = model.predict(hist.reshape(1, -1))
        # # display the image and the prediction
        # cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
        # plt.imshow(image,cmap='Greys_r')
        # plt.title('test')
        # plt.show()
        results.append(prediction[0])
    print("Testing time:" + str(time.time() - start))
    return results
