from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline


def classifier(data, labels):
    """
    Train a Linear SVM on the data
    :param data: features
    :param labels: images writers
    :return: svm trained model
    """
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    model.fit(data, labels)
    return model


def vote_result(results):
    """
    Vote for the predicted writer
    :param results: results returned from model
    :return: the writer predicted
    """
    count = [0] * 3
    for result in results:
        count[int(result)-1] += 1
    return count.index(max(count))+1
