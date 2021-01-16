from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

import time


def classifier(data, labels):
    # train a Linear SVM on the data
    # start_time = time.time()
    # model = LinearSVC(random_state = 0)
    # model.fit(data, labels)

    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    model.fit(data, labels)
    # end_time = time.time()
    # print("applying classifier time: ", end_time - start_time)
    return model

def vote_result(results):
    count = [0] * 3
    for result in results:
        count[int(result)-1] += 1
    return count.index(max(count))+1

