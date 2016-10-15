"""Train a sequence tagger.

Usage:
  lab1.py [-c <integer>] [-t <integer>] [-k <kernel>]

Options:
  -c <integer>    Number of clusters
  -t <integer>    Number of threads
  -k <string>     one of 'intersect' or 'rbf'
"""


from __future__ import print_function
from __future__ import division

import sys

from os import listdir, makedirs
from os.path import join, splitext, abspath, split, exists
from timeit import timeit

import numpy as np
np.seterr(all='raise')

import cv2
import pickle

from skimage.feature import daisy
from skimage.transform import rescale

from sklearn.svm import LinearSVC, SVC

from scipy.spatial import distance
from scipy.io import savemat, loadmat
from docopt import docopt
from datetime import datetime

def save_data(data, filename, force_overwrite=False):
    # if dir/subdir doesn't exist, create it
    dirname = split(filename)[0]
    if not exists(dirname):
        makedirs(dirname)
    savemat(filename, {'data': data}, appendmat=False)


def load_data(filename):
    return loadmat(filename, appendmat=False)['data'].squeeze()


def load_scene_categories(path, random_state=None):
    cname = sorted(listdir(path))  # human-readable names
    cid = []                       # class id wrt cname list
    fname = []                     # relative file paths
    for i, cls in enumerate(cname):
        for img in listdir(join(path, cls)):
            if splitext(img)[1] not in ('.jpeg', '.jpg', '.png'):
                continue
            fname.append(join(cls, img))
            cid.append(i)
    return {'cname': cname, 'cid': cid, 'fname': fname}


def n_per_class_split(dataset, n=100, random_state=None):
    # set RNG
    if random_state is None:
        random_state = np.random.RandomState()

    n_classes = len(dataset['cname'])
    cid = dataset['cid']
    fname = dataset['fname']

    train_set = []
    test_set = []
    for id_ in range(n_classes):
        idxs = [i for i, j in enumerate(cid) if j == id_]
        random_state.shuffle(idxs)

        # train samples
        for i in idxs[:n]:
            train_set.append((fname[i], cid[i]))

        # test samples
        for i in idxs[n:]:
            test_set.append((fname[i], cid[i]))

    random_state.shuffle(train_set)
    random_state.shuffle(test_set)
    return train_set, test_set


SCALES_3 = [1.0, 0.5, 0.25]
SCALES_5 = [1.0, 0.707, 0.5, 0.354, 0.25]
def extract_multiscale_dense_features(imfile, step=8, scales=SCALES_3):
    im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
    feat_all = []
    for sc in scales:
        dsize = (int(sc * im.shape[0]), int(sc * im.shape[1]))
        im_scaled = cv2.resize(im, dsize, interpolation=cv2.INTER_LINEAR)
        feat = daisy(im_scaled, step=step, normalization='l2')
        if feat.size == 0:
            break
        ndim = feat.shape[2]
        feat = np.atleast_2d(feat.reshape(-1, ndim))
        for f in feat:
            np.sqrt(f, f)
        feat_all.append(feat)
    return np.row_stack(feat_all).astype(np.float32)


def compute_features(base_path, im_list, output_path):
    # compute and store low level features for all images
    for fname in im_list:
        # image full path
        imfile = join(base_path, fname)

        # check if destination file already exists
        featfile = join(output_path, splitext(fname)[0] + '.feat')
        if exists(featfile):
            print('{} already exists'.format(featfile))
            continue

        feat = extract_multiscale_dense_features(imfile)
        save_data(feat, featfile)
        print('{}: {} features'.format(featfile, feat.shape[0]))


def sample_feature_set(base_path, im_list, output_path, n_samples,
                       random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    n_per_file = 100
    sample_file = join(output_path, 'sample{:d}.feat'.format(n_samples))
    if exists(sample_file):
        sample = load_data(sample_file)
    else:
        sample = []
        while len(sample) < n_samples:
            i = random_state.randint(0, len(im_list))
            featfile = join(base_path, splitext(im_list[i])[0] + '.feat')
            feat = load_data(featfile)
            idxs = random_state.choice(range(feat.shape[0]), 100)
            sample += [feat[i] for i in idxs]
            print('\r{}/{} samples'.format(len(sample), n_samples), end='')
            sys.stdout.flush()

        sample = np.row_stack(sample)
        save_data(sample, sample_file)
    print('\r{}: {} features'.format(sample_file, sample.shape[0]))
    return sample


def kmeans_fit(samples, n_clusters, maxiter=100, tol=1e-4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    n_samples = samples.shape[0]

    # chose random samples as initial estimates
    idxs = random_state.randint(0, n_samples, n_clusters)
    centroids = samples[idxs, :]

    J_old = np.inf
    for iter_ in range(maxiter):

        # SAMPLE-TO-CLUSTER ASSIGNMENT

        # cdist returns a matrix of size n_samples x n_clusters, where the i-th
        # row stores the (squared) distance from sample i to all centroids
        dist2 = distance.cdist(samples, centroids, metric='sqeuclidean')

        # argmin over columns of the distance matrix
        assignment = np.argmin(dist2, axis=1)

        # CENTROIDS UPDATE (+ EVAL DISTORTION)

        J_new = 0.
        for k in range(n_clusters):
            idxs = np.where(assignment == k)[0]
            if len(idxs) == 0:
                raise RuntimeError('k-means crash!')

            centroids[k, :] = np.mean(samples[idxs], axis=0).astype(np.float32)

            J_new += np.sum(dist2[idxs, assignment[idxs]])
        J_new /= float(n_samples)

        print('iteration {}, potential={:.3e}'.format(iter_, J_new))
        if (J_old - J_new) / J_new < tol:
            print('STOP')
            break
        J_old = J_new

    return centroids


def compute_bovw(vocabulary, features, norm=2):
    if vocabulary.shape[1] != features.shape[1]:
        raise RuntimeError('something is wrong with the data dimensionality')
    # dist2 = 2 * (1 - np.dot(features, vocabulary.transpose()))
    dist2 = distance.cdist(features, vocabulary, metric='sqeuclidean')
    assignments = np.argmin(dist2, axis=1)
    bovw, _ = np.histogram(assignments, range(vocabulary.shape[1]))
    bovw = np.sqrt(bovw)
    nrm = np.linalg.norm(bovw, ord=norm)
    return bovw / (nrm + 1e-7)

'''
updates a list on disk with the provided element, if the file is present
if the file is missing, starts from scratch.
returns the list
'''
def appendToList(filename, element):
    items = []
    if exists(filename):
        f = open(filename, 'rb')
        items = pickle.load(f)

    items.append(element)
    f = open(filename, 'wb')
    pickle.dump(items, f)
    f.close()
    return items


def call_bow(fname):
    # low-level features file
    featfile = join(output_path, splitext(fname)[0] + '.feat')
    bovwfile = join(output_path, splitext(fname)[0] + '.bovw')

    # check if destination file already exists
    if exists(bovwfile):
        print('{} already exists'.format(bovwfile))
        return

    #feat = pickle.load(open(featfile, 'rb'))
    features = load_data(featfile)
    bovw = compute_bovw(vocabulary, features, 2)
    save_data(bovw, bovwfile)
    print('{}'.format(bovwfile))


def search_top_c(param):
    c_candidates = np.arange(2, 4, .2)
    topC, topAcc = 1.6, 0.0

    if param == 'linear' or param == 'intersect':
        for candidate in c_candidates:
            splits = 5
            # map from a split number -> accuracy of that split
            split_acc = {}
            for split in range(splits):
                if param == 'linear':
                    svm = LinearSVC(C=candidate, verbose=1)
                else:
                    svm = SVC(C=candidate, verbose=1, kernel=int_kernel)

                chunk_size = int(len(X_train) / float(splits))
                start = split * chunk_size
                stop = (split + 1) * chunk_size if split is not (splits - 1) else len(X_train)
                x = np.concatenate((X_train[:start], X_train[stop:]))
                y = np.concatenate((y_train[:start], y_train[stop:]))
                svm.fit(x, y)
                y_pred = svm.predict(X_train[start:stop])
                split_acc[split] = sum(y_pred == y_train[start:stop]) / len(y_train[start:stop])

            # compute accuracy for current C
            curr_acc = np.mean(list(split_acc.values()))
            if curr_acc > topAcc:
                topAcc = curr_acc
                topC = candidate
        return topC, topAcc, 0

    if param == 'rbf':
        c_candidates = np.arange(3, 7, .2)
        topGamma = 0
        import itertools
        gamma_candidates = np.arange(2, 6, .2)
        for C, gamma in itertools.product(c_candidates, gamma_candidates):
            splits = 5
            # map from a split number -> accuracy of that split
            split_acc = {}
            for split in range(splits):
                svm = SVC(C=C, kernel='rbf', gamma=gamma, verbose=1)

                chunk_size = int(len(X_train) / float(splits))
                start = split * chunk_size
                stop = (split + 1) * chunk_size if split is not (splits - 1) else len(X_train)
                x = np.concatenate((X_train[:start], X_train[stop:]))
                y = np.concatenate((y_train[:start], y_train[stop:]))
                svm.fit(x, y)
                y_pred = svm.predict(X_train[start:stop])
                split_acc[split] = sum(y_pred == y_train[start:stop]) / len(y_train[start:stop])

            # compute accuracy for current C
            curr_acc = np.mean(list(split_acc.values()))
            if curr_acc > topAcc:
                topAcc = curr_acc
                topC = C
                topGamma = gamma
        return topC, topAcc, topGamma

if __name__ == "__main__":
    random_state = np.random.RandomState(12345)
    opts = docopt(__doc__)
    print(opts['-k'])

    # ----------------
    # DATA PREPARATION
    # ----------------

    # paths
    dataset_path = abspath('scene_categories')
    output_path = 'cache'

    # load dataset, a dictionary with keys {cid, cname, fname : [path_to_file]}
    dataset = load_scene_categories(dataset_path)
    n_classes = len(dataset['cname'])
    n_images = len(dataset['fname'])
    print('{} images of {} categories'.format(n_images, n_classes))

    # train-test split
    # train_set and test_set are lists of tuples [(path_to_file, category_number)]
    train_set, test_set = n_per_class_split(dataset, n=100)
    n_train = len(train_set)
    n_test = len(test_set)
    print('{} training samples / {} testing samples'.format(n_train, n_test))

    # compute and store low level features for all images
    compute_features(dataset_path, dataset['fname'], output_path)

    # --------------------------------
    # UNSUPERVISED DICTIONARY LEARNING
    # --------------------------------

    n_samples = int(1e5)
    n_clusters = int(opts.get('-c', 100)) if opts['-c'] else 100
    vocabulary_file = join(output_path, 'vocabulary{:d}.dat'.format(n_clusters))
    if exists(vocabulary_file):
        #vocabulary = pickle.load(open(vocabulary_file, 'rb'))
        vocabulary = load_data(vocabulary_file)
    else:
        train_files = [fname for (fname, cid) in train_set]
        sample = sample_feature_set(output_path, train_files, output_path,
                                    n_samples, random_state=random_state)
        vocabulary = kmeans_fit(sample, n_clusters=n_clusters,
                                random_state=random_state)
        # L2 normalize each row in vocabulary
        # for word in vocabulary:
        #    word/=(np.linalg.norm(word, ord=2) + 1e-7)
        save_data(vocabulary, vocabulary_file)

    print('{}: {} clusters'.format(vocabulary_file, vocabulary.shape[0]))

    # --------------------
    # COMPUTE BoVW VECTORS
    # --------------------
    start = datetime.now()
    if opts['-t']:
        n_threads = int(opts.get('-t'))
        import multiprocessing
        pool = multiprocessing.Pool(4)
        pool.map(call_bow, dataset['fname'], 4400 // n_threads)
    else:
        # serial version
        for fname in dataset['fname']:
            call_bow(fname)

    # store times
    appendToList('bowt.pk', (datetime.now() - start).total_seconds())


    # -----------------
    # TRAIN CLASSIFIERS
    # -----------------

    # setup training data
    X_train, y_train = [], []
    for fname, cid in train_set:
        bovwfile = join(output_path, splitext(fname)[0] + '.bovw')
        #X_train.append(pickle.load(open(bovwfile, 'rb')))
        X_train.append(load_data(bovwfile))
        y_train.append(cid)
    X_train = np.array(X_train)
    y_train = np.array(y_train)


    def metric(X, Y):
        return np.sum(np.minimum(X, Y))

    def int_kernel(X,Y):
        return distance.cdist(X, Y, metric=metric)

    # ----------------------
    # Find top parameter C
    # ----------------------

    topC, topAcc, topGamma = search_top_c(opts['-k'])

    print('\ntop accuracy = {:.3f}'.format(topAcc))
    print('with C = {:.3f}'.format(topC))

    # setup testing data
    X_test, y_test = [], []
    for fname, cid in test_set:
        bovwfile = join(output_path, splitext(fname)[0] + '.bovw')
        #X_test.append(pickle.load(open(bovwfile, 'rb')))
        X_test.append(load_data(bovwfile))
        y_test.append(cid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if opts['-k'] == 'intersect':
        svm = SVC(C=topC, verbose=1, kernel=int_kernel)
    elif opts['-k'] == 'rbf':
        svm = SVC(C=topC, verbose=1, gamma=topGamma)
    else:
        svm = LinearSVC(C=topC, verbose=0)

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    tp = np.sum(y_test == y_pred)
    acc = float(tp) / len(y_test)
    print('accuracy = {:.3f}, topC = {:.3f}, topGamma = {:.3f}'.format(acc, topC, topGamma))
    appendToList('cs.pk', topC)
    appendToList('acc.pk', acc)



