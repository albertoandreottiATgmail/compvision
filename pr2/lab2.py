# -*- coding: utf-8 -*-
'''AAVC, FaMAF-UNC, 11-OCT-2016

==========================================
Lab 2: Búsqueda y recuperación de imágenes
==========================================

0) Familiarizarse con el código, dataset y métricas:

Nister, D., & Stewenius, H. (2006). Scalable recognition with a vocabulary
tree. In: CVPR. link: http://vis.uky.edu/~stewe/ukbench/

1) Implementar verificación geométrica (ver función geometric_consistency). Si
   se implementa desde cero (estimación afín + RANSAC) cuenta como punto *.

2) PCA sobre descriptores: entrenar modelo a partir del conjunto de 100K
   descriptores random. Utilizarlo para proyectar los descriptores locales.
   Evaluar impacto de la dimensionalidad (16, 32, 64) con y sin re-normalizar L2
   los descriptores después de la proyección.

3) Evaluar influencia del tamaño de la lista corta y las diferentes formas de
   scoring.

4*) Modificar el esquema de scoring de forma tal de rankear las imágenes usando
   el kernel de intersección sobre los vectores BoVW equivalentes. Como se puede
   extender a kernels aditivos?. NOTA: al emplear el kernel de intersección los
   vectores BoVW deben estar normalizados L1.

NOTA: a los fines de evaluar performance en retrieval utilizaremos como imágenes
de query la primera imagen de los 100 primeros objetos disponibles en el
dataset. Para hacer pruebas rápidas y debug, se puede setear N_QUERY a un valor
mas chico.

'''
from __future__ import print_function
from __future__ import division

import sys

from os import listdir, makedirs
from os.path import join, splitext, abspath, split, exists

import numpy as np
from skimage.transform import AffineTransform

np.seterr(all='raise')

import cv2

from utils import load_data, save_data, load_index, save_index, get_random_sample, compute_features

from sklearn.cluster import KMeans

from scipy.spatial import distance

from skimage.measure import ransac

import base64


N_QUERY = 100


def read_image_list(imlist_file):
    return [f.rstrip() for f in open(imlist_file)]


def geometric_consistency(feat1, feat2):
    kp1, desc1 = feat1['kp'], feat1['desc']
    kp2, desc2 = feat2['kp'], feat2['desc']

    number_of_inliers = 0

    # 1) matching de features
    matcher = cv2.BFMatcher()
    matches = matcher.match(desc1, desc2)
    src = []
    dst = []
    for match in matches:
        src.append(kp1[match.queryIdx]), dst.append(kp1[match.queryIdx])

    # src and dst are list of lists that contain 2 elements
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
    '''
    2) Estimar una tranformación afín empleando RANSAC
       a) armar una función que estime una transformación afin a partir de
          correspondencias entre 3 pares de puntos (solución de mínimos
          cuadrados en forma cerrada)
       b) implementar RANSAC usando esa función como estimador base
    3) contar y retornar número de inliers
    '''

    return number_of_inliers


def pca_fit(samples):
    _, ndim = samples.shape
    P = np.empty((ndim, ndim))
    mu = np.empty(ndim)
    '''
    1) computar la media (mu) de las muestras de entrenamiento
    2) computar matriz de covarianza
    3) computar autovectores y autovalores de la matriz de covarianza
    4) ordenar autovectores por valores decrecientes de los autovectores
    '''
    return P, mu


def pca_project(x, P, mu, dim):
    return np.dot((x - mu).reshape(1, -1), P[:, :dim]).squeeze()


if __name__ == "__main__":
    random_state = np.random.RandomState(12345)

    # ----------------
    # BUILD VOCABULARY
    # ----------------

    unsup_base_path = 'full/'
    unsup_image_list_file = 'image_list.txt'

    output_path = 'cache'

    # compute random samples
    n_samples = int(1e5)
    unsup_samples_file = join(output_path, 'samples_{:d}.dat'.format(n_samples))
    if not exists(unsup_samples_file):
        unsup_samples = get_random_sample(read_image_list(unsup_image_list_file),
                                          unsup_base_path, n_samples=n_samples,
                                          random_state=random_state)
        save_data(unsup_samples, unsup_samples_file)
        print('{} saved'.format(unsup_samples_file))

    # compute vocabulary
    n_clusters = 1000
    vocabulary_file = join(output_path, 'vocabulary_{:d}.dat'.format(n_clusters))
    if not exists(vocabulary_file):
        samples = load_data(unsup_samples_file)
        kmeans = KMeans(n_clusters=n_clusters, verbose=1, n_jobs=-2)
        kmeans.fit(samples)
        save_data(kmeans.cluster_centers_, vocabulary_file)
        print('{} saved'.format(vocabulary_file))

    # --------------
    # DBASE INDEXING
    # --------------

    base_path = unsup_base_path
    image_list = read_image_list(unsup_image_list_file)

    # pre-compute local features
    for fname in image_list:
        imfile = join(base_path, fname)
        featfile = join(output_path, splitext(fname)[0] + '.feat')
        if exists(featfile):
            continue
        fdict = compute_features(imfile)
        save_data(fdict, featfile)
        print('{}: {} features'.format(featfile, len(fdict['desc'])))

    # compute inverted index
    index_file = join(output_path, 'index_{:d}.dat'.format(n_clusters))
    if not exists(index_file):
        vocabulary = load_data(vocabulary_file)
        n_clusters, n_dim = vocabulary.shape

        index = {
            'n': 0,                                               # n documents
            'df': np.zeros(n_clusters, dtype=int),                # doc. frec.
            'dbase': dict([(k, []) for k in range(n_clusters)]),  # inv. file
            'id2i': {},                                           # id->index
            'norm': {}                                            # L2-norms
        }

        n_images = len(image_list)

        for i, fname in enumerate(image_list):
            imfile = join(base_path, fname)
            imID = base64.encodestring(fname) # as int? / simlink to filepath?
            if imID in index['id2i']:
                continue
            index['id2i'][imID] = i

            ffile = join(output_path, splitext(fname)[0] + '.feat')
            fdict = load_data(ffile)
            kp, desc = fdict['kp'], fdict['desc']

            nd = len(desc)
            if nd == 0:
                continue

            dist2 = distance.cdist(desc, vocabulary, metric='sqeuclidean')
            assignments = np.argmin(dist2, axis=1)
            idx, count = np.unique(assignments, return_counts=True)
            for j, c in zip(idx, count):
                index['dbase'][j].append((imID, c)) #image 'imID' had feature 'j', 'c' times
            index['n'] += 1
            index['df'][idx] += 1
            #index['norm'][imID] = np.float32(nd)
            index['norm'][imID] = np.linalg.norm(count)

            print('\rindexing {}/{}'.format(i+1, n_images), end='')
            sys.stdout.flush()
        print('')

        save_index(index, index_file)
        print('{} saved'.format(index_file))

    # ---------
    # RETRIEVAL
    # ---------

    vocabulary = load_data(vocabulary_file)

    print('loading index ...', end=' ')
    sys.stdout.flush()
    index = load_index(index_file)
    print('OK')

    idf = np.log(index['n'] / (index['df'] + 2**-23))
    idf2 = idf ** 2.0

    n_short_list = 100

    score = []

    query_list = [image_list[i] for i in range(0, 4 * N_QUERY, 4)]

    for fname in query_list:
        imfile = join(base_path, fname)

        # compute low-level features
        ffile = join(output_path, splitext(fname)[0] + '.feat')
        if exists(ffile):
            fdict = load_data(ffile)
        else:
            fdict = compute_features(imfile)
        kp, desc = fdict['kp'], fdict['desc']

        # retrieve short list
        dist2 = distance.cdist(desc, vocabulary, metric='sqeuclidean')
        assignments = np.argmin(dist2, axis=1)
        idx, count = np.unique(assignments, return_counts=True)

        query_norm = np.linalg.norm(count)

        # score images using the (modified) dot-product with the query
        scores = dict.fromkeys(index['id2i'], 0)
        for i, idx_ in enumerate(idx): # idx_ is a feature number
            index_i = index['dbase'][idx_]
            for (id_, c) in index_i: # id_ is an image id, c is the number of times feature idx_ appears there.
		#scores[id_] += 1
                #scores[id_] += count[i] * c / index['norm'][id_]
                # score for image id_
                # idf2[i] inverse document frequency for feature i, is it ok to use i as index????
                # count[] number of times features i appears in query image
                # c number of times feature appears in current dataset image
                # index['norm'][id_], norm of the histogram for current dataset image???
                scores[id_] += idf2[i] * count[i] * c / index['norm'][id_]

        # rank list
        short_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_short_list]

        # spatial re-ranking
        fdict1 = fdict
        scores = []
        for id_, _ in short_list:
            i = index['id2i'][id_]
            ffile2 = join(output_path, splitext(image_list[i])[0] + '.feat')
            fdict2 = load_data(ffile2)
            consistency_score = geometric_consistency(fdict1, fdict2)
            scores.append(consistency_score)

        # re-rank short list
        if np.sum(scores) > 0:
            idxs = np.argsort(-np.array(scores))
            short_list = [short_list[i] for i in idxs]

        # get index from file name
        n = int(splitext(fname)[0][-5:])

        # compute score for query + print output
        tp = 0
        print('Q: {}'.format(image_list[n]))
        for id_, s in short_list[:4]:
            i = index['id2i'][id_]
            tp += int((i//4) == (n//4))
            print('  {:.3f} {}'.format(s/query_norm, image_list[i]))
        print('  hits = {}'.format(tp))
        score.append(tp)

    print('retrieval score = {:.2f}'.format(np.mean(score)))
