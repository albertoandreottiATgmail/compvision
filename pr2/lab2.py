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
np.seterr(all='raise')

import cv2

from utils import load_data, save_data, load_index, save_index, get_random_sample, compute_features, arr2kp

from sklearn.cluster import KMeans
from scipy.spatial import distance

import base64

from skimage.measure import ransac
from skimage.transform import AffineTransform

N_QUERY = 100
GEO_CHECK = True
NORM_L2 = False
pca_dim = 32
pca_enabled = False


def read_image_list(imlist_file):
    return [f.rstrip() for f in open(imlist_file)]


def geometric_consistency(feat1, feat2):

    if not GEO_CHECK:
        return 0

    kp1, desc1 = feat1['kp'], feat1['desc']
    desc1 = pca_project(desc1, P, mu, pca_dim)

    kp2, desc2 = feat2['kp'], feat2['desc']
    desc2 = pca_project(desc2, P, mu, pca_dim)

    # 1) matching de features
    matcher = cv2.BFMatcher()
    matches = matcher.match(np.float32(desc1), np.float32(desc2))
    src = []
    dst = []
    for match in matches:
        kp_im1 = kp1[match.queryIdx].flatten().take([0,1])
        kp_im2 = kp2[match.trainIdx].flatten().take([0,1])
        src.append(kp_im1), dst.append(kp_im2)

    # src and dst are list of lists that contain 2 elements
    model_robust, inliers = ransac((np.array(src), np.array(dst)), AffineTransform, min_samples=4,
    residual_threshold=6, max_trials=350)

    '''

    2) Estimar una tranformación afín empleando RANSAC
       a) armar una función que estime una transformación afin a partir de
          correspondencias entre 3 pares de puntos (solución de mínimos
          cuadrados en forma cerrada)
       b) implementar RANSAC usando esa función como estimador base
    3) contar y retornar número de inliers
    '''

    return sum(inliers)


def pca_fit(samples):
    _, ndim = samples.shape

    # 1) computar la media (mu) de las muestras de entrenamiento
    mu = samples.mean(axis = 0)

    # 2) computar matriz de covarianza
    diffs = samples - mu
    N = samples.shape[0]
    covmat = np.dot(diffs.transpose(), diffs) / N

    # 3) computar autovectores y autovalores de la matriz de covarianza
    eigvals, eigvects = np.linalg.eig(covmat)

    # 4) ordenar autovectores por valores decrecientes de los autovectores
    indexes = np.argsort(eigvals).tolist()
    indexes.reverse()
    P = eigvects.transpose()[:, indexes]

    return P.transpose(), mu


# project vectors or entire datasets
def pca_project(x, P, mu, dim):

    if not pca_enabled:
        if NORM_L2:
            x /= (np.linalg.norm(x, ord=2) + 1e-7)
        return x

    reshaped = (x - mu)
    if len(x.shape) == 1:
        reshaped = reshaped.reshape(1, -1)

    projection = np.dot(reshaped, P[:, :dim]).squeeze()
    if NORM_L2:
        projection /= (np.linalg.norm(projection, ord=2) + 1e-7)

    return projection

if __name__ == "__main__":
    random_state = np.random.RandomState(12345)

    # ----------------
    # BUILD VOCABULARY
    # ----------------

    #unsup_base_path = '/media/jrg/DATA/Datasets/UKB/ukbench/full/'
    unsup_base_path = './full'
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

    # train PCA
    pca_file = join(output_path, 'pca_{:d}.dat'.format(pca_dim))
    samples = load_data(unsup_samples_file)
    P, mu = pca_fit(samples)

    # compute vocabulary
    n_clusters = 1000
    vocabulary_file = join(output_path, 'vocabulary_{:d}.dat'.format(n_clusters))
    if not exists(vocabulary_file):
        samples = load_data(unsup_samples_file)
        # project samples to n_dim vectors
        # pr_samples = np.dot(P[:n_dim, :], (samples - mu).transpose())
        pr_samples = pca_project(samples, P, mu, pca_dim)
        kmeans = KMeans(n_clusters=n_clusters, verbose=1, n_jobs=-2)
        kmeans.fit(pr_samples)
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
            'norm': {},                                           # L2-norms
            'nd': {}                                              # number of features per image
        }

        n_images = len(image_list)

        for i, fname in enumerate(image_list):
            imfile = join(base_path, fname)
            imID = base64.encodestring(fname) # as int? / simlink to filepath?
            if imID in index['id2i']:
                continue
            index['id2i'][imID] = i

            # retrieve keypoints and local descriptors
            ffile = join(output_path, splitext(fname)[0] + '.feat')
            fdict = load_data(ffile)
            kp, desc = fdict['kp'], fdict['desc']

            nd = len(desc)
            if nd == 0:
                continue

            # project desc
            desc = pca_project(desc, P, mu, pca_dim)
            dist2 = distance.cdist(desc, vocabulary, metric='sqeuclidean')
            assignments = np.argmin(dist2, axis=1)
            # idx and count are the bag of visual words - which visual words appear how many times in the image
            idx, count = np.unique(assignments, return_counts=True)
            for j, c in zip(idx, count):
                index['dbase'][j].append((imID, c))
            index['n'] += 1
            index['df'][idx] += 1 # increase one to all feature ids in idx.
            index['norm'][imID] = np.linalg.norm(count)
            index['nd'][imID] = float(nd)

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
    # number of documents / number of times the VW appears in any document(ignore a VW appearing multiple times)
    idf = np.log(index['n'] / (index['df'] + 2**-23))

    n_short_list = 100

    score = []

    # images used to query, i goes [0, 4, 8, ..., 396]
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

        # project desc
        desc = pca_project(desc, P, mu, pca_dim)
        # get visual word assignments + counts
        dist2 = distance.cdist(desc, vocabulary, metric='sqeuclidean')
        assignments = np.argmin(dist2, axis=1)
        idx_qy, count_qy = np.unique(assignments, return_counts=True)

        # score ALL images using the (modified) dot-product with the query
        scores = dict.fromkeys(index['id2i'], 0) # id-> score

        # flat/cosine/IK similarities ------------------------------------------
        # query_norm = np.linalg.norm(count_qy)
        # count_qy = count_qy.astype(np.float)  # otherwise =/ raises an exception
        # count_qy /= (query_norm + 2**-23)     # comment this line for flat scoring

        # for i, idx_qy_i in enumerate(idx_qy):  # for each VW in the query
        #    inverted_list = index['dbase'][idx_qy_i]   # retrieve inv. list
        #    for (img_id, count_db_i) in inverted_list:
        #        # flat scores
		#        scores[img_id] += 1
        #        # cosine similarity = dot-prod. between l2-normalized BoVWs
        #        # scores[img_id] += count_qy[i] * count_db_i / index['norm'][img_id]

        #        # # intersection kernel
        #        # scores[img_id] += ...

        # tf-idf ---------------------------------------------------------------

        tf_idf_qy = idf[idx_qy] * count_qy / float(len(desc))
        tf_idf_qy /= (np.linalg.norm(tf_idf_qy) + 2**-23)

        tf_idf_db_norm = dict.fromkeys(index['id2i'], 0)

        for i, idx_qy_i in enumerate(idx_qy):
            inverted_list = index['dbase'][idx_qy_i]
            for (img_id, count_db_i) in inverted_list:
                tf_idf_db_i = idf[idx_qy_i] * count_db_i / index['nd'][img_id]
                tf_idf_db_norm[img_id] += tf_idf_db_i ** 2.0
                scores[img_id] += tf_idf_qy[i] * tf_idf_db_i

        for img_id in scores.keys():
            scores[img_id] /= np.sqrt(tf_idf_db_norm[img_id] + 2**-23)

        # ----------------------------------------------------------------------

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
        if np.sum(scores) > 0.0:
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
            print('  {:.3f} {}'.format(s, image_list[i]))
        print('  hits = {}'.format(tp))
        score.append(tp)

    print('retrieval score = {:.2f}'.format(np.mean(score)))
