# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from os.path import exists

import cv2

import numpy as np

import matplotlib.pyplot as plt

# leer argumentos por linea de comandos
import argparse


def show_and_wait(img):
    '''
    muestra una imagen y espera que se presione una tecla
    '''
    cv2.imshow('Imagen', img)
    print('Presione \'c\' para continuar ...')
    ch = None
    while ch not in (ord('c'), ord('C')):
        ch = cv2.waitKey(10) & 255
    cv2.destroyWindow('Imagen')


def ejemplo_1(filename):
    '''
    Lectura y visualización
    '''
    if not exists(filename):
        raise IOError('el archivo \'{}\' no se puede leer'.format(filename))

    # leer la imagen de disco (formato BGR)
    img = cv2.imread(filename)

    # visualización con OpenCV
    print('visualización con OpenCV')
    show_and_wait(img)

    # visualización con matplotlib
    print('visualización con pyplot')
    plt.imshow(img)
    plt.show()

    # conversión de espacio de colores
    print('visualización con pyplot de una imagen leída con OpenCV')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def ejemplo_2(filename):
    '''
    Manipulación
    '''
    if not exists(filename):
        raise IOError('el archivo \'{}\' no se puede leer'.format(filename))

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # leer el tamaño
    h, w = img.shape[:2]
    print('imagen de {}x{} pixels de tipo {}'.format(w, h, img.dtype))
    show_and_wait(img)

    # cuidado con los cambios de tipo
    img = img + 0.0
    print('ahora de tipo {}'.format(img.dtype))
    show_and_wait(img)

    # para visualizar imágenes tipo float, tienen que estar en el rango [0, 1]
    img = img / img.max()
    print('de tipo {} en rango [0, 1]'.format(img.dtype))
    show_and_wait(img)

    # transformaciones puntuales
    print('transformación puntual')
    img = np.sqrt(img)
    show_and_wait(img)

    # "borrar" bloque
    print('acceso a matriz')
    img[0:h//2, 0:w//2] = 0.5    # 0.5 es un valor de gris 128
    show_and_wait(img)

    # rotar 90
    print('rotación')
    img = img.transpose()
    img = np.flipud(img)
    show_and_wait(img)



def ejemplo_3(filename):
    '''
    Filtrado
    '''
    if not exists(filename):
        raise IOError('el archivo \'{}\' no se puede leer'.format(filename))

    # leer la imagen directamente en escala de grises
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.

    # filtro box de (2K+1) x (2K+1)
    print('box filter')
    K = 5
    kernel = np.ones((2*K+1, 2*K+1)) / (2*K+1)**2
    img_filtered = cv2.filter2D(img, -1, kernel)
    show_and_wait(img_filtered)

    # filtro Gaussiano separable
    for sigma in np.arange(1., 10., 2.):
        K = int(3 * sigma)
        h = np.exp(-0.5 * np.arange(-K, K+1)**2 / sigma**2)
        kernel = np.dot(h.reshape((-1, 1)), h.reshape((1, -1)))
        kernel /= kernel.sum()
        img_filtered = cv2.filter2D(img, -1, kernel)
        print('Gaussian filter: sigma={}, K={}'.format(sigma, K))
        show_and_wait(img_filtered)

    # sharpening step-by-step
    print('Sharpening')
    kernel = np.ones((5, 5)) / 25.
    img_smoothed = cv2.filter2D(img, cv2.CV_64F, kernel)
    img_filtered = img + (img - img_smoothed)
    show_and_wait(img_filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Algunas operaciones sobre imágenes con OpenCV',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('FILE', help='archivo', type=str)
    args = vars(parser.parse_args())

    ejemplo_1(args['FILE'])
    ejemplo_2(args['FILE'])
    ejemplo_3(args['FILE'])
