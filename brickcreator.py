#
#import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#import time
import imageio
from PIL import Image

#from sklearn.cluster  import KMeans
#from sklearn.neighbors import NearestNeighbors

from skimage import transform
import pandas as pd
import matplotlib.animation as animation
import os
from os import path
from random import shuffle
from tqdm import tqdm

#from scipy.cluster.vq import kmeans

#from IPython.display  import Image as display_image

#import warnings
#warnings.filterwarnings("ignore")

class SmartBricks:

    def __init__(self, imgpath=None, Ncolors=None, lowsize=None, outdir=None):

        self.imgpath = imgpath
        self.Ncolors = Ncolors
        self.lowsize = lowsize
        self.outdir = outdir
        self.size = (600 - (lowsize*400/64)) if lowsize > 24 else 550
        #self.loader = loader
        #self.fig = plt.figure(figsize=(20,20))
        img = imageio.imread(self.imgpath)

        #height & width of image
        h, w = img.shape[0:2]
        if img.shape[2] >= 4: img = img[:, :, 0:3]
        #image is bigger than 512 pix?
        if (h > 512) or (w > 512): img = self.resize(img=img, low=256)
        self.img_original = img

        #new_img, indices_new = self.toBrick(img=img)
        new_img_red = self.resize(img=img, low=self.lowsize)
        self.img, indices_new2, self.palette_flat = self.toBrick(img=new_img_red)

        self.h, self.w = self.img.shape[0:2]
        #self.img = self.toLegoColors(new_img_red2)
        self.res2x2, self.res2x1, self.res1x1 = self.getResults()

    def imgFlat(self, img):

        img_flat = np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2])).astype(float)

        return img_flat

    def resize(self, img=None, low=54):

        #plt.figure(figsize=(8,8))
        h, w = img.shape[0:2]
        r = h/w
        if w < h: w, h = low, np.floor(low*r )
        else: h, w = low, np.floor(low/r)

        size = (h, w)
        img0 = transform.resize(img, size)

        #Recover normal RGB values
        img_flat = np.reshape(img0, (img0.shape[0]*img0.shape[1], img0.shape[2])).astype(float)
        new_img = np.ones(img_flat.shape, dtype='float64')
        for i in [0,1,2]:
            new_img[:,i] = img_flat[:,i]*256

        new_img = np.array(new_img, dtype=('uint8'))
        new_img = np.reshape(new_img.flatten(), (img0.shape[0], img0.shape[1], img0.shape[2]))

        return new_img
    '''
    def kmeans(self, img):

        img0 = img[:, :, :3]
        img_flat = np.reshape(img0, (img0.shape[0]*img0.shape[1],img0.shape[2])).astype(float)

        # K-means in sklearn.
        clf = KMeans(n_clusters = self.Ncolors, init='k-means++')
        clf.fit(img_flat.astype(float))
        indices= clf.predict(img_flat)

        kmeans_img = np.reshape(indices, (img0.shape[0], img0.shape[1]))

        return kmeans_img, indices
    '''

    def pil_kmeans(self, img):

        # quantize a image
        pill_img = Image.fromarray(img).quantize(self.Ncolors)
        # Convert Image to RGB and make into Numpy array
        na = np.array(pill_img.convert('RGB'))
        # Get used colours and counts of each
        colours, idx, counts = np.unique(na.reshape(-1,3), return_inverse=True, axis=0, return_counts=1)

        return np.array(pill_img), idx

    def closest_color(self, rgb, COLORS):
        r, g, b = rgb
        color_diffs = []
        for color in COLORS:
            cr, cg, cb = color
            color_diff = np.sqrt(abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2)
            color_diffs.append((color_diff, color))
        return min(color_diffs)[1]

    def toBrick(self, img=None):

        #img0 = img[:, :, :3]
        #img0 = img
        img_flat = np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2])).astype(float)
        new_flat = np.ones(img_flat.shape, dtype='float64')
        palette_flat = []

        #kmeans_img, indices = self.kmeans(img)
        kmeans_img, indices = self.pil_kmeans(img)
        #
        path = os.getcwd()
        df = pd.read_csv('%s/legoKeys.cvs' %(path), sep='\t')

        legoColors = np.empty((len(df), 3), dtype='float64')
        for i in range(len(df)):
            legoColors[i] = [df['R'][i], df['G'][i], df['B'][i]]

        for idx in list(set(indices)):

            mask = (indices == idx)
            current_avg_color = img_flat[mask].mean(axis=0)
            closest_to_lego = self.closest_color(current_avg_color, legoColors)

            new_flat[:][mask] = closest_to_lego
            palette_flat.append([closest_to_lego[0], closest_to_lego[1], closest_to_lego[2]])

        new_flat = np.array(new_flat, dtype=('uint8'))
        palette_flat = np.array(palette_flat, dtype=('uint8'))
        #print(indices_new.shape)
        new_img = np.reshape(new_flat.flatten(), (img.shape[0], img.shape[1], img.shape[2]))

        return new_img, new_flat, palette_flat

    '''
    def toBrick.OLD(self, img=None, plots=False):

        #img0 = img[:, :, :3]
        img0 = img
        img_flat = np.reshape(img0, (img0.shape[0]*img0.shape[1],img0.shape[2])).astype(float)

        #kmeans_img, indices = self.kmeans(img)
        kmeans_img, indices, colours = self.pil_kmeans(img)
        #

        indices_new = np.ones(img_flat.shape, dtype='float64')

        for num, band in enumerate(['R', 'G', 'B']):
            for i in list(set(indices)):

                mask = np.where((indices == i))[0]
                #print(i, len(mask))
                if plots:

                    plt.figure(figsize=(10, 12))
                    bins = np.linspace(0, 256, 40)

                    plt.subplot(3, 1, num+1)
                    plt.title(r'PIXEL ENTRY %s' %(band))
                    plt.hist(img_flat[:,num][mask], bins=bins, histtype='step', lw=2, label=r'cluster %i' %(i))
                    plt.legend()
                #
                indices_new[:,num][mask] = np.median(img_flat[:,num][mask])

                #print(band, i, set(indices_new[:,num][mask]))

        indices_new = np.array(indices_new, dtype=('uint8'))
        print(indices_new.shape)
        new_img = np.reshape(indices_new.flatten(), (img0.shape[0], img0.shape[1], img0.shape[2]))

        return new_img, indices_new


    def toLegoColors(self, img):

        index = 0
        df = pd.read_csv('/home/omar/myproj/SmartBricks/legoKeys.cvs', sep='\t')

        legoColors = np.empty((len(df), 3), dtype='float64')
        for i in range(len(df)):
            legoColors[i] = [df['R'][i], df['G'][i], df['B'][i]]

        imgflat = self.imgFlat(img)

        #Nlist = list(set(imgflat[:,0]))
        #N = len(Nlist)
        #if N != self.Ncolors:
        #print('Found %i LEGO colors instead of %i required.' %(N, self.Ncolors))

        #N = 0
        #while N == self.Ncolors:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(legoColors)
        distances, indices = nbrs.kneighbors(imgflat)
        #print('legocolors:', len(set(indices[:,index])))
        #N = len(set(indices[:,index]))


        img_lego = legoColors[indices[:,index]]
        #print('Founf lego colors: \t %.1f' %(len(set(img_lego[:,0]))))
        img_lego = np.array(img_lego, dtype=('uint8'))
        img_lego = np.reshape(img_lego, (img.shape[0], img.shape[1], img.shape[2]))

        return img_lego


    def palette(self, img=None):

        img_flat = np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2])).astype(float)

        Nlist = list(set(img_flat[:,0]))
        N = len(Nlist)
        if N != self.Ncolors:
            print('Found %i LEGO colors instead of %i required.' %(N, self.Ncolors))


        #for i in range(3):
        #    Nlist = list(set(img_flat[:,i]))
        #    N = len(Nlist)
        #    if N != self.Ncolors:
        #        print('NOT EQUAL! N=%.1f' %(N))
        #    else:
        #        break

        palette = np.empty((self.Ncolors, 3), dtype='uint8')
        #for num, i in enumerate(Nlist):
        for num in range(self.Ncolors):

            if num < N:
                mask = np.where((img_flat[:,0] == Nlist[num]))
                palette[num,:] = (img_flat[mask][0])
            else:
                palette[num,:] = [0, 0, 0]

        palette = np.reshape(palette.flatten(), (2, np.int(self.Ncolors/2), img.shape[2]))


        fig = plt.figure(figsize=(self.Ncolors, 4))

        plt.title(r'PALETA DE COLORES', size=20)
        plt.imshow(palette)

        delta = 1/3
        for i in [0, 1]:
            for j in range(np.int(N/2)):
                Ncolori = np.sum(img_flat[:,0] == palette[i][j][0])
                #plt.text(j-delta, i, r'RGB=(%i, %i, %i)' %(palette[i][j][0], palette[i][j][1], palette[i][j][2]), size=12)
                k = 0
                for num, band in enumerate(['R', 'G', 'B']):
                    plt.text(j-delta, i+k, r'%s=%i' %(band, palette[i][j][num]), size=12)
                    k += 0.1
                plt.text(j-delta, i-delta, r'N=%i' %(Ncolori), size=12)

        fig.savefig('paleta_colores_test.jpeg', bbox_inches = 'tight', pad_inches = 0)

        return palette
    '''

    def brickGrid(self, img, fig, ax):

        h, w = img.shape[0:2]

        #Lego-like bricks style
        brickGrid = []
        for i in range(w):
            for j in range(h):
                brickGrid.append([i, j])
        brickGrid = np.array(brickGrid).T

        #plt.scatter(brickGrid[0], brickGrid[1], s=250, c='w', alpha=0.3)
        plt.scatter(brickGrid[0], brickGrid[1], s=self.size, facecolors='w', edgecolors='k', lw=3, alpha=0.2)

        plt.xticks(np.arange(0, w, 5), fontsize=60)
        plt.yticks(np.arange(0, h, 5), fontsize=60)

        return brickGrid

    def tractor2x2(self, img, kind=None):

        img_test = np.zeros((img.shape[0:2]), dtype=('int'))
        h, w = img.shape[0:2]

        binAi = np.arange(0, w+1, 2)
        binAj = np.arange(0, h+1, 2)
        binBi = np.arange(1, w, 2)
        binBj = np.arange(1, h, 2)

        if kind == 'A': bini, binj = binAj, binAi
        elif kind == 'B': bini, binj = binBj, binAi
        elif kind == 'C': bini, binj = binAj, binBi
        elif kind == 'D': bini, binj = binBj, binBi

        k = 1
        for i in range(len(bini[:-1])):
            for j in range(len(binj[:-1])):

                img_flat = self.imgFlat(img[bini[i]:bini[i+1],binj[j]:binj[j+1]])
                if np.all(img_flat[:,0] == img_flat[:,0][0]):
                    img_test[bini[i]:bini[i+1],binj[j]:binj[j+1]] = k
                    k += 1

        return img_test

    def tractor2x1(self, img, kind=None):

        img_test = np.zeros((img.shape[0:2]), dtype=('int'))
        h, w = img.shape[0:2]

        binAi = np.arange(0, w+1, 2)
        binAj = np.arange(0, h+1, 1)
        binBi = np.arange(1, w, 2)
        binCi = np.arange(0, h+1, 2)
        binCj = np.arange(0, w+1, 1)
        binDi = np.arange(1, h, 2)

        if kind == 'A': bini, binj = binAj, binAi
        elif kind == 'B': bini, binj = binAj, binBi
        elif kind == 'C': bini, binj = binCi, binCj
        elif kind == 'D': bini, binj = binDi, binCj

        k = 1
        for i in range(len(bini[:-1])):
            for j in range(len(binj[:-1])):

                img_flat = self.imgFlat(img[bini[i]:bini[i+1],binj[j]:binj[j+1]])
                if np.all(img_flat[:,0] == img_flat[:,0][0]):
                    img_test[bini[i]:bini[i+1],binj[j]:binj[j+1]] = k
                    k += 1

        return img_test

    def tractor1x1(self, img, kind=None):

        img_test = np.zeros((img.shape[0:2]), dtype=('int'))
        h, w = img.shape[0:2]

        binAi = np.arange(0, h+1, 1)
        binAj = np.arange(0, w+1, 1)

        if kind == 'A': bini, binj = binAi, binAj
        elif kind == 'B': bini, binj = binAi, binAj
        elif kind == 'C': bini, binj = binAi, binAj
        elif kind == 'D': bini, binj = binAi, binAj

        k = 1
        for i in range(len(bini[:-1])):
            for j in range(len(binj[:-1])):

                img_flat = self.imgFlat(img[bini[i]:bini[i+1],binj[j]:binj[j+1]])
                if np.all(img_flat[:,0] == img_flat[:,0][0]):
                    img_test[bini[i]:bini[i+1],binj[j]:binj[j+1]] = k
                    k += 1

        return img_test

    def brickFinder(self, img, brick=None):

        img_new = np.zeros((img.shape[0]*img.shape[1], 4), dtype=('int'))

        for num, kind in enumerate(['A', 'B', 'C', 'D']):
            if brick == '2x2':
                #find all possible 2x2 bricks
                img_test = self.tractor2x2(img=img, kind=kind)
            elif brick == '2x1':
                img_test = self.tractor2x1(img=img, kind=kind)
            elif brick == '1x1':
                img_test = self.tractor1x1(img=img, kind=kind)
            else:
                raise ValueError('brick accepts inputs 2x2 & 2x1 only.')

            img_flat = np.reshape(img_test, (img_test.shape[0]*img_test.shape[1])).astype(float)
            img_new[:,num] = img_flat

        #add pass label to array
        img_new_lab = np.zeros((img_new.shape), dtype=('S8'))
        for i in range(len(img_new)):
            for num, kind in enumerate(['A', 'B', 'C', 'D']):
                img_new_lab[i,num] = kind+str(img_new[i,num])

        unq_img = list(np.unique(img_new_lab))
        #for i in ['A', 'B', 'C', 'D']:
        #    if '%s0' %(i) in unq_img: unq_img.remove(['%s0' %(i)])

        #remove all false passes: the ones with zeroes
        if b'A0' in unq_img: unq_img.remove(b'A0')
        if b'B0' in unq_img: unq_img.remove(b'B0')
        if b'C0' in unq_img: unq_img.remove(b'C0')
        if b'D0' in unq_img: unq_img.remove(b'D0')

        #find list of optimized passes
        keep = []
        rej = []
        idxs = []
        for k in range(5):
            if k > 10:
                rej = []
                for ii in keep:
                    #every brick 2x2 found in previus pass and its neighbours
                    idx = np.where(img_new_lab == ii)[0]
                    for reji in np.unique(img_new_lab[idx]):
                        if reji == b'C4': print(ii)
                        rej.append(reji)

                shuffle(unq_img)
                for i in unq_img:
                    if (i not in rej):
                        keep.append(i)

            else:
                for i in unq_img:
                    if (i not in keep) & (i not in rej):
                        keep.append(i)
                        idx = np.where(img_new_lab == i)[0]
                        idxs.append(idx)
                        for reji in np.unique(img_new_lab[idx]):
                            if reji != i : rej.append(reji)

        #recover lsit of passes to image-like
        img_new_lab2 = np.reshape(img_new_lab, (img.shape[0], img.shape[1], 4))
        idxs = np.array(idxs).flatten()

        return [keep, rej, img_new_lab, img_new_lab2, idxs]

    def drawBrick(self, keep, labels, brick, fig, ax, color='r'):

        print(brick, '\t', len(keep))
        for i in keep:

            delta = 0.04
            x, y = np.where(labels == i)[0], np.where(labels == i)[1]
            ymin = x.min() - 0.5
            xmin = y.min() - 0.5
            if brick == '2x2': w, h = 2, 2
            elif brick == '2x1': h, w = np.abs(x[0]-x[1]) + 1, np.abs(y[0]-y[1]) + 1
            elif brick == '1x1': h, w = 1, 1

            rect = plt.Rectangle((xmin+delta, ymin+delta), w-2*delta, h-2*delta, fill=None, edgecolor=color, alpha=0.8, lw=0.3)
            #rect = plt.Rectangle((xmin, ymin), w, h, color=color, alpha=0.2)
            #plt.scatter(x+0.5, y+0.5, s=4, c='w', alpha=0.07)
            ax.add_patch(rect)

            #return rect

    def brickLabelByColor(self, img, keep, label, RGB):

        new_keep = []
        R, G, B = RGB
        for i in keep:
            x, y = np.where(label == i)[0][0], np.where(label == i)[1][0]
            if (img[x][y][0] == R) & (img[x][y][1] == G) & (img[x][y][2] == B):
                new_keep.append(i)

        return len(new_keep)

    def getResults(self):

        keep2x2, _, labels_flat_2x2, labels_2x2, idxs_2x2 = self.brickFinder(self.img, brick='2x2')
        keep2x1, _, labels_flat_2x1, labels_2x1, idxs_2x1 = self.brickFinder(self.img, brick='2x1')
        keep1x1, _, labels_flat_1x1, labels_1x1, idxs_1x1 = self.brickFinder(self.img, brick='1x1')

        #get 2x1 bricks not in 2x2 bricks
        new_keep2x1 = []
        new_idx2x1 = []
        for i in keep2x1:
            idx = np.where(labels_flat_2x1 == i)[0]
            if (idx[0] not in idxs_2x2) & (idx[1] not in idxs_2x2):
                new_keep2x1.append(i)
                new_idx2x1.append(idx)

        #get 1x1 left bricks
        new_keep1x1 = []
        new_idx1x1 = []
        for i in keep1x1:
            idx = np.where(labels_flat_1x1 == i)[0]
            if (idx[0] not in idxs_2x1) & (idx[0] not in idxs_2x2):
                new_keep1x1.append(i)
                new_idx1x1.append(idx)

        new_idx2x1 = np.array(new_idx2x1).flatten()
        new_idx1x1 = np.array(new_idx1x1).flatten()

        res2x2 = [keep2x2, labels_2x2, idxs_2x2]
        res2x1 = [new_keep2x1, labels_2x1, new_idx2x1]
        res1x1 = [new_keep1x1, labels_1x1, new_idx1x1]

        return res2x2, res2x1, res1x1

    def bricksCanvas(self, img, fig=None, ax=None, RGB=None, res2x2=None, res2x1=None, res1x1=None):

        keep2x2, labels_2x2, idxs_2x2 = res2x2
        new_keep2x1, labels_2x1, idxs_2x1 = res2x1
        new_keep1x1, labels_1x1, idxs_1x1 = res1x1

        if RGB is not None:
            N_keep2x2 = self.brickLabelByColor(img, keep2x2, labels_2x2, RGB)
            N_keep2x1 = self.brickLabelByColor(img, new_keep2x1, labels_2x1, RGB)
            N_keep1x1 = self.brickLabelByColor(img, new_keep1x1, labels_1x1, RGB)

        #new_keep2x2 = keep2x2

        _ = self.brickGrid(img=img, fig=fig, ax=ax)

        #
        #drawBrick(keep=new_keep2x2, labels=labels_2x2, brick='2x2', fig=fig, ax=ax, color='r')
        #drawBrick(keep=new_keep2x1, labels=labels_2x1, brick='2x1', fig=fig, ax=ax, color='yellow')
        #drawBrick(keep=new_keep1x1, labels=labels_1x1, brick='1x1', fig=fig, ax=ax, color='blue')
        self.drawBrick(keep=keep2x2, labels=labels_2x2, brick='2x2', fig=fig, ax=ax, color='k')
        self.drawBrick(keep=new_keep2x1, labels=labels_2x1, brick='2x1', fig=fig, ax=ax, color='k')
        self.drawBrick(keep=new_keep1x1, labels=labels_1x1, brick='1x1', fig=fig, ax=ax, color='k')

        if RGB is not None:
            return N_keep2x2, N_keep2x1, N_keep1x1


    def selectAColor(self, img, RGB, beta=0, idxs=None):
        """ Select one of the Ncolors on bricked image """

        img_flat = self.imgFlat(img)
        R, G, B = RGB

        mask = np.ones(len(img_flat), dtype=bool)

        mask &= (img_flat[:,0] == R) & (img_flat[:,1] == G) & (img_flat[:,2] == B)
        N = []

        #print(len(set(img_flat[keep][:,0])), img_flat[keep][0])

        #increase brightness of selected color
        if idxs is not None:
            j = 0.2
            for bricksize in range(3):
                mask_i = np.zeros(len(img_flat), dtype=bool)
                mask_i[idxs[bricksize]] = True
                keep_i = np.where((mask) & (mask_i))

                #rej_i = np.where(~((mask) & (mask_i)))
                #if bricksize == 0: beta_i = beta*0.5
                #else: beta_i = beta*bricksize
                #img_flat[keep_i] = (1.1+j) * img_flat[keep_i] + beta
                img_flat[keep_i] = 1.2 * img_flat[keep_i] + beta*j
                j += 0.2
                #img_flat[keep_i] = bricksize * img_flat[keep_i]

                if bricksize == 0: N.append(int(len(keep_i[0])/4))
                elif bricksize == 1: N.append(int(len(keep_i[0])/2))
                elif bricksize == 2: N.append(int(len(keep_i[0])))

        else:

            keep = np.where(mask)
            rej = np.where(~mask)
            img_flat[keep] = img_flat[keep] + beta
            #reduce contrast of non selected colors
            alpha = 1
            img_flat[rej] = alpha * img_flat[rej]

        img_new = np.array(img_flat, dtype=('uint8'))
        img_new = np.reshape(img_new, (img.shape[0], img.shape[1], img.shape[2]))

        return img_new, N

    #from PIL import Image, ImageDraw

    def makeGiff(self, img, RGB, idxs=None, pathdir=None, fig=None):

        ims = []
        betas = np.linspace(0, 150, 2)
        ax = fig.add_subplot(111)
        #ax.axis('off')
        fig.subplots_adjust(left=0.05, bottom=0.05, right=1, top=1, wspace=None, hspace=None)
        #fig = plt.figure(figsize=(20,20))
        #ax = plt.gca()

        #N2x2, N2x1, N1x1 =
        #bricksCanvas(img, fig=fig, ax=ax, RGB=None, res2x2=res2x2, res2x1=res2x1, res1x1=res1x1)
        #print(N2x2, N2x1, N1x1)

        for beta in betas:
            if beta == 0:
                new_frame, _ = self.selectAColor(img=img, RGB=RGB, beta=beta, idxs=None)
            else:
                new_frame, N = self.selectAColor(img=img, RGB=RGB, beta=beta, idxs=idxs)
            #

            im = ax.imshow(new_frame, animated=True)

            #ax.axvline(15)
            #im2
            #
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                    repeat_delay=5)
        R,G,B = RGB
        filename = '%s/%s_%s_%s.gif' %(pathdir, str(R), str(G), str(B))
        if path.exists(filename): os.remove(filename)
        ani.save(filename, writer='imagemagick')

        #N_keep2x2 = brickLabelByColor(img, res2x2[0], res2x2[1], RGB)
        #N_keep2x1 = brickLabelByColor(img, res2x1[0], res2x1[1], RGB)
        #N_keep1x1 = brickLabelByColor(img, res1x1[0], res1x1[1], RGB)

        #return N_keep2x2, N_keep2x1, N_keep1x1
        return N[0], N[1], N[2]

    def saveProj(self):

        #start = time.time()
        ispathdir = os.path.isdir(self.outdir)
        if not ispathdir: os.makedirs(self.outdir, exist_ok=True)


        #fig0 = fig
        fig = plt.figure(figsize=(30,30))
        ax = plt.gca()

        self.bricksCanvas(img=self.img, fig=fig, ax=ax, RGB=None, res2x2=self.res2x2, res2x1=self.res2x1, res1x1=self.res1x1)
        figcvs = fig
        figall = fig
        #figoriginal = fig.copy

        #paletteLego = self.palette(self.img)
        #palette_flat = self.imgFlat(paletteLego)

        table = []
        #for num, pal in enumerate(palette_flat):
        for i in tqdm(range(len(self.palette_flat))):

            pal = self.palette_flat[i]
            N2x2, N2x1, N1x1 = self.makeGiff(img=self.img, RGB=pal, idxs=[self.res2x2[2], self.res2x1[2], self.res1x1[2]], pathdir=self.outdir, fig=figcvs)
            r,g,b = pal
            color = '%s_%s_%s' %(r,g,b)
            table.append([color, N2x2, N2x1, N1x1])
            self.loader = i

        t = np.array(table)
        N2x2total = np.sum(t[:,1].astype(int))
        N2x1total = np.sum(t[:,2].astype(int))
        N1x1total = np.sum(t[:,3].astype(int))
        table.append(['total', N2x2total, N2x1total, N1x1total])

        #fig = plt.figure(figsize=(20,20))
        ax = figall.add_subplot(111)
        ax.imshow(self.img)
        #bricksCanvas(img, fig=fig, ax=ax, RGB=None, res2x2=res2x2, res2x1=res2x1, res1x1=res1x1)

        figall.savefig('%s/all.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)

        fig0 = plt.figure(figsize=(12,12))
        ax = fig0.add_subplot(111)
        plt.imshow(self.img_original)
        fig0.savefig('%s/original.jpeg' %(self.outdir), bbox_inches = 'tight', pad_inches = 0)

        np.save('%s/table' %(self.outdir), table)

        #end = time.time()
        #print('Total run time: %f sec' %(end - start))
