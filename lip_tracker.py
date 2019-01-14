
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from os import listdir
import matplotlib.gridspec as gridspec

from collections import namedtuple
from math import sqrt
import random
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import colorsys
import sys
import numpy.ma as ma
Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))
Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(img, n=3):

    w, h = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

def get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#")  # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)
def rgb2gray(rgb):
    grey=np.zeros(np.shape(rgb))
    for x in range(np.shape(rgb)[0]):
        grey[x,:]=np.dot(rgb[x,:], [0.299, 0.587, 0.114])

    return grey

def extraction(image,idx):
    width=np.shape(image)[1]
    height=np.shape(image)[0]
    face_landmark = face_recognition.face_landmarks(image)[idx]

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(face_landmark['top_lip'], outline=1, fill=1)
    ImageDraw.Draw(img).polygon(face_landmark['bottom_lip'], outline=1, fill=1)
    mask = np.array(img)

    extracted_points=np.expand_dims(image[mask==1,:], 0)

    ncolors=1

    collist = list(colorz(Image.fromarray(extracted_points), n=ncolors))

    return collist
def main():
    folder_name='riri_samps'
    locations=listdir(folder_name)
    fig, ax =plt.subplots(2,4,figsize=(20,10),sharey=True,sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)



    for i,location in enumerate(locations):

        image = face_recognition.load_image_file(folder_name+'/'+location)
        width=np.shape(image)[1]
        height=np.shape(image)[0]
        face_landmark = face_recognition.face_landmarks(image)[0]

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(face_landmark['top_lip'], outline=1, fill=1)
        ImageDraw.Draw(img).polygon(face_landmark['bottom_lip'], outline=1, fill=1)
        mask = np.array(img)

        new_image = image
        extracted_points=np.expand_dims(image[mask==1,:], 0)

        new_image[mask==0,:] = rgb2gray(new_image[mask==0,:])

        pil_image = Image.fromarray(new_image)
        width, height = pil_image.size   # Get dimensions

        left = (width - 1080)/2
        top = (height - 1080)/2
        right = (width + 1080)/2
        bottom = (height + 1080)/2

        pil_image.crop((left, top, right, bottom))


        img = new_image
        ncolors=2

        ax[int(np.floor(i / 4)),i%4].imshow(pil_image)
        ax[int(np.floor(i / 4)),i%4].axis('off')
        recs=[]
        collist = list(colorz(Image.fromarray(extracted_points), n=ncolors))



        RGB_list = []
        for color in collist:
            RGB_list.append(get_hsv(color))

        ### hacky fix but i'm noticing that it's generally taking to dark of a color here, we take the top two colors and choose the lighter of the two
        s = sorted(RGB_list,key=lambda x: x[2])
        sorted_colors = [colorsys.hsv_to_rgb(color[0], color[1], color[2]) for color in s]
        sorted_colors=[sorted_colors[-1]]

        for color in sorted_colors:
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=color))
        ax[int(np.floor(i / 4)),i%4].legend(recs, [str(col) for col in collist],loc='upper right')
        plt.axis('off')

    plt.savefig('figures/out.png',bbox_inches='tight',pad_inches=0.0,dpi=300)



if __name__ == '__main__':
    main()