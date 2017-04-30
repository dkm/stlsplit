#!/usr/bin/env python

# Copyright (C) 2017 Marc Poulhiès <dkm@kataplop.net>
#               2017 Diane Larlus  <diane.larlus@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import stl
import sys
import numpy
from scipy.spatial import distance
from scipy.cluster.vq import vq, kmeans, whiten

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage

import argparse

parser = argparse.ArgumentParser(description='Split STL in several objects.')
parser.add_argument("-f", '--file', required=True, help='STL file to split')
parser.add_argument("-o", '--output', default='split', required=True, help='STL file basename for splited objects')
parser.add_argument("-n", '--number-of-objects', type=int, required=True, help='Number of objects expected after the split')
parser.add_argument("-d", '--view', action='store_true', help='Display split objets')

parsed_args = parser.parse_args(sys.argv[1:])

stlmesh = stl.mesh.Mesh.from_file(parsed_args.file)

# take 1 point in all triangle
w = stlmesh.vectors.sum(axis=1)/3

# do not split along Z
wf = w[:,0:2]

def kmeanstest():
    #whitened = whiten(wf)
    res = kmeans(wf, 2)
    c1 = res[0][0]
    c2 = res[0][1]
    assign = [ 0 if distance.euclidean(x, c1) < distance.euclidean(x, c2) else 1 for x in wf]
    return assign

def display(clusters):
    from matplotlib import pyplot
    from mpl_toolkits import mplot3d

    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    for k,cluster in clusters.items():
        print ('New cluster to print: {}'.format(k))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(numpy.array(cluster)))

    # Auto scale to the mesh size
    scale = stlmesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()
    

def hierachical():
    Z = linkage(wf, 'single')
    res = fcluster(Z, parsed_args.number_of_objects, criterion='maxclust')
    return res

clusters = {}

assign = hierachical()

for clus_assign,triangle in zip(assign, stlmesh.vectors):
    ## triangle is [ [3] * 3 ]
    if clus_assign not in clusters:
        clusters[clus_assign] = []
    clusters[clus_assign].append(triangle)

for k,cluster in clusters.items():
    print ('New cluster to print: {}'.format(k))
    data = numpy.zeros(len(cluster), dtype=stl.mesh.Mesh.dtype)
    data['vectors'] = numpy.array(cluster)
    m = stl.mesh.Mesh(data.copy())
    m.save('{}_{}.stl'.format(parsed_args.output, k), mode=stl.Mode.ASCII)

if parsed_args.view:
    display(clusters)
