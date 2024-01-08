from get_convolve_img import get_convovle_img
import tensorflow as tf
import numpy as np

class Vertex_1():
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f


class Vertex_2():
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f

class Vertex():
    def __init__(self, num, t_x = None, t_y = None):
        self.num = num
        self.x = num % 28
        self.y = num // 28
        self.t_x = t_x
        self.t_y = t_y

class Edge():
    def __init__(self, vertex_1, vertex_2, e_x = None, e_y = None):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.e_x = e_x
        self.e_y = e_y

class Face():
    def __init__(self, vertex_1, vertex_2, vertex_3, f_x = None, f_y = None):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3
        self.f_x = f_x
        self.f_y = f_y

def get_points(img1, img2):
    lv = len(img1)
    vertex1 = []
    vertex2 = []
    vertex = []
    for k in range(lv):
        for w in range(lv):
            vertex1.append(Vertex_1(k,w,img1[k][w]))
    for k in range(lv):
        for w in range(lv):
            vertex2.append(Vertex_2(k,w,img2[k][w]))
    for k in range(lv*lv):
        vertex.append(Vertex(k, t_x=vertex1[k].f, t_y=vertex2[k].f))
    return vertex

def get_edges(vertex, n):
    edge = []
    for i_0 in range(n):
        for j_0 in range(n-1):
            edge.append(Edge(vertex[n*i_0+j_0], vertex[n*i_0+j_0+1]))
    for i_1 in range(n):
        for j_1 in range(n-1):
            edge.append(Edge(vertex[i_1+n*j_1], vertex[i_1+n*j_1+n]))
    for i_2 in range(n-1):
        for j_2 in range(n-1):
            edge.append(Edge(vertex[i_2*n+j_2],vertex[(i_2+1)*n+j_2+1]))

    for ed in edge:
        ed.e_x = max(ed.vertex_1.t_x, ed.vertex_2.t_x)
        ed.e_y = max(ed.vertex_1.t_y, ed.vertex_2.t_y)
    return edge

def get_faces(vertex, n):
    face = []
    for i in range(n-1):
        for j in range(n-1):
            face.append(Face(vertex[n*i+j],vertex[n*i+j+1],vertex[n*(i+1)+j+1]))
            face.append(Face(vertex[n*i+j],vertex[n*(i+1)+j],vertex[n*(i+1)+j+1]))
    for i in range(len(face)//2):
        face[2*i].f_x = max(face[2*i].vertex_1.t_x, face[2*i].vertex_2.t_x, face[2*i].vertex_3.t_x)
        face[2 * i].f_y = max(face[2 * i].vertex_1.t_y, face[2 * i].vertex_2.t_y, face[2 * i].vertex_3.t_y)
        face[2 * i + 1].f_x = max(face[2 * i + 1].vertex_1.t_x, face[2 * i + 1].vertex_2.t_x,
                                  face[2 * i + 1].vertex_3.t_x)
        face[2 * i+1].f_y = max(face[2 * i+1].vertex_1.t_y, face[2 * i+1].vertex_2.t_y, face[2 * i+1].vertex_3.t_y)
    return face