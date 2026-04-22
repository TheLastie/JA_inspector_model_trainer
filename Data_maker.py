import numpy as np


class SplittedData:
    def __init__(self, points):
        self.points = {i: (np.array(points[i][0]), points[i][1]) for i in points}  # [np.arr(x, y), conf] * [1 - 13]
        self.body_vectors = [(0, 0, 0)] * 11  # x, y, length
        self.angles = [0] * 11
        self.maked = 0
        self.make_vectors()
        self.make_angles()

    def make_vectors(self):
        self.maked = 1
        self.body_vectors[0] = self.points[2][0] - self.points[5][0]
        self.body_vectors[1] = self.points[2][0] - self.points[3][0]
        self.body_vectors[2] = self.points[3][0] - self.points[4][0]
        self.body_vectors[3] = self.points[5][0] - self.points[6][0]
        self.body_vectors[4] = self.points[6][0] - self.points[7][0]
        self.body_vectors[5] = self.points[1][0] - self.points[11][0]
        self.body_vectors[6] = self.points[1][0] - self.points[8][0]
        self.body_vectors[7] = self.points[8][0] - self.points[9][0]
        self.body_vectors[8] = self.points[11][0] - self.points[12][0]
        self.body_vectors[9] = self.points[12][0] - self.points[13][0]
        self.body_vectors[10] = self.points[9][0] - self.points[10][0]
        for i, val in enumerate(self.body_vectors):
            self.body_vectors[i] = val, np.linalg.norm(val)

    def angle_2_vec(self, vec1, vec2, vec1l=None, vec2l=None):
        if vec1l and vec2l:
            cos_val = np.clip(np.dot(vec1 / vec1l, vec2 / vec2l), -1.0, 1.0)
            return np.arccos(cos_val)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_val = np.clip(np.dot(vec1 / norm1, vec2 / norm2), -1.0, 1.0)
        return np.arccos(cos_val)

    def make_angles(self):
        if not self.maked:
            self.make_vectors()
        self.angles[0] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[3][0],
                                          self.body_vectors[0][1], self.body_vectors[3][1])
        self.angles[1] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[1][0],
                                          self.body_vectors[0][1], self.body_vectors[1][1])
        self.angles[2] = self.angle_2_vec(self.body_vectors[5][0], self.body_vectors[6][0],
                                          self.body_vectors[5][1], self.body_vectors[6][1])
        self.angles[3] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[5][0],
                                          self.body_vectors[0][1], self.body_vectors[5][1])
        self.angles[4] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[6][0],
                                          self.body_vectors[0][1], self.body_vectors[6][1])
        self.angles[5] = self.angle_2_vec(self.body_vectors[1][0], self.body_vectors[2][0],
                                          self.body_vectors[1][1], self.body_vectors[2][1])
        self.angles[6] = self.angle_2_vec(self.body_vectors[6][0], self.body_vectors[7][0],
                                          self.body_vectors[6][1], self.body_vectors[7][1])
        self.angles[7] = self.angle_2_vec(self.body_vectors[5][0], self.body_vectors[8][0],
                                          self.body_vectors[5][1], self.body_vectors[8][1])
        self.angles[8] = self.angle_2_vec(self.body_vectors[8][0], self.body_vectors[9][0],
                                          self.body_vectors[8][1], self.body_vectors[9][1])
        self.angles[9] = self.angle_2_vec(self.body_vectors[7][0], self.body_vectors[10][0],
                                          self.body_vectors[7][1], self.body_vectors[10][1])
        self.angles[10] = self.angle_2_vec(self.body_vectors[3][0], self.body_vectors[4][0],
                                           self.body_vectors[3][1], self.body_vectors[4][1])

    def return_data(self):
        full_dict = {}
        full_dict['vectors'] = [((float(i[0][0]), float(i[0][1])), float(i[1])) for i in self.body_vectors]
        full_dict['angles'] = list(map(float, self.angles))
        return full_dict
