import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations


def hypergraph_pcm(
    H1: np.ndarray, H2: np.ndarray, reordered: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    r1, n1 = H1.shape
    r2, n2 = H2.shape

    HX_left = np.kron(H1, np.eye(n2))
    HX_right = np.kron(np.eye(r1), H2.T)
    HX = np.append(HX_left, HX_right, axis=1)

    HZ_left = np.kron(np.eye(n1), H2)
    HZ_right = np.kron(H1.T, np.eye(r2))
    HZ = np.append(HZ_left, HZ_right, axis=1)

    if reordered:
        HX_left_split = np.split(HX_left, n1, axis=1)
        HX_right_split = np.split(HX_right, r1, axis=1)
        HX_split = []
        for i in range(n1):
            HX_split.append(HX_left_split[i])
            if i < r1:
                HX_split.append(HX_right_split[i])
        HX = np.concatenate(tuple(HX_split), axis=1)

        HZ_left_split = np.split(HZ_left, n1, axis=1)
        HZ_right_split = np.split(HZ_right, r1, axis=1)
        HZ_split = []
        for i in range(n1):
            HZ_split.append(HZ_left_split[i])
            if i < r1:
                HZ_split.append(HZ_right_split[i])
        HZ = np.concatenate(tuple(HZ_split), axis=1)

    return HX.astype(int), HZ.astype(int)


def classical_pcm(clist: list) -> np.ndarray:
    num_bits = clist.count("B")
    H = []
    for i in range(len(clist)):
        if clist[i] == "C":
            peak_i = i + 1
            one_hot_vec = np.zeros(num_bits)
            while peak_i < len(clist) and type(clist[peak_i]) != str:
                one_hot_vec[clist[peak_i]] = 1
                peak_i += 1
            H.append(one_hot_vec)

    return np.array(H, dtype=int)