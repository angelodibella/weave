import numpy as np


def repetition(n: int) -> np.ndarray:
    H = np.zeros((n - 1, n), dtype=int)
    np.fill_diagonal(H, 1)
    H[:, -1] = 1

    return H


def hamming(n: int) -> np.ndarray:
    m = int(np.ceil(np.log2(n + 1)))
    if 2 ** m - 1 != n:
        raise ValueError("Invalid n for a Hamming code. Ensure n = 2^m - 1.")

    H = np.zeros((m, n), dtype=int)
    for i in range(1, n + 1):
        binary_str = format(i, f"0{m}b")[::-1]
        H[:, i - 1] = [int(bit) for bit in binary_str]

    return H


def hypergraph_product(
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


def to_clist(H: np.ndarray) -> list:
    clist = ["B"] * H.shape[1]
    for row in H:
        clist.append("C")
        for i, col in enumerate(row):
            if col == 1:
                clist.append(i)

    return clist


def to_matrix(clist: list) -> np.ndarray:
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


def find_pivot_columns(matrix: np.ndarray) -> list[int]:
    rows, cols = matrix.shape
    pivot_columns = []

    row_index = 0
    for col_index in range(cols):
        found_pivot = False
        for i in range(row_index, rows):
            if matrix[i, col_index] == 1:
                pivot_columns.append(col_index)
                found_pivot = True

                if i != row_index:
                    matrix[[i, row_index]] = matrix[[row_index, i]]

                for j in range(rows):
                    if j != row_index and matrix[j, col_index] == 1:
                        matrix[j] = (matrix[j] + matrix[row_index]) % 2

                row_index += 1
                break

        if not found_pivot:
            continue

    return pivot_columns
