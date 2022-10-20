import torch


def generate_positions(root_paths, max_width, max_depth):
    """
    root_paths: List([ch_ids]), ch_ids \in [0, 1, ..., max_width)
    returns: Tensor [len(root_paths), max_width * max_depth]
    """
    for i, path in enumerate(root_paths):
        # stack-like traverse
        root_paths[i] = [min(ch_id, max_width - 1) for ch_id in root_paths[i]]
        if len(root_paths[i]) > max_depth:
            root_paths[i] = root_paths[i][-max_depth:]
        # pad
        root_paths[i] = root_paths[i][::-1] + \
                        [max_width] * (max_depth - len(root_paths[i]))
    root_path_tensor = torch.LongTensor(root_paths)
    onehots = torch.zeros((max_width + 1, max_width))
    for i in range(max_width):
        onehots[i, i] = 1.0
    embeddings = torch.index_select(
        onehots, dim=0, index=root_path_tensor.view(-1))
    embeddings = embeddings.view(
        root_path_tensor.shape + (embeddings.shape[-1],))
    embeddings = embeddings.view((root_path_tensor.shape[0], -1))
    return embeddings


def get_adj_matrix(edges, code_len, use_self_loops):
    adj_matrix = torch.zeros((2, len(edges), code_len, code_len))
    for i, edge_list in enumerate(edges):
        edge_tensor = torch.LongTensor(edge_list)
        adj_matrix[0, i][edge_tensor[:, 0], edge_tensor[:, 1]] = 1
        adj_matrix[1, i][edge_tensor[:, 1], edge_tensor[:, 0]] = 1
        if use_self_loops:
            arange = torch.arange(code_len)
            adj_matrix[0, i][arange, arange] = 1
            adj_matrix[1, i][arange, arange] = 1
    # adj_matrix = torch.cat([adj_matrix, adj_matrixT], dim=0)
    return adj_matrix  # 2, edges, seq, seq


def generate_local_relations(relations, max_ary):
    '''
    relations:[{"0':[1,2,1]."1":[2,3,0]}]
    return tensor:[len(root_paths) x len(root_paths) x 3]
    '''
    overflow_flag = max_ary

    def convert(x):
        # +1 for the overflow flag
        if x >= max_ary:
            return overflow_flag
        else:
            return x

    totals = len(relations)
    result = torch.zeros(totals, totals, 3)
    for i, relation in enumerate(relations):
        for (key, val) in relation.items():
            new_val = [convert(val[0] + 1), convert(val[1]), val[-1]]
            result[i, int(key)] = torch.tensor(new_val)
    return result


def generate_TD_position(td_paths, max_ary, max_depth):
    '''
    td_paths: len,depth,2
    the root is [], need converted first
    first dim range 0-max_width
    second dim range 1-max_width+1
    returns: tensor: [len(root_paths), max_depth, 2]
    # [ [], [[1,1]], [[1,1],[2,2]] , [[3,3],[2,2],[3,3]] ]

    '''

    for td_path in td_paths:
        td_path.insert(0, [0, 1])
    # add virtual node

    # [ [[1, 1]], [[1, 1],[1,1]], [[1, 1],[1,1],[2,2]] , [[1, 1],[3,3],[2,2],[3,3]] ]

    overflow_flag = max_ary


    def abs(length):
        return length if length >= 0 else 0

    def convert(x):
        # +1 for the overflow flag
        if x >= max_ary:
            return overflow_flag
        else:
            return x

    sample_idx = []

    for td_path in td_paths:
        tds = [[convert(x + 1), convert(y)] for x, y in td_path]
        sample_idx.append(tds[:max_depth] + [[0, 0]]
                          * abs(max_depth - len(tds)))
    return torch.tensor(sample_idx)


if __name__ == '__main__':
    root_path = [[], [1, 2, 3], [2, 3, 4]]
    print(generate_positions(root_path, 4, 3))
    # tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 0
    #         [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.], 3 2 1
    #         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0.]]) 3 3 2

    td_paths = [[], [[0, 1]], [[0, 1], [1, 2]], [[0, 3], [1, 2], [2, 3]]]
    print(generate_TD_position(td_paths, 2, 4))

    relations = [{"1": [0, 3, 0], "2": [1, 2, 0]}, {"0": [0, 3, 1], }, {"0": [0, 3, 1], "2": [1, 2, 0]}]
    print(generate_local_relations(relations, 2))
