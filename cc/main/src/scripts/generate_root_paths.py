# reads asts
# generates tree positional embeddings data for the datapoints
# from the paper "Novel positional encodings to enable tree-based transformers"


import argparse
import json
import logging
import os

import sys

sys.setrecursionlimit(10000)

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils.utils import separate_dps, file_tqdm, separate_lrs
from utils.tree_utils import Node, clamp_and_slice_ids

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/paths.txt", help="filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="max context length for each dp"
    )
    parser.add_argument(
        "--max_width", type=int, default=16, help="max number of child ids"
    )
    parser.add_argument(
        "--max_depth", type=int, default=8, help="max depth of the leaf to root path"
    )
    parser.add_argument(
        "--mode", default="all", choices=["all", "values"], help="types and values | only values",
    )

    # HERE
    parser.add_argument("--td", action="store_true", help="Advance Setting, store both child path and father path")
    parser.add_argument("--local_relation", action="store_true", help="")

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Number of context: {}".format(args.n_ctx))

    if args.local_relation:
        print("Save Relation between Child and Father")
        num_dps = 0
        with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
            for line in file_tqdm(f,True):
                dp = json.loads(line.strip())
                if len(dp) <= 1:
                    continue
                try:
                    root = Node.build_tree(dp)
                except RecursionError:
                    print(line)
                    exit(1)
                root.create_local_relation()
                node_list = root.dfs()
                local_relations = Node.extract_data(node_list,
                                                    f=lambda node: node.local_relation)

                lrs = separate_lrs(local_relations, args.n_ctx)

                for lr, extended in lrs:
                    if len(lr) - extended > 1:
                        json.dump(lr, fp=fout)  # each line is the json of a list [dict,dict,...]
                        num_dps += 1
                        fout.write("\n")

    else:
        if args.td:
            print("Save Child Paths and Father Paths")

        num_dps = 0
        with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
            for line in file_tqdm(f,True):
                dp = json.loads(line.strip())
                if len(dp) <= 1:
                    continue
                try:
                    root = Node.build_tree(dp)
                except RecursionError:
                    print(line)
                    exit(1)
                node_list = root.dfs()
                # f=lambda node: clamp_and_slice_ids(
                #     node.child_rel, max_width=args.max_width, max_depth=args.max_depth
                # )
                root_paths = Node.extract_data(
                    node_list,
                    f=lambda node: clamp_and_slice_ids(
                        node.child_rel, max_width=-1, max_depth=-1
                    )
                )
                asts = separate_dps(root_paths, args.n_ctx)
                if args.td:
                    father_paths = Node.extract_data(
                        node_list,
                        f=lambda node: clamp_and_slice_ids(
                            node.father_rel, max_width=-1, max_depth=-1
                        )
                    )
                    assert len(father_paths) == len(root_paths)
                    father_asts = separate_dps(father_paths, args.n_ctx)
                if args.td:
                    for child, father in zip(asts, father_asts):
                        child_ast, extended = child
                        father_ast, _ = father
                        if len(child_ast) - extended > 1:
                            json.dump([child_ast, father_ast],
                                      fp=fout)  # each line is the json of a list [[[1],[1,2,3,4]],[[1],[1,2,3,4]]]
                            # list[0] = childs = [[],[1],[1,2],[1,2,3]]
                            # list[1] = fathers = [[],[1],[1,2],[1,2,3]]
                            num_dps += 1
                            fout.write("\n")
                else:
                    for ast, extended in asts:
                        if len(ast) - extended > 1:
                            json.dump(ast, fp=fout)
                            num_dps += 1
                            fout.write("\n")
        logging.info("Wrote {} data points to: {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
