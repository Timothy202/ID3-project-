import math

import argparse as argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import deque


def val_entropy(input_column):
    val1, val_counts = np.unique(input_column, return_counts=True)
    summ = len(input_column)
    entropy = 0
    if len(val_counts) == 1:
        return entropy
    for count in val_counts:
        if count > 0:
            entropy += (count / summ) * math.log(summ / count, 2)
    return entropy


def info_gain(df: pd.DataFrame, features, label_column_name, original_entropy):
    attr_column = df[features]
    new_entropy = 0

    for value in attr_column.unique():
        pdf = df[df[features] == value]
        label_column = pdf[label_column_name]
        split_entropy = val_entropy(label_column)
        new_entropy += len(pdf) / len(df) * split_entropy

    return original_entropy - new_entropy


def part_str(best_partition, best_features, new_partition_name_queue):
    if len(new_partition_name_queue) == 1:
        raise ValueError("")

    elif len(new_partition_name_queue) == 2:

        return f"partition {best_partition} was replaced with partition {new_partition_name_queue[0]}" \
               f' and {new_partition_name_queue[-1]} using feature {best_features}'

    else:
        partition_names = ', '.join(new_partition_name_queue)
        partition_names = ', '.join(partition_names.rsplit(', ', 1))
        return f'Partition {best_partition} was replaced with partitions {partition_names} using ' \
               f'feature {best_features}'


def newpartition(best_partition, new_partitions, existing_partitions):
    new_partition_names = list()
    num = 1
    #from pydeck.types import string
    new_name = best_partition + str(num)
    for _ in range(len(new_partitions)):
        while new_name in existing_partitions:
            num += 1
            new_name = best_partition + str(num)
        existing_partitions.add(new_name)
        new_partition_names += [new_name]

    return new_partition_names


def calc_best_partition(df, partition):
    df_columns = list(df.columns)
    label_column = df_columns[-1]

    best_partition, best_features, best_f_value = None, None, -1
    splitt_features = [col_name for col_name in df_columns if col_name != label_column]
    for name, elements in partition.items():
        pdf = df.loc[elements]
        print(pdf)
        initial_entropy = val_entropy(pdf[label_column])
        print(name, initial_entropy)
        if initial_entropy == 0:
            continue
        else:
            for features in splitt_features:
                gain = info_gain(pdf, features, label_column, initial_entropy)
                print(f"Gain from splitting partition {name} on {features} is {gain}")
                f_value = gain * len(elements) / len(df)

                if f_value > best_f_value:
                    best_partition, best_features, best_f_value = name, features, f_value


    if best_f_value == -1:
        print('')
    else:
        attr_values, attr_counts = np.unique(splitt_features, return_counts=True)

        pdf = df.loc[partition[best_partition]]
        part_set = set(partition.keys())
        new_partitions = pdf[best_features].unique()
        new_partition_names = newpartition(best_partition, new_partitions, part_set)

        if len(new_partition_names) == 1:
            print('No part change ')
            return partition
        print(part_str(best_partition, best_features, new_partition_names))
        del partition[best_partition]
        for value in new_partitions:
            subpartition = pdf[pdf[best_features] == value]
            new_name = new_partition_names.pop(0)
            partition[new_name] = list(subpartition.index)
    return partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='test.csv', help="Input csv file name of dataset")
    parser.add_argument("--input", type=str, default='partition-2.txt', help="Input partition")
    parser.add_argument("--output", type=str, default='partition-5.txt', help="Output partition")
    args = parser.parse_args()
    dataset = args.dataset
    input_partition_filename = args.input
    output_partition_filename = args.output

    # Read input dataset and partition
    df = pd.read_csv(dataset, skipinitialspace=True)
    df.index += 1
    print(df.head(2))
    outfile = open(output_partition_filename, "w")
    partition = {}
    with open(input_partition_filename) as f:
        for line in f.readlines():
            word = line.split(' ')
            key = word[0]
            value = []
            for i in range(1, len(word)):
                value.append(int(word[i]))
            partition[key] = value

    result_partition = calc_best_partition(df, partition)
    for key, value in result_partition.items():
        value_str = ''
        for val in value:
            value_str = value_str + ' ' + str(val)
        line = str(key) + ' ' + value_str + '\n'
        outfile.writelines(line)
    outfile.close()


main()
