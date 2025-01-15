###########################################################################
#                                                                         #
#           This file implements two ways to improve convergence,         #
#           including Attribute Appending and Separate&Join.              #
#                                                                         #
###########################################################################

import itertools
import logging
import copy

import networkx as nx
import numpy as np

def sep_graph(logger, domain, marginals, iterate_marginals, enable=True):
    '''
    
    Separate&Join
    
    '''
    def _find_keys_set(keys):
        keys_set = set()

        for key in keys:
            keys_set.update(key)

        return tuple(keys_set)
    
    logger.info("separating graph")
    
    #constructing graph
    graph = _construct_graph(domain, marginals)

    #finding separate graphs
    for m in marginals:
        graph.add_edges_from(itertools.combinations(m, 2))
        
        iterate_keys = {}

        if enable is False:
            keys = []

            for marginal in marginals:
                if marginal in iterate_marginals:
                    keys.append(marginal)

            iterate_keys[_find_keys_set(marginals)] = keys

        else:
            for component in nx.connected_components(graph):
                keys = []

                for marginal in marginals:
                    if set(marginal) < component and marginal in iterate_marginals:
                        keys.append(marginal)

                iterate_keys[_find_keys_set(keys)] = keys

    return iterate_keys

def clip_graph(logger, domain, marginals, enable=True):
    '''
    
    find nodes with 1 degree, clip them for attribute appending
    
    '''
    
    clip_layers = []
    
    #constructing graph
    graph = _construct_graph(domain, marginals)
    
    logger.info("clipping graph for appending")

    clip_marginals = copy.deepcopy(marginals)

    while enable:
        layer = ClipLayer()
        num_nodes = 0

        for node, degree in graph.degree():
            if degree == 1:
                neighbor = next(graph.neighbors(node))

                layer.attrs.add(node)
                layer.attrs_ancestor[node] = neighbor

                if (node, neighbor) in marginals:
                    layer.attrs_marginal[node] = (node, neighbor)
                else:
                    layer.attrs_marginal[node] = (neighbor, node)

                num_nodes += 1

        if num_nodes != 0:
            clip_layers.append(layer)
            isolated_attr = []

            for attr in layer.attrs_marginal:
                try:
                    graph.remove_edge(layer.attrs_marginal[attr][0], layer.attrs_marginal[attr][1])
                    clip_marginals.remove(layer.attrs_marginal[attr])
                except:
                    logger.info("isolated attr: %s" % (attr,))
                    isolated_attr.append(attr)

            for attr in isolated_attr:
                layer.attrs.remove(attr)
                layer.attrs_ancestor.pop(attr)
                layer.attrs_marginal.pop(attr)
                clip_marginals.append((attr,))
        else:
            break

    logger.info("totally %s layers" % (len(clip_layers)))

    return clip_marginals, clip_layers

def append_attrs(logger, domain, clip_layers, df, margs):
        for index, layer in enumerate(clip_layers[::-1]):
            logger.info("appending %s layer" % (index,))

            for append_attr in layer.attrs:
                anchor_attr = layer.attrs_ancestor[append_attr]
                anchor_record = np.copy(df[anchor_attr])
                unique_value = np.unique(anchor_record)
                append_record = np.zeros(anchor_record.size, dtype=np.uint32)

                marginal = margs[layer.attrs_marginal[append_attr]].calculate_count_matrix()

                for value in unique_value:
                    indices = np.where(anchor_record == value)[0]

                    if domain.attr_index_mapping[anchor_attr] < domain.attr_index_mapping[append_attr]:
                        if np.sum(marginal[value, :]) != 0:
                            dist = marginal[value, :] / np.sum(marginal[value, :])
                        else:
                            dist = np.full(marginal.shape[1], 1.0 / marginal.shape[1])
                    else:
                        if np.sum(marginal[:, value]) != 0:
                            dist = marginal[:, value] / np.sum(marginal[:, value])
                        else:
                            dist = np.full(marginal.shape[0], 1.0 / marginal.shape[0])

                    cumsum = np.cumsum(dist)
                    start = 0

                    for i, v in enumerate(cumsum):
                        end = int(round(v * indices.size))
                        append_record[indices[start: end]] = i
                        start = end

                df[append_attr] = append_record
                np.random.shuffle(df.values)

def _construct_graph(domain, marginals):
        graph = nx.Graph()
        graph.add_nodes_from(domain.attrs)

        for m in marginals:
            graph.add_edges_from(itertools.combinations(m, 2))

        return graph

class ClipLayer:
    def __init__(self):
        self.attrs = set()
        self.attrs_ancestor = {}
        self.attrs_marginal = {}
