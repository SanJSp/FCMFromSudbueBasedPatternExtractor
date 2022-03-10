import copy
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
import re
import Subdue.src.Parameters as Parameters
import Subdue.src.Subdue as Subdue
import Subdue.src.Graph as Graph
import argparse


# implement to avoid mining with prom
def mine_petri_net(log, miner):
    net, initial_marking, final_marking = miner.apply(log)
    return net, initial_marking, final_marking


def import_xes(file_path):
    return pm4py.read_xes(file_path)


def recursive_loop_removal(graph, vertex, path=[], back_tracking_path=[], coloring={}):
    # based on dfs, removes loops (backwards edge in back_tracking_path) and updates adjacency list of node.
    # Coloring used to be sure that all nodes have been worked on => node in coloring == white, 1==grey, 2==black

    path.append(str(vertex[0]))
    coloring[str(vertex[0])] = "1"  # grey
    back_tracking_path.append(str(vertex[0]))

    for neighbour in list(set(graph[str(vertex[0])])):
        if neighbour in back_tracking_path or (type(neighbour) is not str and (neighbour.label) in back_tracking_path):
            # remove vertex from adjacency list of predecessor
            old_neighbours = graph[str(vertex[0])]
            del old_neighbours[old_neighbours.index(neighbour)]
            new_neighbours = old_neighbours
            graph[str(vertex[0])] = new_neighbours
            print("$*+***$$***+*$ Removed egde from dag", f'{vertex[0]}->{neighbour.label}')

        # this is only because manually inserted first node is of type string and has no label
        label = neighbour.label if type(neighbour) is not str else neighbour

        if label not in coloring:
            neighbour_vertex = [label, graph[str(neighbour)]]
            path = recursive_loop_removal(graph, neighbour_vertex, path, back_tracking_path, coloring)[0]

    back_tracking_path.remove(str(vertex[0]))
    coloring[str(vertex[0])] = "2"  # black

    return path, graph


def make_graph_to_dag(graph, first_node):
    dag = graph.copy()

    # vertex represented as (predecessor, neighbour_list)
    first_vertex = (first_node, (graph[str(first_node)]))
    dag = recursive_loop_removal(dag, first_vertex)[1]
    return dag


def find_adjacent_transitions_for_single_transition(transition):
    list_of_adjacent_transitions = []

    list_of_adjacent_transitions = recursively_scrape_neighbours_of_transition(list_of_adjacent_transitions, transition)
    if list_of_adjacent_transitions is None:
        return []

    # removes duplicates
    return list(set(list_of_adjacent_transitions))


def recursively_scrape_neighbours_of_transition(list_of_adjacent_transitions, transition):
    # for a transition, find its adjacent named transitions. For each adjacent transition, check if it is named. If so,
    # append it to adjacency list, else dig deeper in the nameless transitions neighbours.
    current_neighbours = []

    outgoing_arcs_of_transition = transition.out_arcs

    # for each outgoing arc of adjacent places of transition
    for arc in outgoing_arcs_of_transition:
        targeted_place = arc.target
        outgoing_arcs_from_place = targeted_place.out_arcs
        if len(outgoing_arcs_from_place) == 0:
            return
        # for each adjacent transition of place
        for outgoing_arc in outgoing_arcs_from_place:
            two_step_transition = outgoing_arc.target
            if two_step_transition.label is not None and is_transition_nameless(two_step_transition.label) is False:
                current_neighbours.append(two_step_transition)
            else:
                further_down_neighbours = recursively_scrape_neighbours_of_transition(list_of_adjacent_transitions,
                                                                                      two_step_transition)
                if further_down_neighbours is not None:
                    for neighbour in further_down_neighbours:
                        current_neighbours.append(neighbour)

    return current_neighbours


def construct_graph(net):
    # for each transition in petri net, make each named transition adjacent that can be reached by only traversing
    # nameless transitions
    adjacency_list_of_transitions = {}

    for transition in net.transitions:
        if transition.label is None:
            transition.label = str(transition)

        if transition.label is not None and is_transition_nameless(transition.label) is False:
            adjacent_transitions = find_adjacent_transitions_for_single_transition(transition)
            adjacency_list_of_transitions[transition.label] = adjacent_transitions

    return adjacency_list_of_transitions


def is_transition_nameless(label):
    # nameless transitions are not in the event log, also called silent transitions
    x = re.search("n\d+", label)
    if x is None:
        return False
    return True


def build_instance_graphs(adjacency_list_of_transitions, log, pattern_belongings_of_node={}):
    # build the instance graphs by iterating over each trace and in this each activity of a trace. For each activity
    # the neighbours that are in the trace log are added to the instance graph, therefore taking the subgraph of only the
    # trace activities and the edges between them

    all_instance_graphs = []

    for trace in log:
        instance_graph = {}
        trace_event_names = []

        for event in trace:
            # TODO check if concept:name ist standard way for names in xes"
            trace_event_names.append(event["concept:name"])

        trace_event_names = list(set(trace_event_names))

        # add pattern to trace if pattern activity is in trace
        for trace_event_name in trace_event_names:
            if trace_event_name in pattern_belongings_of_node \
                    and f'Pattern-{pattern_belongings_of_node[trace_event_name]}' not in trace_event_names:
                trace_event_names.append(f'Pattern-{pattern_belongings_of_node[trace_event_name]}')

        for event_name in trace_event_names:
            if event_name in adjacency_list_of_transitions:
                neighbours = adjacency_list_of_transitions[event_name]
                copy_of_neighbours = neighbours.copy()

                # remove neighbour if not related to subgraph or included in pattern
                for neighbour in neighbours:
                    if ((neighbour.label if type(neighbour) != str else neighbour) not in trace_event_names) \
                            or ((neighbour.label if type(neighbour) != str else neighbour)
                                in pattern_belongings_of_node):
                        copy_of_neighbours.remove(neighbour)

                # filter self loops
                if event_name in copy_of_neighbours:
                    copy_of_neighbours.remove(event_name)

                instance_graph[event_name] = copy_of_neighbours

        all_instance_graphs.append(instance_graph)

    return all_instance_graphs


def parse_data(instance_graphs):
    # target format: list of instance graphs
    # graph = (list of vertex, list of edges)
    # vertex = {vertex: {id, attributes: {label}, timestamp}
    # edges = [{edge: {id, source, target, directed:"false", attributes: { label}, timestamp}]

    node_id_dictionary = {}
    final_graph = []
    timestamp = 1
    vertex_id = 1
    edge_id = 1
    for instance_graph in instance_graphs:
        upcoming_node_id = vertex_id
        for node in instance_graph:
            if node not in node_id_dictionary.keys():
                node_id_dictionary[node] = upcoming_node_id
                upcoming_node_id += 1

        for node in instance_graph:
            vertex = {"vertex": {
                "id": str(vertex_id),
                "attributes": {"label": node},
                "timestamp": str(timestamp)
            }}
            final_graph.append(vertex)
            vertex_id += 1

        for node in instance_graph:
            adjacency_list_for_node = instance_graph[node]
            for neighbour in adjacency_list_for_node:
                edge = {"edge": {
                    "id": str(edge_id),
                    "source": str(node_id_dictionary[node]),
                    "target": str(node_id_dictionary[(neighbour.label if type(neighbour) != str else neighbour)]),
                    "attributes": {"label": f'{node}->{neighbour}'},
                    "directed": "false",
                    "timestamp": str(timestamp)
                }}
                final_graph.append(edge)
                edge_id += 1

        node_id_dictionary = {}
        timestamp += 1

    return final_graph


def subdue(json_file, args):
    # is not an actual json, just a list containing dicts.
    # Set up graph
    graph = Graph.Graph()
    graph.load_from_json(json_file)

    # set up parameters
    # for flags see Subdue.src.Parameters
    parameters = Parameters.Parameters()
    parameters.writeCompressed = args.writecompressed
    parameters.writePattern = args.writepattern
    parameters.outputFileName = args.outputfile
    parameters.set_defaults_for_graph(graph)
    parameters.numBest = 1
    parameters.limit = 1000 # hard coded, should be v^2
    parameters.maxSize = 1000 # hard coded should be V^2

    # finally, execute subdue
    return Subdue.Subdue(parameters, graph)


def recursive_transition_traversal(place, found_neighbours=[], path=[]):
    # checks for each place if adjacent transitions are "nameless" (=not in event log). If not, add to list of found
    # neighbours of original place. Else go deeper and check adjacent places of place (skip over transition)

    path.append(place.name)
    outgoing_arcs = place.out_arcs

    for outgoing_arc in outgoing_arcs:
        # regex checks if transition is "nameless"
        x = re.search("n\d+", outgoing_arc.target.label)
        if x is not None and outgoing_arc.target.label not in path:
            outgoing_arc_of_transition = outgoing_arc.target.out_arcs
            for outgoing_arc_of_transition in outgoing_arc_of_transition:
                found_neighbours = recursive_transition_traversal(outgoing_arc_of_transition.target, found_neighbours,
                                                                  path)
        else:
            if outgoing_arc.target not in found_neighbours:
                found_neighbours.append(outgoing_arc.target)

    return found_neighbours


def get_start_node(net, graph):
    # get first node in graph or build a new node that is adjazent to all nodes that can be reached from the first
    # place in petri net

    first_place = [place for place in list(net.places) if place.name == "n1"][0]
    start_node_neighbours = recursive_transition_traversal(first_place)

    if len(start_node_neighbours) > 1:
        start_node = "manually_inserted_start_node"
        graph[start_node] = start_node_neighbours
    else:
        start_node = start_node_neighbours[0]

    return graph, start_node


def contract_discovered_nodes(current_pattern_graph, pattern_index, old_dag):
    # contracts pattern to node and returns new DAG

    contracted_node = f'Pattern-{pattern_index}'
    adjacent_nodes_to_pattern = []

    for node in current_pattern_graph:
        node_neighbours = old_dag[node]
        for neighbour in node_neighbours:
            if (str(neighbour) not in current_pattern_graph.keys()):
                adjacent_nodes_to_pattern.append(neighbour)

        del old_dag[node]

        for activity_key in old_dag:
            activity_neighbours = old_dag[activity_key]
            neighbour_labels = []

            for neighbour in activity_neighbours:
                neighbour_labels.append(neighbour.label if type(neighbour) != str else neighbour)

            # remove self from adjacency lists and then add pattern node as new neighbour
            if node in neighbour_labels:
                activity_neighbours.remove([neighbour for neighbour in activity_neighbours if
                                            type(neighbour) is not str and neighbour.label == node or type(
                                                neighbour) is str and neighbour == node][0] if type(
                    neighbour) != str else neighbour)
                if f'Pattern-{pattern_index}' not in activity_neighbours:
                    activity_neighbours.append(f'Pattern-{pattern_index}')
            old_dag[activity_key] = activity_neighbours

    adjacent_nodes_to_pattern = list(set(adjacent_nodes_to_pattern))
    old_dag[contracted_node] = adjacent_nodes_to_pattern
    return old_dag


def reconstruct_pattern_graph(pattern):
    # reconstructs pattern adjacency list from the subdue graph object generated by subdue algorithm

    pattern_graph = {}
    edges = pattern.definition.edges
    vertices = pattern.definition.vertices

    for vertex_key in vertices:
        pattern_graph[vertices[vertex_key].attributes["label"]] = []

    for edge_key in edges:
        edge_source = edges[edge_key].source.attributes["label"]
        edge_target = edges[edge_key].target.attributes["label"]
        pattern_graph[edge_source].append(edge_target)

    return pattern_graph


def construct_pattern_node_edges(found_patterns, pattern_belongings_of_node, inverted_adjacency_list, dag):
    # checks for every found pattern, which nodes in it can be reached from a earlier found patterns and stores node
    # and its incoming edges for each pattern. Additionally it tracks which nodes in previously found patterns can be
    # reached from the current node to save the outgoing edges to higher up patterns

    possible_pattern_to_node_edges = {}
    possible_node_to_pattern_edges = {}
    for pattern in found_patterns:

        # evade first pattern
        if pattern == "Pattern-0":
            continue

        edges_from_pattern_node_to_higher_patterns = {}
        edges_to_pattern_nodes_in_higher_patterns = {}
        for node in found_patterns[pattern]:

            is_node_not_a_pattern = True if re.search("Pattern-\d+", node) is None else False
            if is_node_not_a_pattern:

                # skip nodes that have no incoming edges from other nodes
                inverted_adjacency_list_keys = list(inverted_adjacency_list.keys())
                string_keys = []
                for key in inverted_adjacency_list_keys:
                    string_keys.append(str(key))
                if node not in string_keys:
                    continue

                # get all the neighbours that have an edge towards node
                possible_incoming_neighbours_in_higher_patterns = inverted_adjacency_list[node]

                for possible_incoming_neighbour in possible_incoming_neighbours_in_higher_patterns:
                    # if node is in one of the patterns above the one you are currently in
                    if pattern_belongings_of_node[possible_incoming_neighbour] is not None and \
                            pattern_belongings_of_node[possible_incoming_neighbour] < pattern_belongings_of_node[node]:
                        if node in edges_from_pattern_node_to_higher_patterns.keys() and \
                                possible_incoming_neighbour not in edges_from_pattern_node_to_higher_patterns[node]:
                            edges_from_pattern_node_to_higher_patterns[node].append(possible_incoming_neighbour)
                        else:
                            edges_from_pattern_node_to_higher_patterns[node] = [possible_incoming_neighbour]


                # get all the neighbours that have an edge towards node
                possible_outgoing_neighbours_in_higher_patterns = dag[node]

                for possible_outgoing_neighbour in possible_outgoing_neighbours_in_higher_patterns:
                    # if node is in one of the patterns above the one you are currently in
                    if pattern_belongings_of_node[str(possible_outgoing_neighbour)] is not None and \
                            pattern_belongings_of_node[str(possible_outgoing_neighbour)] < pattern_belongings_of_node[node]:
                        if node in edges_to_pattern_nodes_in_higher_patterns.keys() and \
                                str(possible_outgoing_neighbour) not in edges_to_pattern_nodes_in_higher_patterns[node]:
                            edges_to_pattern_nodes_in_higher_patterns[node].append(possible_outgoing_neighbour.label)
                        else:
                            edges_to_pattern_nodes_in_higher_patterns[node] = [possible_outgoing_neighbour.label]

        possible_pattern_to_node_edges[pattern] = edges_from_pattern_node_to_higher_patterns
        possible_node_to_pattern_edges[pattern] = edges_to_pattern_nodes_in_higher_patterns

    return possible_pattern_to_node_edges, possible_node_to_pattern_edges


def main(args):
    # import log
    log = import_xes('testing/test_workout_log.xes')

    # mine to petri net
    # make sure the miner you want to use has been imported already
    miner = inductive_miner
    # net, initial_marking, final_marking = mine_petri_net(log, miner)
    net, initial_marking, final_marking = pnml_importer.apply(
        "testing/test_workout_log.pnml")

    # construct graph from petri net - removes also transitions that are not on event log (=nameless)
    activity_graph = construct_graph(net)

    # activity_graph = filter_nameless_transitions(adjacency_list_of_transitions)
    activity_graph, start_node = get_start_node(net, activity_graph)

    # removes loops from graph
    dag = make_graph_to_dag(activity_graph, start_node)
    print(dag)

    # prepares input format for subdue using log and dag structure
    instance_graphs = build_instance_graphs(dag, log)
    json_instance_graphs = parse_data(instance_graphs)

    # Construct inverted adjacency list, so for node, who has an edge towards me
    inverted_adjacency_list = {}
    for node in dag:
        for neighbour in dag[node]:
            if neighbour.label in inverted_adjacency_list.keys() and node not in inverted_adjacency_list[neighbour.label]:
                inverted_adjacency_list[neighbour.label].append(node)
            else:
                inverted_adjacency_list[neighbour.label] = [node]

    # find patterns that each activity belongs to
    subdue_can_still_find_patterns = True
    found_patterns = {}
    pattern_belongings_of_node = {}
    dag_with_contractions = copy.deepcopy(dag)

    while subdue_can_still_find_patterns:
        subdue_output = subdue(json_instance_graphs, args)
        if len(subdue_output) == 0:
            subdue_can_still_find_patterns = False
            break

        current_pattern_graph = reconstruct_pattern_graph(subdue_output[0][0])
        pattern_index = len(found_patterns)
        found_patterns[f'Pattern-{pattern_index}'] = current_pattern_graph

        for node in current_pattern_graph:
            pattern_belongings_of_node[node] = pattern_index

        dag_with_contractions = contract_discovered_nodes(current_pattern_graph, pattern_index, dag_with_contractions)
        instance_graphs = build_instance_graphs(dag_with_contractions, log, pattern_belongings_of_node)
        json_instance_graphs = parse_data(instance_graphs)

    # add nodes that have not been added to pattern to belongings
    for node in dag:
        if node not in pattern_belongings_of_node.keys():
            pattern_belongings_of_node[node] = None

    # construct the pattern-node edges for virtual data objects
    possible_pattern_to_node_edges, possible_node_to_pattern_edges = construct_pattern_node_edges(found_patterns, pattern_belongings_of_node, inverted_adjacency_list, dag)

    print(found_patterns)
    print(pattern_belongings_of_node)
    print("Virtual Dataobjects to start a fragment", possible_pattern_to_node_edges)
    print("Virtual Dataobjects to end a fragment", possible_node_to_pattern_edges)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-wc", "--writecompressed", dest="writecompressed",
                        action="store_true", default=False, help="writes compressed graph at each iteration")
    parser.add_argument("-w", "--writepattern", dest="writepattern",
                        action="store_true", default=False, help="writes best pattern at each iteration")
    parser.add_argument("-of", "--outputfile", dest="outputfile",
                        type=str, default=" ", help="file where outputs will be stored, without .json ending")

    args = parser.parse_args()
    main(args)
