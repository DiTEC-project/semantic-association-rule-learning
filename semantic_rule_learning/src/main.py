from dotenv import load_dotenv

from src.repository.graphdb.node_repository import NodeRepository
from src.preprocessing.semantic_enrichment import *
from src.preprocessing.base_preprocessing import *
from src.algorithm.naive_semrl import NaiveSemRL
from src.algorithm.hho import HHO
from src.algorithm.ae_semrl import AESemRL
from src.algorithm.ae_semrl_benchmark import *
import pandas as pd

# list of configurable parameters for the 3 algorithms implemented
FP_GROWTH_MIN_SUPPORT = 0.3
FP_GROWTH_MIN_CONFIDENCE = 0.8
AE_SEMRL_SIMILARITY_THRESHOLD = 0.5
AE_SEMRL_MAX_ANTECEDENT = 1
AE_SEMRL_NOISE_FACTOR = 0.5
HHO_POPULATION_SIZE = 40
HHO_MAX_ITERATION = 50
TRANSACTION_PERIOD_LENGTH_IN_MINUTES = 1440
NUM_OF_BINS = 10  # to discretize numerical data into
NUM_OF_NEIGHBORS = 0
NUM_OF_RUNS = 2


def print_params():
    print("----------------------------------------------------")
    print("Algorithm parameters:\n")
    print("FP_GROWTH_MIN_SUPPORT:", FP_GROWTH_MIN_SUPPORT)
    print("FP_GROWTH_MIN_CONFIDENCE:", FP_GROWTH_MIN_CONFIDENCE)
    print("AE_SEMRL_SIMILARITY_THRESHOLD:", AE_SEMRL_SIMILARITY_THRESHOLD)
    print("AE_SEMRL_MAX_ANTECEDENT:", AE_SEMRL_MAX_ANTECEDENT)
    print("AE_SEMRL_NOISE_FACTOR:", AE_SEMRL_NOISE_FACTOR)
    print("HHO_POPULATION_SIZE:", HHO_POPULATION_SIZE)
    print("HHO_MAX_ITERATION:", HHO_MAX_ITERATION)
    print("TRANSACTION_PERIOD_LENGTH_IN_MINUTES:", TRANSACTION_PERIOD_LENGTH_IN_MINUTES)
    print("NUM_OF_BINS:", NUM_OF_BINS)
    print("NUM_OF_NEIGHBORS:", NUM_OF_NEIGHBORS)
    print("----------------------------------------------------\n")


load_dotenv()
if __name__ == "__main__":
    print_params()
    node_repository = NodeRepository()

    # sensor data
    sensor_data_repository = SensorDataRepository()

    # create nodes on the KG for sensors, if they don't exist already
    unique_sensor_ids = sensor_data_repository.get_unique_sensor_ids()
    for sensor in unique_sensor_ids:
        node_repository.add_sensor(sensor[0], sensor[1])

    # init algorithms
    naive_semrl = NaiveSemRL(min_support=FP_GROWTH_MIN_SUPPORT, min_confidence=FP_GROWTH_MIN_CONFIDENCE,
                             num_bins=NUM_OF_BINS, max_antecedent=AE_SEMRL_MAX_ANTECEDENT)
    hho = HHO(population_size=HHO_POPULATION_SIZE, max_iterations=HHO_MAX_ITERATION)
    ae_semrl = AESemRL(num_bins=NUM_OF_BINS, num_neighbors=NUM_OF_NEIGHBORS, noise_factor=AE_SEMRL_NOISE_FACTOR,
                       max_antecedents=AE_SEMRL_MAX_ANTECEDENT, similarity_threshold=AE_SEMRL_SIMILARITY_THRESHOLD)

    stats = []
    for i in range(NUM_OF_RUNS):
        print("Number of executions: ", (i + 1), "/", NUM_OF_RUNS)
        current_iteration_stats = []
        # knowledge graph
        knowledge_graph_neo4j = node_repository.get_all_nodes_with_relations()
        # convert KG to networkx format for ease of processing
        knowledge_graph_networkx = neo4j_to_networkx(knowledge_graph_neo4j)

        # get grouped sensor data by time, and the function also filters sensors due to time and space complexity of the
        # FP-growth-based Naive SemRL algorithm.
        sensor_data = sensor_data_repository.get_grouped_data_by_time(TRANSACTION_PERIOD_LENGTH_IN_MINUTES)
        transactions = timeseries_to_transactions(sensor_data)

        # filter the kg properties, but keep the name as an identifier of the nodes which won't be used in the learning
        knowledge_graph = filter_knowledge_graph_props(knowledge_graph_networkx,
                                                       categorical_attributes + numerical_attributes + ["name"])

        hho_rules, hho_exec_time = hho.learn_rules(knowledge_graph, transactions)
        hho_support, hho_confidence, hho_lift, hho_zhangs, hho_leverage = hho_rules.mean("support"), \
            hho_rules.mean("confidence"), hho_rules.mean("lift"), hho_rules.mean("zhang"), hho_rules.mean(
            "leverage")
        current_iteration_stats += [len(hho_rules), hho_exec_time, hho_support, hho_confidence, hho_lift, hho_zhangs,
                                    hho_leverage]

        # discretize numerical attributes in the knowledge graph
        knowledge_graph = discretize_numerical_attributes(knowledge_graph, numerical_attributes, num_bins=NUM_OF_BINS)

        # FP-Growth-based Naive SemRL
        fpgrowth_rules, fpgrowth_exec_time = naive_semrl.fp_growth(knowledge_graph, transactions)
        fpgrowth_support, fpgrowth_confidence, fpgrowth_lift, fpgrowth_leverage, fpgrowth_zhangs = \
            evaluate_rules(fpgrowth_rules)
        current_iteration_stats += [len(fpgrowth_rules), fpgrowth_exec_time, fpgrowth_support, fpgrowth_confidence,
                                    fpgrowth_lift, fpgrowth_zhangs, fpgrowth_leverage]

        # Autoencoder-based SemRL, AE SemRL
        ae_semrl.create_input_vectors(knowledge_graph, transactions)
        ae_semrl.train()
        ae_semrl_association_rules, ae_exec_time = ae_semrl.generate_rules()
        ae_semrl_association_rules = ae_semrl.calculate_stats(ae_semrl_association_rules, transactions)
        ae_semrl_association_rules = ae_semrl.reformat_rules(ae_semrl_association_rules)

        ae_support, ae_confidence, ae_lift, ae_leverage, ae_zhangs = evaluate_rules(ae_semrl_association_rules)
        current_iteration_stats += [len(ae_semrl_association_rules), ae_exec_time, ae_support, ae_confidence, ae_lift,
                                    ae_leverage, ae_zhangs]

        coverage, false_positive_count = get_coverage_false_positives(fpgrowth_rules, ae_semrl_association_rules)
        current_iteration_stats += [coverage, false_positive_count]

        stats.append(current_iteration_stats)

    stats = pd.DataFrame(stats).mean()

    print("HHO (number of rules, execution time, support, confidence, lift, leverage, zhang): "
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" % (stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6]))
    print("FP-Growth (number of rules, execution time, support, confidence, lift, leverage, zhang): "
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %
          (stats[7], stats[8], stats[9], stats[10], stats[11], stats[12], stats[13]))
    print("AE-SemRL (number of rules, execution time, support, confidence, lift, leverage, zhang): "
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %
          (stats[14], stats[15], stats[16], stats[17], stats[18], stats[19], stats[20]))
    print("AE-SemRL rule coverage on FP-Growth:", stats[21])
    print("AE-SemRL false positives based on FP-Growth:", stats[22])
