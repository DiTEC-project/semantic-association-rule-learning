import os

import pandas as pd
import csv
import warnings
import tracemalloc

from datetime import datetime
from dotenv import load_dotenv
from niapy.algorithms.basic import DifferentialEvolution, GeneticAlgorithm, ParticleSwarmOptimization
from niapy.algorithms.modified import SuccessHistoryAdaptiveDifferentialEvolution, SelfAdaptiveDifferentialEvolution

from src.repository.graphdb.node_repository import NodeRepository
from src.preprocessing.base_preprocessing import filter_knowledge_graph_props
from src.util.graph_util import discretize_numerical_attributes
from src.repository.timescaledb.sensor_data_repository import SensorDataRepository
from src.preprocessing.semantic_enrichment import *
from src.algorithm.naive_semrl import NaiveSemRL
from src.algorithm.ts_narm import TSNARM
from src.algorithm.our_ae_based_arm.our_ae_based_arm import OurAEBasedARM
from src.algorithm.arm_ae.armae import ARMAE
from src.util.converter_util import *
from src.util.rule_quality import *

# load environment parameters
load_dotenv(".env")

# todo: resolve the warnings
warnings.filterwarnings("ignore")


def print_params():
    print("----------------------------------------------------")
    print("Algorithm parameters:\n")
    print("NAIVE_SEMRL_MIN_SUPPORT:", os.getenv("NAIVE_SEMRL_MIN_SUPPORT"))
    print("NAIVE_SEMRL_MIN_CONFIDENCE:", os.getenv("NAIVE_SEMRL_MIN_CONFIDENCE"))
    print("(OUR AE-based ARM)SIMILARITY_THRESHOLD:", os.getenv("SIMILARITY_THRESHOLD"))
    print("MAX_ANTECEDENT:", os.getenv("MAX_ANTECEDENT"))
    print("TS_NARM_POPULATION_SIZE:", os.getenv("TS_NARM_POPULATION_SIZE"))
    print("TS_NARM_MAX_EVALUATIONS:", os.getenv("TS_NARM_MAX_EVALUATIONS"))
    print("TRANSACTION_PERIOD_LENGTH_IN_MINUTES:", os.getenv("TRANSACTION_PERIOD_LENGTH_IN_MINUTES"))
    print("NUM_OF_BINS:", os.getenv("NUM_OF_BINS"))
    print("NUM_OF_NEIGHBORS:", os.getenv("NUM_OF_NEIGHBORS"))
    print("----------------------------------------------------\n")


# load parameters
load_dotenv()
min_support = float(os.getenv("NAIVE_SEMRL_MIN_SUPPORT"))
min_confidence = float(os.getenv("NAIVE_SEMRL_MIN_CONFIDENCE"))
similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
max_antecedent = int(os.getenv("MAX_ANTECEDENT"))
population_size = int(os.getenv("TS_NARM_POPULATION_SIZE"))
max_evals = int(os.getenv("TS_NARM_MAX_EVALUATIONS"))
transaction_period = int(os.getenv("TRANSACTION_PERIOD_LENGTH_IN_MINUTES"))
num_bins = int(os.getenv("NUM_OF_BINS"))
num_neighbors = int(os.getenv("NUM_OF_NEIGHBORS"))
num_runs = int(os.getenv("NUM_OF_RUNS"))
dataset = os.getenv("TIMESCALEDB_TABLE")


def save_results(results):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    dataset = os.getenv("TIMESCALEDB_TABLE")
    print("\nRESULTS: Rule quality evaluation results for the dataset", dataset)
    with open(dataset + "_" + timestamp + '.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Dataset: " + dataset])
        columns = ["Algorithm", "Rule Count", "Training Time (s)", "Rule Extraction Time (s)", "Support", "Confidence",
                   "Rule Coverage", "Zhang", "Data Coverage"]
        writer.writerow(columns)
        print(columns)
        for algorithm in results:
            if len(results[algorithm]['stats']) > 0:
                alg_stats = [round(float_value, 2) for float_value in
                             list(pd.DataFrame(results[algorithm]["stats"]).mean())]
                row = [algorithm] + alg_stats
            else:
                row = [algorithm, "No rules found!"]
            print(row)
            writer.writerow(row)
    print("\nSAVED: The results are saved into '", dataset + "_" + timestamp + ".csv' file.")


if __name__ == "__main__":
    print_params()
    node_repository = NodeRepository()

    # sensor data
    sensor_data_repository = SensorDataRepository()

    # create nodes on the KG for sensors, if they don't exist already
    unique_sensor_ids = sensor_data_repository.get_unique_sensor_ids()
    for sensor in unique_sensor_ids:
        node_repository.add_sensor(sensor[0], sensor[1])

    # initialize algorithms
    fp_growth = NaiveSemRL(min_support, min_confidence, num_bins, max_antecedent, "fpgrowth")
    hmine = NaiveSemRL(min_support, min_confidence, num_bins, max_antecedent, "hmine")
    our_ae_based_arm = OurAEBasedARM(num_bins, num_neighbors, max_antecedent, similarity_threshold)
    de = TSNARM(DifferentialEvolution(population_size, differential_weight=0.5, crossover_probability=0.9), max_evals)
    ga = TSNARM(GeneticAlgorithm(population_size, mutation_rate=0.01, crossover_rate=0.8), max_evals)
    pso = TSNARM(ParticleSwarmOptimization(population_size, c1=0.1, c2=0.1, w=0.8), max_evals)
    lshade = TSNARM(SuccessHistoryAdaptiveDifferentialEvolution(population_size), max_evals)
    jde = TSNARM(SelfAdaptiveDifferentialEvolution(population_size, tao1=0.1, crossover_probability=0.9,
                                                   differential_weight=0.5), max_evals)

    stats = {"fpgrowth": {'rules': [], 'stats': []}, "hmine": {'rules': [], 'stats': []},
             "de": {'rules': [], 'stats': []}, "ga": {'rules': [], 'stats': []}, "pso": {'rules': [], 'stats': []},
             "lshade": {'rules': [], 'stats': []}, "jde": {'rules': [], 'stats': []},
             "our_ae_based_arm": {'rules': [], 'stats': []}, "arm_ae": {'rules': [], 'stats': []}}

    # run each of the algorithms "num_runs" time and calculate the average
    for i in range(num_runs):
        print("Number of executions: ", (i + 1), "/", os.getenv("NUM_OF_RUNS"))
        current_iteration_stats = []
        # knowledge graph
        knowledge_graph_neo4j = node_repository.get_all_nodes_with_relations()
        # convert KG to networkx format for ease of processing
        knowledge_graph_networkx = neo4j_to_networkx(knowledge_graph_neo4j)

        # get grouped sensor data by time, and the function also filters sensors due to time and space complexity of the
        # FP-growth-based Naive SemRL algorithm.
        sensor_data = sensor_data_repository.get_grouped_data_by_time(transaction_period, subsample=10)
        # encode sensor data as transactions, by coupling sensor measurements with sensor id and sensor type
        transactions = timeseries_to_transactions(sensor_data)

        # filter the kg properties (to include useful props only),
        # but keep the name as an identifier of the nodes which won't be used in the learning
        knowledge_graph = filter_knowledge_graph_props(knowledge_graph_networkx,
                                                       categorical_attributes + numerical_attributes + ["name"])

        # discretize numerical attributes in the knowledge graph
        knowledge_graph = discretize_numerical_attributes(knowledge_graph, numerical_attributes, num_bins)

        tracemalloc.start()

        # optimization-based ARM
        de_stats, de_rules = de.learn_rules(knowledge_graph, transactions)
        ga_stats, ga_rules = ga.learn_rules(knowledge_graph, transactions)
        pso_stats, pso_rules = pso.learn_rules(knowledge_graph, transactions)
        lshade_stats, lshade_rules = lshade.learn_rules(knowledge_graph, transactions)
        jde_stats, jde_rules = jde.learn_rules(knowledge_graph, transactions)

        if len(de_rules) > 0:
            stats["de"]["stats"].append(de_stats)
            stats["de"]["rules"] = de_rules
        if len(de_rules) > 0:
            stats["ga"]["stats"].append(ga_stats)
            stats["ga"]["rules"] = ga_rules
        if len(de_rules) > 0:
            stats["pso"]["stats"].append(pso_stats)
            stats["pso"]["rules"] = pso_rules
        if len(de_rules) > 0:
            stats["lshade"]["stats"].append(lshade_stats)
            stats["lshade"]["rules"] = lshade_rules
        if len(de_rules) > 0:
            stats["jde"]["stats"].append(jde_stats)
            stats["jde"]["rules"] = jde_rules

        # Naive SemRL with FP-Growth and HMine
        enriched_transactions = enrich_transactions_naivesemrl(knowledge_graph, transactions, num_bins)
        # this line just changes the encoding of the transactions in a way that is easier to deconstruct rules
        # non_enriched_transactions = transactions_without_semantics(transactions, num_bins)

        fpgrowth_rules, fpgrowth_exec_time, fpgrowth_coverage = fp_growth.mine_rules(enriched_transactions)
        if len(fpgrowth_rules) > 0:
            stats["fpgrowth"]["rules"].append(fpgrowth_rules)
            stats["fpgrowth"]["stats"].append(
                evaluate_rules(fpgrowth_rules, fpgrowth_exec_time, 0) + [fpgrowth_coverage])

        hmine_rules, hmine_exec_time, hmine_coverage = hmine.mine_rules(enriched_transactions)
        if len(hmine_rules) > 0:
            stats["hmine"]["rules"].append(hmine_rules)
            stats["hmine"]["stats"].append(evaluate_rules(hmine_rules, hmine_exec_time, 0))

        # Our AE-based ARM approach
        our_ae_based_arm.create_input_vectors(knowledge_graph, transactions)
        our_ae_based_arm.train(dataset)
        our_ae_based_arm_association_rules, ae_exec_time, ae_training_time = our_ae_based_arm.generate_rules()
        our_ae_based_arm_association_rules, ae_coverage = our_ae_based_arm.calculate_stats(
            our_ae_based_arm_association_rules, transactions)
        our_ae_based_arm_association_rules = our_ae_based_arm.reformat_rules(our_ae_based_arm_association_rules)

        if len(our_ae_based_arm_association_rules) > 0:
            stats["our_ae_based_arm"]["rules"].append(our_ae_based_arm_association_rules)
            stats["our_ae_based_arm"]["stats"].append(
                evaluate_rules(our_ae_based_arm_association_rules, ae_exec_time, ae_training_time) + [ae_coverage])

        # ARM-AE from Berteloot et al. (2023)
        input_vectors = enrich_transactions_arm_ae(knowledge_graph, transactions, num_bins, num_neighbors=1)
        arm_ae = ARMAE(len(input_vectors.loc[0]))
        dataLoader = arm_ae.dataPreprocessing(input_vectors)
        arm_ae.train(dataLoader)
        arm_ae.generateRules(input_vectors, numberOfRules=2, nbAntecedent=max_antecedent)
        arm_ae_stats = evaluate_rules(arm_ae.results, arm_ae.exec_time, arm_ae.arm_ae_training_time)
        if len(arm_ae.results) > 0:
            stats["arm_ae"]["stats"].append(
                arm_ae_stats + [round((arm_ae.dataset_coverage.sum()) / len(input_vectors), 2)])
            stats["arm_ae"]["rules"] = arm_ae.results

    save_results(stats)
