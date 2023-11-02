# Semantic Association Rule Learning

This repository consists of a set of association rule learning approaches with semantic information [2] (only 1 as of
November 2023). The use case which the algorithms are tested on is digital twin of water distribution networks.

## Modules

- data: contains partial LeakDB dataset [1]
- graphdb: contains a [Neo4j](https://neo4j.com/) implementation together with python scripts that is used to create
  a knowledge graph from LeakDB dataset
- timescaledb: a [TimescaleDB](https://www.timescale.com/) implementation together with a python script that stores
  LeaKDB
  sensor data in the database
- semantic_rule_learning: a python application that creates semantic association rules
  from a given transaction dataset and knowledge graph

### References:

1. Vrachimis, Stelios G., and Marios S. Kyriakou. "LeakDB: a Benchmark Dataset for Leakage Diagnosis in Water
   Distribution Networks:(146)." WDSA/CCWI Joint Conference Proceedings. Vol. 1. 2018.
2. Karabulut, Erkan, Victoria Degeler, and Paul Groth. "Semantic Association Rule Learning from Time Series Data and
   Knowledge Graphs." arXiv preprint arXiv:2310.07348 (2023).

### Contact

Please send an email to the following address for your feedback and questions: e.karabulut@uva.nl