# Knowledge Modeling and Reasoning Systems for DTs

## Importing Water Network Metadata

Is there any tool/software that supports semantic technologies when using water network data?

- epanettools, https://github.com/asselapathirana/epanettools: no longer maintained since 2016
- WNTR, https://github.com/USEPA/WNTR: "An EPANET compatible python package to simulate and analyze water distribution
  networks under disaster scenarios.". Supports exporting a network to json. No semantics-related functionality.
- EPYNET: An Object-Oriented wrapper around EPANET 2.1. Most related features: "Object oriented access to Nodes,
  Junctions, Reservoirs, Tanks, Links, Pipes, Valves and Pumps" and support for time-series. No support for semantic
  technologies
- Open Water Analytics, http://wateranalytics.org/: A water network analytics community built around EPANET and SWMM.

Properties of the .inp file:

- version
- comment
- name
- options
- curves
- patterns
- nodes
- links
- sources
- controls

Standalone nodes:

- version
- comment
- name
- options

Was always empty:

- sources
- controls
- curves

To be imported:

- Nodes->Links
- Patterns->Nodes (Patterns aren't stored in the graph, there are e.g., 60k+ pattern entries in the Oosterbeek file)
- Coordinates->Junctions (already done by the wntr package)

## Importing Sensor Data

There is no direct given connection between the sensor data and the metadata. In the sensor data files, name of the
node that the data belongs to given as a comment. ID of the sensors are detected manually and matched with the metadata.
