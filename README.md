# ODE-Flow: Fast and lean encrypted Internet traffic classification

Identifying the type of a network flow or a specific application has many advantages but becomes harder in recent years due to the use of encryption, e.g., by VPN. As a result, there is a recent wave of solutions that harness deep learning for traffic classification. These solutions either require a rather long time (15–60 Seconds) of flow data or rely on handcrafted features for solutions that classify flows faster. In this work, we suggest a novel approach for classification that extracts the most out of the two simple yet defining features of a flow: packet sizes and inter-arrival times. We employ a model that uses the inter-arrival times to parameterize the derivative of the flow hidden-state using a neural network (Neural ODE). We compare our results with a solution that uses the same data without the ODE solver and show the benefit of this approach. Our results can classify flows based on 20 or 30 consecutive packets taken from anywhere in one direction of a flow. This reduces the amount of traffic between the sampling point and the analyzer and does not require matching between two directions of the flow. As a result, our solution can classify traffic with good accuracy within a few seconds, and we show how to combine it with a more accurate (and a slower) classifier to achieve (mostly) fast and accurate classifications.

<br/><br/><img src='http://talshapira.github.io/files/LSTM_ODE_arch.png'>


# Code

Can be found in this [repository](https://github.com/talshapira/ODE-Flow).

# Dataset

We use the processed dataset from [FlowPic](https://talshapira.github.io/portfolio/flowpic) which is absed on the labeled datasets of packet capture (pcap) files from the Uni. of New Brunswick (UNB): ["ISCX VPN-nonVPN traffic dataset" (ISCX-VPN)](https://www.unb.ca/cic/datasets/vpn.html) and ["ISCX Tor-nonTor dataset" (ISCX-Tor)](https://www.unb.ca/cic/datasets/tor.html), as well as our own small packet capture (TAU).

# TrafficParser

Contains the code use to generate the dataset (npy files) per experiment.
If you choose to use our proceesed dataset (i.e. CSV files) directly, run the scripts in the following order:
1. Run traffic_csv_converter_ode.py
2. Run datasets_generator_ode.py

The other two scripts (generic_parser.py + traffic_csv_merger.py) used to generate the proceesed dataset.

# ode_const_20_multiclass_reg

Contains a Google-Colab-ready jupyter notebook used to run an experiment - "ode_const_20_multiclass_reg" - training over a dataset of 20 packets per sample, regular traffic (non-VPN) for the multiclass problem.
The notebook is provided along with the numpy files used to train and evaluate the sepcific experiment.

# Cite

* S. Roy, T. Shapira and Y. Shavitt, "Fast and lean encrypted Internet traffic classification," in Computer Communications, Volume 186, 2022, Pages 166-173, ISSN 0140-3664.

[Download paper here](https://www.sciencedirect.com/science/article/pii/S0140366422000408?via%3Dihub)
