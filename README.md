# NeighborKAN
This repository provides a reference implementation of the core network architecture and key functional components of the proposed NeighborKAN-DIC framework.

.
├── network.py        # KAN-inspired network blocks (KANLayer / KAN_MLP)
└── function.py       # losses + neighbor input construction utilities

The released code focuses on:
- the KAN-inspired nonlinear mapping network,
- the neighbor-aware input construction,
- and representative loss formulations.

It is intended to support understanding and verification of the model structure
described in the paper, rather than to serve as a complete, reproducible training pipeline.

To protect intellectual property during the review stage, the full training procedure,
data preprocessing, and optimization strategies are not included in this release.
A complete implementation may be released after publication.
