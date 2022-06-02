# EFID_EBO - Evolutionary Fuzzy Influence Diagrams for Effects-Based-Operations
Work in progress repository for my master's thesis research on using fuzzy logic and cognitive processing with knowledge graphs to develop AI solutions to effects-based-operations planning environments. The system will use a genetic algorithm based solver to answer source-determination queries to a self-designed fuzzy inluence diagram. Further information on the foundation of this work is available on request via the contact form on my personal website [zphill.com](https://www.zphill.com/contact)

*genalg* is a package I've built that implements a basic genetic algorithm for use in the source-determination solver component of my research.

*efid* is a work in progress package I'm creating to implement nodal fuzzy inference systems for influence diagrams as part of my research. Basic 1 input fuzzy inferencing is complete at this time.

*MAIN_EqApprox_Test.py* is a basic runnable test case that I've made for testing the efficiency of my genetic algorithm and fuzzy inference system, and it also serves as a tool for debugging. It trains a basic 1 input (X), 1 output (Y) fuzzy inference system to approximate a sin wave using sum-squared-error as fitness.

#### REQUIRED PACKAGES: numpy, matplotlib
