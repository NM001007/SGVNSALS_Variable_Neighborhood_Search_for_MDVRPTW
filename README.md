# SGVNSALS_Variable_Neighborhood_Search_for_MDVRPTW
The implementation of the paper "A variable neighborhood search-based algorithm with adaptive local search for the Vehicle Routing Problem with Time Windows and multi-depots aiming for vehicle fleet reduction"

# Abstract
This article addresses the Multi-Depot Vehicle Routing Problem with Time Windows with the minimization of the number of used vehicles, denominated as MDVRPTW*. This problem is a variant of the classical MDVRPTW, which only minimizes the total traveled distance. We developed an algorithm named Smart General Variable Neighborhood Search with Adaptive Local Search (SGVNSALS) to solve this problem, and, for comparison purposes, we also implemented a Smart General Variable Neighborhood Search (SGVNS) and a General Variable Neighborhood Search (GVNS) algorithms. The SGVNSALS algorithm alternates the local search engine between two different strategies. In the first strategy, the Randomized Variable Neighborhood Descent method (RVND) performs the local search, and, when applying this strategy, most successful neighborhoods receive a higher score. In the second strategy, the local search method is applied only in a single neighborhood, chosen by a roulette method. Thus, the application of the first local search strategy serves as a learning method for applying the second strategy. To test these algorithms, we use benchmark instances from MDVRPTW involving up to 960 customers, 12 depots, and 120 vehicles. The results show SGVNSALS performance surpassed both SGVNS and GVNS concerning the number of used vehicles and covered distance. As there are no algorithms in the literature dealing with MDVRPTW*, we compared the results from SGVNSALS with those of the best-known solutions concerning these instances for MDVRPTW, where the objective is only to minimize the total distance covered. The results showed that the proposed algorithm reduced the vehicle fleet by 91.18% of the evaluated instances, and the fleet size achieved an average reduction of up to 23.32%. However, there was an average increase of up to 31.48% in total distance traveled in these instances. Finally, the article evaluated the contribution of each neighborhood to the local search and shaking operations of the algorithm, allowing the identification of the neighborhoods that most contribute to a better exploration of the solution space of the problem.


# Reference
@article{BEZERRA2023106016,
title = {A variable neighborhood search-based algorithm with adaptive local search for the Vehicle Routing Problem with Time Windows and multi-depots aiming for vehicle fleet reduction},
journal = {Computers & Operations Research},
volume = {149},
pages = {106016},
year = {2023},
issn = {0305-0548},
doi = {https://doi.org/10.1016/j.cor.2022.106016},
url = {https://www.sciencedirect.com/science/article/pii/S0305054822002465},
author = {Sinaide Nunes Bezerra and Marcone Jamilson Freitas Souza and Sérgio Ricardo {de Souza}},
keywords = {Multi-depot Vehicle Routing Problem with Time Windows, Variable neighborhood search, Neighborhood, Local search},
}
