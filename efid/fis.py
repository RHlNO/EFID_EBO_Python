from typing import List
import numpy as np

class FIS_1_In:

    def __init__(self, mf_centers: List[float], rules: List[float]):
        """
        Parameters
        ----------
        mf_centers : iterable of floats
            Iterable of mf array of mf centers for each input
        rules : iterable of  floats
            List of 1 x N mf array of mf rules for each input
        """

        self.mf_centers = sorted(mf_centers)
        self.rules = rules

    def evaluate(self, input: float):
        # Get membership function weights of each rule based on input
        rule_weights = self.getMembership(input, self.mf_centers)
        # Weighted average output of each rule value
        weightSum = sum([weight*rule for weight, rule in zip(rule_weights, self.rules)])
        return weightSum/sum(rule_weights)

    def getMembership(self, value: float, mf_centers: List[float]):
        mvs = []  # Initialize membership values list
        # For first membership function apply trapezoidal ruling
        mvs.append((value-mf_centers[1]) / (mf_centers[0]-mf_centers[1]))

        # For other membership functions apply triangular ruling
        for idx, cen in enumerate(mf_centers[1:-1]):
            try:
                side = mf_centers[int(idx + np.sign(value-cen) + 1)]
                mvs.append((value-side) / (cen-side))
            except ZeroDivisionError: # Catch zero division error if value is at mf center and give mv of 1
                mvs.append(1)

        # For last membership function apply  trapezoidal ruling
        mvs.append((value - mf_centers[-2]) / (mf_centers[-1] - mf_centers[-2]))

        #  Any membership values less than 0 are 0 and greater than 1 are 1
        mvs = [max(min(mv, 1), 0) for mv  in mvs]

        return mvs
