"""This file is dedicated to to generate simulation doc page plots.
"""
from os.path import join

import matplotlib.pyplot as plt

import simulation_exp3
import simulation_exp4p
import simulation_linthompsamp
import simulation_linucb
import simulation_ucb1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate features.')
    parser.add_argument('-p', '--path', default="",
                        help="the path of simulation result plots to save")
    args = parser.parse_args()

    for sim in [simulation_exp3, simulation_exp4p, simulation_linthompsamp,
                simulation_linucb, simulation_ucb1]:
        sim.simulate_bandit()
        plt.savefig(join(args.path, "%s" % sim.__name__))
        plt.clf()

if __name__ == "__main__":
    main()
