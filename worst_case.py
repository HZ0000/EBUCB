import numpy as np
import numpy.random as rnd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import scipy.stats as st
from scipy.stats import beta
import argparse
from utils import reg_cal_worst_case


def main():
    # Training settings
	parser = argparse.ArgumentParser(description='BUCB worst cases')

	parser.add_argument('--N', type=int, default=2, metavar='N',
	                    help='')
	parser.add_argument('--times', type=int, default=10, metavar='N',
	                    help='')
	parser.add_argument('--T0', type=int, default=500,
	                    help='')
	parser.add_argument('--means', type=list, default=[0.3, 0.7], metavar='MEANS',
	                    help='')
	parser.add_argument('--r-list', type=list, default=[1.011, 1.005, 1.003],
	                    help='worst case distribution')
	parser.add_argument('--repeat-time', type=int, default=1, metavar='N',
	                    help='')

	args = parser.parse_args()


	i = 0
	result_list = []
	for r in args.r_list:
	    regret_final5 = []
	    for _ in range(args.repeat_time):
	        r5 = reg_cal_worst_case(args, r)
	        regret_final5.append(r5[:-1])

	    # present the results
	    clrs = sns.color_palette("husl", 5)
	    r5 = np.mean(regret_final5, 0)
	    std5 = np.std(regret_final5, 0)
	    result_list.append(regret_final5)

	    x_list = range(len(r5))
	    k = 5
	    with sns.axes_style("darkgrid"):

	        plt.plot(x_list, r5, c = clrs[i])
	        plt.fill_between(x_list, r5-std5/k, r5+std5/k ,alpha=0.1, facecolor=clrs[i])
	    i += 1
	plt.xlabel('Time Step')
	plt.ylabel('Cumulative Regret')
	plt.title('BUCB Worst Case')
	plt.legend(['r = ' + str(r) for r in r_list])
	plt.savefig( 'BUCB_worst_case.png', dpi=300)

if __name__ == '__main__':
    main()
