import numpy as np
import numpy.random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import reg_cal

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='GBUCB worst cases')

	parser.add_argument('--N', type=int, default=2, metavar='N',
	                    help='')
	parser.add_argument('--times', type=int, default=10, metavar='N',
	                    help='')
	parser.add_argument('--T0', type=int, default=1000,
	                    help='')
	parser.add_argument('--means', type=list, default=[0.3, 0.7], metavar='MEANS',
	                    help='')
	parser.add_argument('--r-list', type=list, default=[1.111, 1.076, 1.058], metavar='MEANS',
	                    help='')
	parser.add_argument('--repeat', type=int, default=500, metavar='LR',
	                    help='')
	parser.add_argument('--repeat-time', type=int, default=30, metavar='N',
	                    help='')
	parser.add_argument('--alpha1', type=float, default=0.9,
	                    help='')

	args = parser.parse_args()

	regret_final1 = []
	regret_final2 = []
	regret_final3 = []
	regret_final4 = []
	regret_final5 = []

	for _ in range(args.repeat_time):
	    r1,r2,r3,r4,r5 = reg_cal(args)
	    regret_final1.append(r1[:-1])
	    regret_final2.append(r2[:-1])
	    regret_final3.append(r3[:-1])
	    regret_final4.append(r4[:-1])
	    regret_final5.append(r5[:-1])
	    

	clrs = sns.color_palette("husl", 5)
	r1 = np.mean(regret_final1, 0)
	r2 = np.mean(regret_final2, 0)
	r3 = np.mean(regret_final3, 0)
	r4 = np.mean(regret_final4, 0)
	r5 = np.mean(regret_final5, 0)

	std1 = np.std(regret_final1, 0)
	std2 = np.std(regret_final2, 0)
	std3 = np.std(regret_final3, 0)
	std4 = np.std(regret_final4, 0)
	std5 = np.std(regret_final5, 0)

	x_list = range(len(r1))
	k = 5
	with sns.axes_style("darkgrid"):
	    plt.plot(x_list, r1, c = clrs[0])
	    plt.fill_between(x_list, r1-std1/k, r1+std1/k ,alpha=0.1, facecolor=clrs[0])

	    plt.plot(x_list, r2, c = clrs[1])
	    plt.fill_between(x_list, r2-std2/k, r2+std2/k ,alpha=0.1, facecolor=clrs[1])

	    plt.plot(x_list, r3, c = clrs[2])
	    plt.fill_between(x_list, r3-std3/k, r3+std3/k ,alpha=0.1, facecolor=clrs[2])

	    plt.plot(x_list, r4, c = clrs[3])
	    plt.fill_between(x_list, r4-std4/k, r4+std4/k ,alpha=0.1, facecolor=clrs[3])

	    plt.plot(x_list, r5, c = clrs[4])
	    plt.fill_between(x_list, r5-std5/k, r5+std5/k ,alpha=0.1, facecolor=clrs[4])
	    plt.xlabel('Time Step')
	    plt.ylabel('Cumulative Regret')
	    plt.title('Comparisons')
	    plt.legend(['EBUCB (c=0)', 'BUCB (c=0)', 'BUCB (c=5)', 'EBUCB (c=5)', 'Thompson Sampling'])
	    plt.savefig('EBUCB_normal.png', dpi=300)

if __name__ == '__main__':
    main()