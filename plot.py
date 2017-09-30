import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

file_name = 'win_stats'


def main(win_stats, count, epsilon_start, min_epsilon, epsilon_episodes):
    epsilons = np.linspace(epsilon_start, min_epsilon, epsilon_episodes + 1)
    win_stat_count = win_stats.shape[0]
    epsilon_count = epsilons.shape[0]
    if epsilon_count < win_stat_count:
        epsilons = np.append(epsilons, np.full(win_stat_count - epsilon_count,
                min_epsilon))
    win_stat_means = np.mean(win_stats.reshape(-1, count), axis=1)
    epsilon_means = np.mean(epsilons.reshape(-1, count), axis=1)
    episodes = np.arange(count, win_stat_count + 1, count)
    plt.plot(episodes, win_stat_means)
    plt.plot(episodes, epsilon_means)
    plt.axhline(0.5, alpha=0.7, linestyle='dotted', color='grey')
    plt.legend(['Win rate', r'$\epsilon$', '50 %'])
    plt.xlabel('Episodes')
    plt.ylabel(r'Means of wins and $\epsilon$ values over ' + str(count)
            + ' episodes each')
    plt.title('Win rate during learning')
    plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: plot.py win_stats.npy [epsilon_start=1] '
                '[min_epsilon=0.1] [epsilon_episodes=6000] '
                '[number of wins to average over=100]')
    else:
        win_stats = np.load(sys.argv[1], allow_pickle=False)
        if len(sys.argv) > 4:
          epsilon_start = float(sys.argv[2])
          min_epsilon = float(sys.argv[3])
          epsilon_episodes = int(sys.argv[4])
        else:
          epsilon_start = 1
          min_epsilon = 0.1
          epsilon_episodes = 6000
        if len(sys.argv) > 5:
            count = int(sys.argv[5])
        else:
            count = 100
        main(win_stats, count, epsilon_start, min_epsilon, epsilon_episodes)
