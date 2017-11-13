import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

file_name = 'win_stats'


def main(durak_stats, epsilon_start, min_epsilon, epsilon_episodes, count):
    epsilons = np.linspace(epsilon_start, min_epsilon, epsilon_episodes + 1)
    durak_stat_count = durak_stats.shape[0]
    epsilon_count = epsilons.shape[0]
    if epsilon_count < durak_stat_count:
        epsilons = np.append(epsilons, np.full(durak_stat_count
                - epsilon_count, min_epsilon))
    ix_count = max(durak_stats) + 1
    durak_stat_means_list = []
    for ix in range(ix_count):
        durak_stat_means = np.ones(durak_stat_count, dtype=np.float)
        durak_stat_means[np.where(durak_stats == ix)] = 0
        durak_stat_means_list.append(
                np.mean(durak_stat_means.reshape(-1, count), axis=1))
    epsilon_means = np.mean(epsilons.reshape(-1, count), axis=1)
    episodes = np.arange(count, durak_stat_count + 1, count)
    win_rate_strings = []
    for ix in range(ix_count):
        plt.plot(episodes, durak_stat_means_list[ix])
        win_rate_strings.append('Win rate for player {0}'.format(ix))
    plt.plot(episodes, epsilon_means)
    plt.axhline(0.5, alpha=0.7, linestyle='dotted', color='grey')
    plt.legend(win_rate_strings + [r'$\epsilon$', '50 %'])
    plt.xlabel('Episodes')
    plt.ylabel('Means of wins over ' + str(count)
            + r' episodes each and $\epsilon$ values')
    plt.title('Win rate during learning')
    plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: plot_onlyais.py durak_stats.npy [epsilon_start=1] '
                '[min_epsilon=0.1] [epsilon_episodes=3000] '
                '[number of wins to average over=100]')
    else:
        durak_stats = np.load(sys.argv[1], allow_pickle=False)
        if len(sys.argv) > 4:
          epsilon_start = float(sys.argv[2])
          min_epsilon = float(sys.argv[3])
          epsilon_episodes = int(sys.argv[4])
        else:
          epsilon_start = 1
          min_epsilon = 0.1
          epsilon_episodes = 3000
        if len(sys.argv) > 5:
            count = int(sys.argv[5])
        else:
            count = 100
        main(durak_stats, epsilon_start, min_epsilon, epsilon_episodes, count)
