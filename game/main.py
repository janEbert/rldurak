import deck, player as player_m, field, game as game_m
import threading, queue
from random import choice
import numpy as np

# name 'Kraudia' is required to find the agent
names = ['Kraudia', 'Bob', 'Alice']
deck_size = 52
hand_size = 6

iterations = 1000
# how often random bots check
# calculated from a normal distribution with the given values
epsilon_mu = 0.2
epsilon_sigma = 0.15


def main():
    """Main loop for Durak for receiving and executing actions and
    giving rewards."""

    global durak_ix, game, epsilon, threads, action_queue
    event = threading.Event()
    for n in range(iterations):
        epsilon = max(0.07, np.random.normal(epsilon_mu, epsilon_sigma))
        game = game_m.Game(names, deck_size, hand_size)
        if hand_size == 6:
            # reshuffle if a player has more than five cards of the
            # same suit (except trump) in their hand
            for player in game.players:
                counts = [0] * 4
                for card in player.cards:
                    counts[card.num_suit] += 1
                if max(counts) >= 5 and counts[game.deck.num_trump_suit] < 5:
                    game = game_m.Game(names, deck_size, hand_size)
                    break
            while max(counts) >= 5 and counts[game.deck.num_trump_suit] < 5:
                for player in game.players:
                    counts = [0] * 4
                    for card in player.cards:
                        counts[card.num_suit] += 1
                    if (max(counts) >= 5
                            and counts[game.deck.num_trump_suit] < 5):
                        game = game_m.Game(names, deck_size, hand_size)
                        break
        if durak_ix < 0:
            beginner_ix, beginner_card = game.find_beginner()
            if beginner_card == game.deck.bottom_trump:
                print('Beginner was chosen randomly')
        else:
            game.defender_ix = durak_ix
        while not game.ended():
            active_player_indices = game.active_player_indices()
            for ix in active_player_indices:
                print(ix)
                deck.print_cards(game.players[ix].cards, True)
                thread = ActionReceiver(ix, event)
                thread.start()
                threads.append(thread)
            first_attacker_ix = active_player_indices[0]
            while not game.attack_ended():
                # TODO reward if player_ix == game.kraudia_ix
                player_ix, action = action_queue.get()
                if action[0] == 0:
                    print('try', player_ix, action)
                    game.attack(player_ix, [make_card(action)])
                elif action[0] == 1:
                    to_defend, card = make_card(action)
                    game.defend(to_defend, card)
                elif action[0] == 2:
                    for ix in active_player_indices:
                        game.check(ix)
                    while not action_queue.empty():
                        action_queue.get()
                        action_queue.task_done()
                    for ix in active_player_indices:
                        threads[ix].join()
                        game.uncheck(ix)
                    threads.clear()
                    active_player_indices = game.active_player_indices()
                    for ix in active_player_indices:
                        thread = threading.Thread(target=receive_action, args=(ix,))
                        thread.start()
                        threads.append(thread)
                    game.push([make_card(action)])
                elif action[0] == 3:
                    game.check(player_ix)
                action_queue.task_done()
            # attack ended
            print('attack ended')
            event.set()
            for ix in active_player_indices:
                game.check(ix)
            while not action_queue.empty():
                action_queue.get()
                action_queue.task_done()
            for ix in active_player_indices:
                threads[ix].join()
                game.uncheck(ix)
            threads.clear()
            while first_attacker_ix != game.defender_ix:
                # first attacker till last attacker
                game.draw(first_attacker_ix)
                first_attacker_ix += 1
                if first_attacker_ix == game.player_count:
                    first_attacker_ix = 0
            game.draw(first_attacker_ix + 1)
            if game.field.attack_cards:
                game.take()
            else:
                game.draw(first_attacker_ix)
                game.update_defender()
        for ix, player in enumerate(game.players):
            if player.cards:
                durak_ix = ix
        if durak_ix == game.kraudia_ix:
            # TODO negative reward
            print('Kraudia lost...')
        else:
            # TODO positive reward
            print('Kraudia did not lose!')


class ActionReceiver(threading.Thread):
    """Receives all actions for the given player for one round."""

    def __init__(self, player_ix, event):
        threading.Thread.__init__(self)
        self.player_ix = player_ix
        self.event = event

    def run(self):
        global game
        player = game.players[self.player_ix]
        if False and self.player_ix == game.kraudia_ix:
            while not player.checks and possible_actions:
                possible_actions = game.get_actions(self.player_ix)
                if (not game.defender_ix == self.player_ix
                        and not (game.field.attack_cards
                        and game.field.defended_pairs)
                        or game.defender_ix == self.player_ix):
                    # TODO receive action from neural net
                    if action[0] != 4:
                        possible_actions.remove(action)
                else:
                    add_action(self.player_ix, game.wait_action())
        else:
            # first attacker
            if (self.player_ix == game.prev_neighbour(game.defender_ix)
                    and game.field.is_empty()):
                possible_actions = game.get_actions(self.player_ix)
                action = choice(possible_actions[:len(possible_actions) - 1])
                add_action(self.player_ix, action)
            self.event.wait()
            possible_actions = game.get_actions(self.player_ix)
            # attacker
            if game.defender_ix != self.player_ix:
                while not player.checks and len(possible_actions) > 1:
                    # everything is defended
                    if (not game.field.attack_cards
                            and np.random.random() < epsilon):
                        action = choice(possible_actions)
                        add_action(self.player_ix, action)
                add_action(self.player_ix, game.check_action())
            # defender
            else:
                while not player.checks and len(possible_actions) > 1:
                    # defender
                    elif game.defender_ix == self.player_ix:
                        if np.random.random() < epsilon:
                            action = choice(possible_actions)
                            add_action(self.player_ix, action)
                        else:
                            add_action(self.player_ix, game.wait_action())
                    else:
                        add_action(self.player_ix, game.wait_action())
                add_action(self.player_ix, game.check_action())


def add_action(player_ix, action):
    """Add an action with the belonging player to the action queue."""

    global action_queue
    print(player_ix, action)
    action_queue.put((player_ix, action))


def make_card(action):
    """Create a card from an action.
    Creates a tuple of two cards if action is defending."""

    if action[0] == 1:
        return (deck.Card(action[3], action[4], numerical=True),
                deck.Card(action[1], action[2], numerical=True))
    else:
        return deck.Card(action[1], action[2], numerical=True)


if __name__ == '__main__':
    durak_ix = -1
    game = None
    epsilon = None
    threads = []
    action_queue = queue.Queue(50)

    main()