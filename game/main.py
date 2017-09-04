import deck, player as player_m, field, game as game_m
import threading, queue
from random import choice
import numpy as np

# name 'Kraudia' is required to find the agent
names = ['Kraudia', 'Bob', 'Alice']
deck_size = 52
hand_size = 6

iterations = 1000
# how often random bots wait
# calculated from a normal distribution with the given values
epsilon_mu = 0.7
epsilon_sigma = 0.15


def main():
    """Main loop for Durak for receiving and executing actions and
    giving rewards."""

    global durak_ix, game, epsilon, threads, action_queue
    for n in range(iterations):
        epsilon = min(0.9, np.random.normal(epsilon_mu, epsilon_sigma))
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
                thread = threading.Thread(target=receive_action, args=(ix,))
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
            print('attack ended')
            # attack ended
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
            if game.field.attack_cards != []:
                game.take()
            else:
                game.draw(first_attacker_ix)
                game.update_defender()
        for ix, player in enumerate(game.players):
            if player.cards != []:
                durak_ix = ix
        if durak_ix == game.kraudia_ix:
            # TODO negative reward
            print('Kraudia lost...')
        else:
            # TODO positive reward
            print('Kraudia did not lose!')


def receive_action(player_ix):
    """Receives all actions for the player for one round."""

    global game
    player = game.players[player_ix]
    possible_actions = [0]
    if False and player_ix == game.kraudia_ix:
        while not player.checks and possible_actions != []:
            possible_actions = game.get_actions(player_ix)
            if (not game.defender_ix == player_ix
                    and (game.field.attack_cards == []
                    or game.field.defended_pairs == [])
                    or game.defender_ix == player_ix):
                # TODO receive action from neural net
                if action[0] != 4:
                    possible_actions.remove(action)
            else:
                add_action(player_ix, game.wait_action())
    else:
        while not player.checks and possible_actions != []:
            possible_actions = game.get_actions(player_ix)
            if not game.defender_ix == player_ix and game.field.is_empty():
                action = choice(possible_actions)
                while action[0] == 3 or action[0] == 4:
                    action = choice(possible_actions)
                add_action(player_ix, action)
            elif (not game.defender_ix == player_ix
                    and game.field.attack_cards == []):
                action = choice(possible_actions)
                add_action(player_ix, action)
            elif game.defender_ix == player_ix:
                if np.random.random() > epsilon:
                    action = choice(possible_actions)
                    add_action(player_ix, action)
                else:
                    add_action(player_ix, game.wait_action())
            else:
                add_action(player_ix, game.wait_action())


def add_action(player_ix, action):
    """Add an action with the belonging player to the action queue."""

    global action_queue
    print(player_ix, action)
    action_queue.put((player_ix, action))


def make_card(action):
    """Create a card from an action.

    Creates a tuple of two cards if action is defending."""

    if action[0] == 0:
        return deck.Card(num_value=action[1], num_suit=action[2])
    elif action[0] == 1:
        return deck.Card(num_value=action[3], num_suit=action[4]), \
                deck.Card(num_value=action[1], num_suit=action[2])


if __name__ == '__main__':
    durak_ix = -1
    game = None
    epsilon = None
    threads = []
    action_queue = queue.Queue(50)

    main()