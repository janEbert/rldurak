import deck, player as player_m, field, game as game_m
import random, threading, queue

if __name__ == '__main__':
    # name "Kraudia" is required to find the agent
    names = ["Kraudia", "Bob"]
    deck_size = 52
    hand_size = 6
    durak_ix = -1
    threads = []
    action_queue = queue.Queue()
    game = None
    # how often random bots wait
    epsilon = 0.8

    main()


def main():
    while (True):
        game = game_m.Game(names, deck_size, hand_size)
        if (durak_ix < 0):
            beginner_ix, beginner_card = game.find_beginner()
            if (beginner_card == game.deck.bottom_trump):
                print("Beginner was chosen randomly")
        else:
            game.defender_ix = durak_ix
        while not game.ended():
            for ix in game.active_player_indices:
                thread = threading.Thread(target=receive_action,
                        args=(ix,))
                thread.start()
                threads.append(thread)
            while not game.attack_ended():


def receive_action(player_ix):
    player = game.players[player_ix]
    if player_ix == game.kraudia_ix:
        # receive action from neural net
    else:
        while not player.checks:
            if random.random() > epsilon:
                action_queue.put(random.choice(game.get_actions(player_ix)))
            else:
                action_queue.put(game.wait_action())