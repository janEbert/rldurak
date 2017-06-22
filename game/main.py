import deck, player, field, game

# name "Kraudia" is required to find the agent
names = ["Kraudia", "Bob"]
deck_size = 52
hand_size = 6
durak_ix = -1

main()


def main():
    while (True):
        game = Game(names, deck_size, hand_size)
        if (durak_ix < 0):
            beginner_ix, beginner_card = game.find_beginner()
            if (beginner_card == game.deck.bottom_trump):
                print("Beginner was chosen randomly")
            game.defender_ix = (beginner_ix + 1) % game.player_count
        else:
            game.defender_ix = durak_ix
        while not game.ended():
            while not game.attack_ended():


def get_attack():
