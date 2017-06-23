from random import shuffle

values = [str(x) for x in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
value_dict = dict(zip(values, range(13)))
suits = ['C', 'S', 'H', 'D']
suit_dict = dict(zip(suits, range(4)))


def same_value(cards):
    """Tests whether all given cards have the same value."""

    comp = cards[0].value
    for card in cards[1:]:
        if card.value != comp:
            return False
    return True


def print_cards(cards):
    """Prints a list of cards on the console."""

    string = str(cards[0].value) + str(cards[0].suit)
    for card in cards[1:]:
        string += ", " + str(card.value) + str(card.suit)


class Card:
    """A standard playing card with value and suit."""

    def __init__(self, value, suit):
        assert value in values and suit in suits, "Value or suit is not valid"
        self.value = value;
        self.suit = suit;
        self.num_value = value_dict[self.value]
        self.num_suit = suit_dict[self.suit]


class Deck:
    """A deck of standard playing cards with a trump card at
    the bottom."""

    def __init__(self, size=52):
        """Initializes the deck with the lowest cards removed if size
        is less than 52."""

        assert size >= 20 and size <= 104 and size % 4 == 0, 
                "Size does not make sense"
        self.size = size
        self.cards = []
        self.fill()
        self.shuffle()

    def fill(self):
        """Fills the deck up to the desired size (unshuffled!).

        If size is greater than 52, duplicate cards are added."""

        if self.size <= 52:
            self.cards = [Card(value, suit) for value in values
                    for suit in suits][52 - self.size:]
        else:
            self.cards = [Card(value, suit) for value in values
                    for suit in suits][52 - self.size / 2:]
            self.cards = self.cards + self.cards[:]

    def shuffle(self):
        """Shuffles the deck's cards and updates the revealed trump."""

        shuffle(self.cards);
        self.bottom_trump = self.cards[len(self.cards) - 1]
        self.trump_suit = self.bottom_trump.suit;

    def take(self, amount):
        """Returns and removes the given amount of cards from the deck."""

        self.size -= amount
        if self.size < 0:
            self.size = 0
        taken = self.cards[:amount]
        self.cards = self.cards[amount:]
        return taken