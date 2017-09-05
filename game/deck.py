from random import shuffle as rshuffle

values = [str(x) for x in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
value_dict = dict(zip(values, range(13)))
suits = ['C', 'S', 'H', 'D']
suit_dict = dict(zip(suits, range(4)))


def same_value(cards):
    """Tests whether all given cards have the same value."""

    if len(cards) > 0:
        comp = cards[0].value
        for card in cards[1:]:
            if card.value != comp:
                return False
        return True
    return False


def cards_to_string(cards, numerical=False):
    """Returns a list of cards as a string.

    Also allows numerical representation."""

    if numerical:
        string = '[' + str(cards[0])
        for card in cards[1:]:
            string += ', ' + str(card)
    else:
        string = '[' + repr(cards[0])
        for card in cards[1:]:
            string += ', ' + repr(card)
    return string + ']'


class Card:
    """A standard playing card with value and suit."""

    def __init__(self, value, suit, numerical=False):
        if not numerical:
            assert value in values and suit in suits, \
                    'Value or suit is not valid'
            self.value = value;
            self.suit = suit;
            self.num_value = value_dict[self.value]
            self.num_suit = suit_dict[self.suit]
        else:
            assert (value >= 0 and value < 13 and suit >= 0
                    and suit < 4), 'Value or suit is not valid'
            self.value = values[value]
            self.suit = suits[suit]
            self.num_value = value
            self.num_suit = suit
        # index in full feature vector
        self.index = self.num_value + self.num_suit * 13

    def __eq__(self, other):
        return self.value == other.value and self.suit == other.suit

    def __str__(self):
        return str(self.value) + str(self.suit)

    def __repr__(self):
        return str(self.num_value) + str(self.num_suit)


class Deck:
    """A deck of standard playing cards with a trump card at
    the bottom."""

    def __init__(self, size=52):
        """Initializes the deck with the lowest cards removed if size
        is less than 52."""

        assert size >= 20 and size <= 104 and size % 4 == 0, \
                'Size does not make sense'
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

        rshuffle(self.cards);
        self.bottom_trump = self.cards[len(self.cards) - 1]
        self.trump_suit = self.bottom_trump.suit;
        self.num_trump_suit = suit_dict[self.trump_suit];

    def take(self, amount):
        """Returns and removes the given amount of cards from
        the deck."""

        self.size -= amount
        if self.size < 0:
            self.size = 0
        taken = self.cards[:amount]
        self.cards = self.cards[amount:]
        return taken