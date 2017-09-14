import sys
from random import shuffle as rshuffle, randint

if sys.version_info[0] == 2:
    range = xrange

values = [str(x) for x in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
value_dict = dict(zip(values, range(13)))
suits = ['C', 'S', 'H', 'D']
suit_dict = dict(zip(suits, range(4)))


def same_value(cards):
    """Test whether all given cards have the same value."""
    if len(cards) > 0:
        comp = cards[0].value
        for card in cards[1:]:
            if card.value != comp:
                return False
        return True
    return False


def cards_to_string(cards, numerical=False):
    """Return a list of cards as a string.

    Also allows numerical representation.
    """
    if cards:
        card = cards[0]
        if not numerical:
            if isinstance(card, tuple):
                string = '[(' + str(card[0]) + ', ' + str(card[1]) + ')'
                for card in cards[1:]:
                    string += ', (' + str(card[0]) + ', ' + str(card[1]) + ')'
            else:
                string = '[' + str(card)
                for card in cards[1:]:
                    string += ', ' + str(card)
        else:
            string = '[' + repr(card)
            for card in cards[1:]:
                string += ', ' + repr(card)
    else:
        string = '['
    return string + ']'


class Card:
    """A standard playing card with value and suit."""

    def __init__(self, value, suit, index, numerical=True):
        """Construct a card via readable or numerical value or suit
        with the given index in the feature vector.
        """
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
        self.index = index

    def __eq__(self, other):
        return self.value == other.value and self.suit == other.suit

    def __str__(self):
        return str(self.value) + str(self.suit)

    def __repr__(self):
        return str(self.num_value) + str(self.num_suit)


class Deck:
    """A deck of standard playing cards with a trump card at
    the bottom.
    """

    def __init__(self, size=52, trump_suit=None, buffer_features=False):
        """Initialize the deck with the lowest cards removed if size
        is less than 52.
        """
        assert size >= 20 and size <= 104 and size % 4 == 0, \
                'Size does not make sense'
        self.size = size
        if size <= 52:
            self.cards_per_suit = size // 4
        else:
            self.cards_per_suit = size // 8
        self.fill(buffer_features)
        if trump_suit is not None:
            assert trump_suit in suits or trump_suit in range(4), \
                    'Trump suit is invalid'
            if trump_suit in suits:
                trump_suit = suit_dict[trump_suit]
            trump_num_value = randint(0, 12)
            bottom_trump = self.cards.pop(trump_num_value * 4 + trump_suit)
            self.shuffle()
            self.cards.append(bottom_trump)
        else:
            self.shuffle()
        self.reveal_trump()

    def fill(self, buffer_features):
        """Fill the deck up to the desired size (unshuffled!).

        If size is greater than 52, add duplicate cards.
        """
        if buffer_features:
            cards_per_suit = 13
        else:
            cards_per_suit = self.cards_per_suit
        if self.size <= 52:
            self.cards = [Card(num_value, num_suit, num_value + num_suit
                    * cards_per_suit) for num_value in range(13)
                    for num_suit in range(4)][52 - self.size:]
        else:
            self.cards = [Card(num_value, num_suit, num_value + num_suit
                    * cards_per_suit) for num_value in range(13)
                    for num_suit in range(4)][52 - self.size / 2:]
            self.cards = self.cards + self.cards[:]

    def shuffle(self):
        """Shuffle the deck's cards."""
        rshuffle(self.cards)

    def reveal_trump(self):
        """Reveal the trump card at the bottom of the deck."""
        self.bottom_trump = self.cards[-1]
        self.trump_suit = self.bottom_trump.suit;
        self.num_trump_suit = suit_dict[self.trump_suit];

    def take(self, amount):
        """Return and remove the given amount of cards from
        the deck.
        """
        self.size -= amount
        if self.size < 0:
            self.size = 0
        taken = self.cards[:amount]
        self.cards = self.cards[amount:]
        return taken