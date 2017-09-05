class Field:
    """A playing field containing the deck, players and cards that are
    put for everyone to see.

    Contains a list of cards on the field and a list that pairs an
    attack card with a card used to defend it."""

    def __init__(self):
        self.attack_cards = []
        self.defended_pairs = []

    def __eq__(self, other):
        return (self.attack_cards == other.attack_cards
                and self.defended_pairs == other.defended_pairs)

    def attack(self, cards):
        """Puts the cards as attacks on the field."""

        self.attack_cards.extend(cards)

    def defend(self, to_defend, card):
        """Defends the card to defend with the given card."""

        assert to_defend in self.attack_cards, 'Card is not on the field'
        self.attack_cards.remove(to_defend)
        self.defended_pairs.append((to_defend, card))

    def push(self, cards):
        """Adds the cards used to push to the attack cards."""

        self.attack_cards.extend(cards)

    def take(self):
        """Returns all cards on the field and clears it."""

        to_take = self.attack_cards[:]
        for (att_card, def_card) in self.defended_pairs:
            to_take.extend([att_card, def_card])
        self.clear()
        return to_take

    def values_on_field(self, cards):
        """Tests whether the given cards' values are on the
        field aleady."""

        found = [False] * len(cards)
        for i, card in enumerate(cards):
            for att_card in self.attack_cards:
                if card.value == att_card.value:
                    found[i] = True
                    break
            if not found[i]:
                for (att_card, def_card) in self.defended_pairs:
                    if (card.value == att_card.value
                            or card.value == def_card.value):
                        found[i] = True
                        break
        return False not in found

    def on_field(self, card):
        """Tests whether the given card is on the field."""

        for att_card in self.attack_cards:
            if att_card == card:
                return True
        for (att_card, def_card) in self.defended_pairs:
            if att_card == card or def_card == card:
                return True
        return False

    def on_field_attack(self, card):
        """Tests whether the given card is on the field
        as an attack."""

        for att_card in self.attack_cards:
            if att_card == card:
                return True
        return False

    def is_empty(self):
        """Tests whether no cards are on the field."""

        return not (self.attack_cards or self.defended_pairs)

    def clear(self):
        """Clears the playing field of cards."""

        self.attack_cards.clear()
        self.defended_pairs.clear()