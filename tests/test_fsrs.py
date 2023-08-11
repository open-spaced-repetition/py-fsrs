from src.fsrs import *
from datetime import datetime

f = FSRS()
f.params.w = (
    1.14,
    1.01,
    5.44,
    14.67,
    5.3024,
    1.5662,
    1.2503,
    0.0028,
    1.5489,
    0.1763,
    0.9953,
    2.7473,
    0.0179,
    0.3105,
    0.3976,
    0.0,
    2.0902,
)

ratings = (
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Again,
    Rating.Again,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
)


def test_ivl():
    card = Card()
    ivl_history = []
    now = datetime.now()
    for rating in ratings:
        card, log = f.review(card, rating, now=now)
        ivl_history.append(card.scheduled_days)
        now = card.due
    assert ivl_history == [0, 5, 16, 43, 106, 236, 0, 0, 12, 25, 47, 85, 147]


def test_state():
    card = Card()
    state_history = []
    now = datetime.now()
    for rating in ratings:
        state_history.append(card.state)
        card, log = f.review(card, rating, now=now)
        now = card.due
    assert state_history == [
        State.New,
        State.Learning,
        State.Review,
        State.Review,
        State.Review,
        State.Review,
        State.Review,
        State.Relearning,
        State.Relearning,
        State.Review,
        State.Review,
        State.Review,
        State.Review,
    ]
