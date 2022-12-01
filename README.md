## About The Project

Py-fsrs is a Python Package implements [Free Spaced Repetition Scheduler algorithm](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler). It helps developers apply FSRS in their flashcard apps.

## Getting Started

```
pip install fsrs
```

## Usage

Create a card and review it at a given time:
```python
from fsrs import *
f = FSRS()
card = Card()
now = datetime(2022, 11, 29, 12, 30, 0, 0)
scheduling_cards = f.repeat(card, now)
```

There are four ratings:
```python
Rating.Again # forget; incorrect response
Rating.Hard # recall; correct response recalled with serious difficulty
Rating.Good # recall; correct response after a hesitation
Rating.Easy # recall; perfect response
```


Get the new state of card for each rating:
```python
scheduling_cards[Rating.Again].card
scheduling_cards[Rating.Hard].card
scheduling_cards[Rating.Good].card
scheduling_cards[Rating.Easy].card
```

Get the scheduled days for each rating:
```python
card_again.scheduled_days
card_hard.scheduled_days
card_good.scheduled_days
card_easy.scheduled_days
```

Update the card after rating `Good`:
```python
card = scheduling_cards[Rating.Good].card
```

Get the review log after rating `Good`:
```python
review_log = scheduling_cards[Rating.Good].review_log
```

Get the due date for card:
```python
due = card.due
```

There are four states:
```python
State.New # Never been studied
State.Learning # Been studied for the first time recently
State.Review # Graduate from learning state
State.Relearning # Forgotten in review state
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
