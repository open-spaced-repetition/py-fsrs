<div align="center">
  <img src="https://avatars.githubusercontent.com/u/96821265?s=200&v=4" height="100" alt="Open Spaced Repetition logo"/>
</div>
<div align="center">

# Py-FSRS

</div>
<div align="center">
  <em>🧠🔄 Build your own Spaced Repetition System in Python 🧠🔄   </em>
</div>
<br />
<div align="center" style="text-decoration: none;">
    <a href="https://pypi.org/project/fsrs/"><img src="https://img.shields.io/pypi/v/fsrs"></a>
    <a href="https://github.com/open-spaced-repetition/py-fsrs/blob/main/LICENSE" style="text-decoration: none;"><img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</div>
<br />


**Py-FSRS is a python package that allows developers to easily create their own spaced repetition system using the <a href="https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler">Free Spaced Repetition Scheduler algorithm</a>.**


---


## Installation
You can install the `fsrs` python package from [PyPI](https://pypi.org/project/fsrs/) using pip:
```
pip install fsrs
```

## Quickstart

Import and initialize the FSRS scheduler

```python
from fsrs import Scheduler, Card, Rating, ReviewLog

scheduler = Scheduler()
```

Create a new Card object
```python
# note: all new cards are 'due' immediately upon creation
card = Card()
```

Choose a rating and review the card with the scheduler
```python
"""
Rating.Again # (==0) forgot the card
Rating.Hard # (==1) remembered the card, but with serious difficulty
Rating.Good # (==2) remembered the card after a hesitation
Rating.Easy # (==3) remembered the card easily
"""

rating = Rating.Good

card, review_log = scheduler.review_card(card, rating)

print(f"Card rated {review_log.rating} at {review_log.review_datetime}")
# > Card rated 3 at 2024-11-30 17:46:58.856497+00:00
```

See when the card is due next
```python
from datetime import datetime, timezone

due = card.due

# how much time between when the card is due and now
time_delta = due - datetime.now(timezone.utc)

print(f"Card due on {due}")
print(f"Card due in {time_delta.seconds} seconds")

"""
> Card due on 2024-11-30 18:42:36.070712+00:00
> Card due in 599 seconds
"""
```

## Usage

### Custom parameters

You can initialize the FSRS scheduler with your own custom parameters.

```python
from datetime import timedelta

# note: the following arguments are also the defaults
scheduler = Scheduler(
    parameters = (
            0.4072,
            1.1829,
            3.1262,
            15.4722,
            7.2102,
            0.5316,
            1.0651,
            0.0234,
            1.616,
            0.1544,
            1.0824,
            1.9813,
            0.0953,
            0.2975,
            2.2042,
            0.2407,
            2.9466,
            0.5034,
            0.6567,
        ),
    desired_retention = 0.9,
    learning_steps = (timedelta(minutes=1), timedelta(minutes=10)),
    relearning_steps = (timedelta(minutes=10),),
    maximum_interval = 36500,
    enable_fuzzing = True
)
```

#### Explanation of parameters

`parameters` are a set of 19 model weights that affect how the FSRS scheduler will schedule future reviews. If you're not familiar with optimizing FSRS, it is best not to modify these default values.

`desired_retention` is a value between 0 and 1 that sets the desired minimum retention rate for cards when scheduled with the scheduler. For example, with the default value of `desired_retention=0.9`, a card will be scheduled at a time in the future when the predicted probability of the user correctly recalling that card falls to 90%. A higher `desired_retention` rate will lead to more reviews and a lower rate will lead to fewer reviews.

`learning_steps` are custom time intervals that schedule new cards in the Learning state. By default, cards in the Learning state have short intervals of 1 minute then 10 minutes. You can also disable `learning_steps` with `Scheduler(learning_steps=())`

`relearning_steps` are analogous to `learning_steps` except they apply to cards in the Relearning state. Cards transition to the Relearning state if they were previously in the Review state, then were rated Again - this is also known as a 'lapse'. If you specify `Scheduler(relearning_steps=())`, cards in the Review state, when lapsed, will not move to the Relearning state, but instead stay in the Review state.

`maximum_interval` sets the cap for the maximum days into the future the scheduler is capable of scheduling cards. For example, if you never want the scheduler to schedule a card more than one year into the future, you'd set `Scheduler(maximum_interval=365)`.

`enable_fuzzing`, if set to True, will apply a small amount of random 'fuzz' to calculated intervals. For example, a card that would've been due in 50 days, after fuzzing, might be due in 49, or 51 days.

### Timezone

**Py-FSRS uses UTC only.** 

You can still specify custom datetimes, but they must use the UTC timezone.

### Retrievability

You can calculate the current probability of correctly recalling a card (its 'retrievability') with

```python
retrievability = card.get_retrievability()

print(f"There is a {retrievability} probability that this card is remembered.")
# > There is a 0.94 probability that this card is remembered.
```

### Serialization

`Scheduler`, `Card` and `ReviewLog` objects are all JSON-serializable via their `to_dict` and `from_dict` methods for easy database storage:

```python
# serialize before storage
scheduler_dict = scheduler.to_dict()
card_dict = card.to_dict()
review_log_dict = review_log.to_dict()

# deserialize from dict
new_scheduler = Scheduler.from_dict(scheduler_dict)
new_card = Card.from_dict(card_dict)
new_review_log = ReviewLog.from_dict(review_log_dict)
```

## Reference

Card objects have one of three possible states
```python
State.Learning # new card being studied for the first time
State.Review # card that has "graduated" from the Learning state
State.Relearning # card that has lapsed from the Review state
```

There are four possible ratings when reviewing a card object:
```python
Rating.Again # (==0) forgot the card
Rating.Hard # (==1) remembered the card, but with serious difficulty
Rating.Good # (==2) remembered the card after a hesitation
Rating.Easy # (==3) remembered the card easily
```

## Contribute

Checkout [CONTRIBUTING](CONTRIBUTING.md) to help improve Py-FSRS!