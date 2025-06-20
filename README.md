<div align="center">
  <img src="https://raw.githubusercontent.com/open-spaced-repetition/py-fsrs/main/osr_logo.png" height="100" alt="Open Spaced Repetition logo"/>
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
    <a href="https://pypi.org/project/fsrs/"><img src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
    <a href="https://github.com/open-spaced-repetition/py-fsrs/blob/main/LICENSE" style="text-decoration: none;"><img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://codecov.io/gh/open-spaced-repetition/py-fsrs" ><img src="https://codecov.io/gh/open-spaced-repetition/py-fsrs/graph/badge.svg?token=3G0FF6HZQD"/></a>
</div>
<br />


**Py-FSRS is a python package that allows developers to easily create their own spaced repetition system using the <a href="https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler">Free Spaced Repetition Scheduler algorithm</a>.**


## Table of Contents
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Optimizer (optional)](#optimizer-optional)
- [Reference](#reference)
- [API Documentation](#api-documentation)
- [Other FSRS implementations](#other-fsrs-implementations)
- [Other SRS python packages](#other-srs-python-packages)
- [Contribute](#contribute)

## Installation
You can install the `fsrs` python package from [PyPI](https://pypi.org/project/fsrs/) using pip:
```bash
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
# Rating.Again (==1) forgot the card
# Rating.Hard (==2) remembered the card with serious difficulty
# Rating.Good (==3) remembered the card after a hesitation
# Rating.Easy (==4) remembered the card easily

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

# > Card due on 2024-11-30 18:42:36.070712+00:00
# > Card due in 599 seconds
```

## Usage

### Custom parameters

You can initialize the FSRS scheduler with your own custom parameters.

```python
from datetime import timedelta

# note: the following arguments are also the defaults
scheduler = Scheduler(
    parameters = (
            0.2172,
            1.1771,
            3.2602,
            16.1507,
            7.0114,
            0.57,
            2.0966,
            0.0069,
            1.5261,
            0.112,
            1.0178,
            1.849,
            0.1133,
            0.3127,
            2.2934,
            0.2191,
            3.0004,
            0.7536,
            0.3332,
            0.1437,
            0.2,
    ),
    desired_retention = 0.9,
    learning_steps = (timedelta(minutes=1), timedelta(minutes=10)),
    relearning_steps = (timedelta(minutes=10),),
    maximum_interval = 36500,
    enable_fuzzing = True
)
```

#### Explanation of parameters

`parameters` are a set of 21 model weights that affect how the FSRS scheduler will schedule future reviews. If you're not familiar with optimizing FSRS, it is best not to modify these default values.

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
retrievability = scheduler.get_card_retrievability(card)

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

## Optimizer (optional)

If you have a collection of `ReviewLog` objects, you can optionally reuse them to compute an optimal set of parameters for the `Scheduler` to make it more accurate at scheduling reviews. You can also compute an optimal retention rate to reduce the future workload of your reviews.

### Installation

```bash
pip install "fsrs[optimizer]"
```

### Optimize scheduler parameters

```python
from fsrs import ReviewLog, Optimizer, Scheduler

# load your ReviewLog objects into a list (order doesn't matter)
review_logs = [ReviewLog1, ReviewLog2, ...]

# initialize the optimizer with the review logs
optimizer = Optimizer(review_logs)

# compute a set of optimized parameters
optimal_parameters = optimizer.compute_optimal_parameters()

# initialize a new scheduler with the optimized parameters
scheduler = Scheduler(optimal_parameters)
```

### Optimize desired retention

```python
optimal_retention = optimizer.compute_optimal_retention(optimal_parameters)

# initialize a new scheduler with both optimized parameters and retention
scheduler = Scheduler(optimal_parameters, optimal_retention)
```

Note: The computed optimal parameters and retention may be slightly different than the numbers computed by Anki for the same set of review logs. This is because the two implementations are slightly different and updated at different times. If you're interested in the official Rust-based Anki implementation, please see [here](https://github.com/open-spaced-repetition/fsrs-rs).

## Reference

Card objects have one of three possible states
```python
State.Learning # (==1) new card being studied for the first time
State.Review # (==2) card that has "graduated" from the Learning state
State.Relearning # (==3) card that has "lapsed" from the Review state
```

There are four possible ratings when reviewing a card object:
```python
Rating.Again # (==1) forgot the card
Rating.Hard # (==2) remembered the card with serious difficulty
Rating.Good # (==3) remembered the card after a hesitation
Rating.Easy # (==4) remembered the card easily
```

## API Documentation

You can find additional documentation for py-fsrs [here](https://open-spaced-repetition.github.io/py-fsrs).

## Other FSRS implementations

You can find various other FSRS implementations and projects [here](https://github.com/orgs/open-spaced-repetition/repositories?q=fsrs+sort%3Astars).

## Other SRS python packages

The following spaced repetition schedulers are also available as python packages:

- [SM-2](https://github.com/open-spaced-repetition/sm-2)
- [Leitner System](https://github.com/open-spaced-repetition/leitner-box)
- [Anki Default Scheduler](https://github.com/open-spaced-repetition/anki-sm-2)

## Contribute

If you encounter issues with py-fsrs or would like to contribute code, please see [CONTRIBUTING](https://github.com/open-spaced-repetition/py-fsrs/blob/main/CONTRIBUTING.md).