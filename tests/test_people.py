"""demo/people.py — recognising the person in front of the robot, and meeting
one it has not seen before."""
import pytest

from demo.people import (ASK_NAME, GREET_KNOWN, GREET_NEW, NOT_CAUGHT, People,
                         Seen, name_from_answer)


class _Memory:
    """FaceMemory stand-in: answers with a scripted match, records enrolments."""

    def __init__(self, match=None):
        self._match = match or _Match(None, 0.0)
        self.enrolled = []

    def recognize(self, embedding):
        return self._match

    def enroll(self, name, embeddings):
        shots = list(embeddings)
        self.enrolled.append((name, len(shots)))
        return len(shots)


class _Match:
    def __init__(self, name, score):
        self.name, self.score = name, score

    @property
    def known(self):
        return self.name is not None and self.score >= 0.35

    @property
    def is_new(self):
        return self.score < 0.25


class _Reader:
    def __init__(self, faces=None, fails=False):
        self._faces = faces if faces is not None else [
            {"box": [0.3, 0.2, 0.6, 0.7], "score": 0.9, "embedding": [0.1] * 512}]
        self._fails = fails
        self.reads = 0

    def read(self, frame, embed=True):
        self.reads += 1
        if self._fails:
            raise OSError("the Mac is not answering")
        return list(self._faces)


FRAME = object()


# — recognising —

def test_a_known_face_is_named_and_greeted_once():
    people = People(_Memory(_Match("Sasha", 0.72)), _Reader())
    seen = people.observe(FRAME)
    assert seen.known and seen.name == "Sasha" and seen.box == [0.3, 0.2, 0.6, 0.7]
    assert people.greeting() == GREET_KNOWN.format(name="Sasha")
    # A robot that greets you on every turn is a robot with a glitch.
    people.observe(FRAME)
    assert people.greeting() is None


def test_a_stranger_is_not_named_and_is_asked_who_they_are():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader())
    seen = people.observe(FRAME)
    assert not seen.known and seen.box is not None
    assert people.greeting() is None
    assert people.should_ask_name()
    assert people.ask_name() == ASK_NAME
    assert people.awaiting_name
    assert not people.should_ask_name(), "asked already; do not ask twice"


def test_meeting_someone_stores_several_shots_under_their_name():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), shots=3)
    for _ in range(5):
        people.observe(FRAME)          # more turns than shots wanted
    people.ask_name()
    name, line = people.answer_name("Sasha")
    assert name == "Sasha"
    assert line == GREET_NEW.format(name="Sasha")
    assert memory.enrolled == [("Sasha", 3)], "several poses, capped"
    assert not people.awaiting_name
    assert people.current.name == "Sasha"
    # Already met: no second greeting on the next turn.
    assert people.greeting() is None


def test_a_name_that_was_not_understood_is_not_enrolled():
    # A wrong name attached to a face outlives the mistake.
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader())
    people.observe(FRAME)
    people.ask_name()
    name, line = people.answer_name("Sorry, what?")
    assert name is None and line == NOT_CAUGHT
    assert memory.enrolled == []
    assert not people.awaiting_name


def test_a_borderline_match_is_neither_greeted_nor_enrolled():
    # Between the thresholds: too close to call.
    memory = _Memory(_Match("Sasha", 0.30))
    people = People(memory, _Reader())
    people.observe(FRAME)
    assert people.greeting() is None
    assert not people.should_ask_name(), "no shots collected for an unsure face"


def test_a_frame_with_no_face_keeps_the_box_but_names_nobody():
    people = People(_Memory(), _Reader(faces=[]))
    seen = people.observe(FRAME)
    assert seen == Seen(None, 0.0, None)
    assert not people.should_ask_name()


def test_a_face_service_that_fails_does_not_break_the_turn(capsys):
    people = People(_Memory(), _Reader(fails=True))
    assert people.observe(FRAME) == Seen(None, 0.0, None)
    assert "[faces] skip" in capsys.readouterr().out


def test_without_a_memory_or_a_reader_nothing_happens():
    people = People()
    assert not people.enabled
    assert people.observe(FRAME) == Seen(None, 0.0, None)
    assert people.greeting() is None and not people.should_ask_name()


# — the name in an answer —

@pytest.mark.parametrize("heard,name", [
    ("Sasha", "Sasha"), ("Sasha.", "Sasha"), ("I'm Sasha", "Sasha"),
    ("My name is Anna.", "Anna"), ("call me Bob", "Bob"), ("It is Anna", "Anna"),
])
def test_a_name_is_taken_from_the_answer(heard, name):
    assert name_from_answer(heard) == name


@pytest.mark.parametrize("heard", [
    "", "Sorry, what?", "Thanks for asking", "Hey there",
    "I am not sure I want to say",
])
def test_an_answer_without_a_name_gives_none(heard):
    assert name_from_answer(heard) is None


# — the same person, still in front of the robot —

def _clocked(memory, shots=5):
    now = [0.0]
    return People(memory, _Reader(), shots=shots, clock=lambda: now[0]), now


def test_someone_just_met_is_not_asked_again_while_they_stay_in_view():
    # Seen live: met with two shots, two turns later the face scored as new and
    # the robot asked "What's your name?" again.
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("I'm Sasha.")
    memory._match = _Match(None, 0.30)    # the same face, under the line
    for t in (2.0, 4.0, 6.0):             # the detect loop keeps seeing a face
        now[0] = t
        people.face_seen()
    now[0] = 8.0
    seen = people.observe(FRAME)
    assert seen.name == "Sasha"
    assert not people.should_ask_name()
    assert memory.enrolled[-1] == ("Sasha", 1), "the missed pose is learned"


def test_a_face_that_was_gone_a_while_is_looked_at_afresh():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 30.0                          # nobody in view for 30 s
    seen = people.observe(FRAME)
    assert seen.name is None
    assert people.should_ask_name()


def test_a_recognised_person_is_kept_through_a_bad_angle():
    memory = _Memory(_Match("Sasha", 0.7))
    people, now = _clocked(memory)
    assert people.observe(FRAME).name == "Sasha"
    memory._match = _Match(None, 0.30)     # turned their head: close, under the line
    now[0] = 3.0
    assert people.observe(FRAME).name == "Sasha"


def test_learning_poses_is_capped():
    from demo.people import MAX_LEARNED_SHOTS

    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    memory._match = _Match(None, 0.30)    # the same face, at a worse angle
    for turn in range(MAX_LEARNED_SHOTS + 5):
        now[0] = float(turn)
        people.observe(FRAME)
    learned = sum(count for name, count in memory.enrolled[1:])
    assert learned == MAX_LEARNED_SHOTS


def test_the_cap_on_learning_poses_is_per_person():
    # A demo has visitors: the first one standing in bad light used up the
    # whole run's cap, and nobody after them had a pose learned. The cap is
    # each person's for the run — coming back does not start it again.
    from demo.people import MAX_LEARNED_SHOTS

    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)

    def at_bad_angles(name, start, turns):
        memory._match = _Match(name, 0.30)   # the same face, at a worse angle
        for turn in range(turns):
            now[0] = start + turn
            people.observe(FRAME)

    # Sasha's first visit learns only 3: under the cap, so her return must
    # carry on from there — neither start over nor stop.
    for name, start, turns in (("Sasha", 0.0, 3),
                               ("Robin", 1000.0, MAX_LEARNED_SHOTS + 5)):
        now[0] = start                       # far apart: a new person
        memory._match = _Match(None, 0.05)   # someone new
        people.observe(FRAME)
        people.ask_name()
        people.answer_name(name)
        at_bad_angles(name, start, turns)
    now[0] = 2000.0
    memory._match = _Match("Sasha", 0.7)     # Sasha comes back, recognised
    people.observe(FRAME)
    at_bad_angles("Sasha", 2000.0, MAX_LEARNED_SHOTS + 5)

    learned = {}
    for name, count in memory.enrolled:
        learned.setdefault(name, []).append(count)
    assert sum(learned["Sasha"][1:]) == MAX_LEARNED_SHOTS
    assert sum(learned["Robin"][1:]) == MAX_LEARNED_SHOTS


# — who is in a frame —

def test_in_frame_names_known_faces_and_leaves_strangers_unnamed():
    class _Two(_Reader):
        def read(self, frame, embed=True):
            return [{"box": [0, 0, 1, 1], "embedding": [0.1]},
                    {"box": [0.5, 0, 1, 1], "embedding": [0.2]}]

    class _ByVector(_Memory):
        def recognize(self, embedding):
            return _Match("Sasha", 0.7) if embedding == [0.1] else _Match(None, 0.1)

        def people(self):
            return ["Sasha"]

    people = People(_ByVector(), _Two())
    found = people.in_frame(FRAME)
    assert [p["name"] for p in found] == ["Sasha", None]
    assert people.met() == ["Sasha"]
    assert memory_untouched(people)


def memory_untouched(people):
    return people._memory.enrolled == [] and not people.awaiting_name


def test_in_frame_keeps_the_tracked_person_named_at_a_bad_angle():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 3.0
    people.face_seen()
    assert [p["name"] for p in people.in_frame(FRAME)] == ["Sasha"]


def test_in_frame_without_faces_or_models_is_empty():
    assert People().in_frame(FRAME) == []
    assert People(_Memory(), _Reader(fails=True)).in_frame(FRAME) == []


def test_faces_in_says_when_the_faces_could_not_be_read():
    # in_frame keeps a frame without names; the `who` answer has to tell
    # "nobody is there" from "I cannot tell who is there".
    with pytest.raises(OSError):
        People(_Memory(), _Reader(fails=True)).faces_in(FRAME)
    assert People().faces_in(FRAME) == []


def test_a_clear_stranger_is_never_given_the_name_of_the_person_in_view():
    # Live: a second person in front of the robot was called Sasha,
    # and their face was learned into Sasha's point.
    memory = _Memory(_Match(None, 0.02))          # nothing like anyone known
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    enrolled_before = len(memory.enrolled)
    now[0] = 2.0
    people.face_seen()
    seen = people.observe(FRAME)
    assert seen.name is None, "a stranger keeps no name"
    assert len(memory.enrolled) == enrolled_before, "and teaches the robot nothing"


def test_the_same_person_at_a_bad_angle_keeps_their_name():
    memory = _Memory(_Match(None, 0.30))          # between the two thresholds
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 2.0
    people.face_seen()
    assert people.observe(FRAME).name == "Sasha"


def test_the_name_is_read_by_the_model_with_the_pattern_as_a_fallback():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=lambda heard: "Robin")
    people.observe(FRAME)
    people.ask_name()
    name, line = people.answer_name("they call me Robin")
    assert name == "Robin" and memory.enrolled[-1][0] == "Robin"

    def broken(heard):
        raise OSError("the Mac is not answering")

    people = People(_Memory(_Match(None, 0.05)), _Reader(), read_name=broken)
    people.observe(FRAME)
    people.ask_name()
    assert people.answer_name("I'm Sasha.")[0] == "Sasha", "the pattern still reads it"


def test_the_models_no_name_is_an_answer_not_a_reason_to_guess():
    # The model read "What's yours?" as no name; the pattern would take
    # "What's" for one. The pattern is for when the model cannot be asked.
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=lambda heard: None)
    people.observe(FRAME)
    people.ask_name()
    assert people.answer_name("I'm Sasha, what's yours?") == (None, NOT_CAUGHT)
    assert memory.enrolled == []


def test_the_pattern_reads_the_name_when_the_model_cannot_be_asked():
    def unreachable(heard):
        raise OSError("laptop gone")

    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=unreachable)
    people.observe(FRAME)
    people.ask_name()
    name, _line = people.answer_name("I'm Sasha.")
    assert name == "Sasha"
