"""Who the robot is talking to.

One face lookup per turn, not per frame: the robot takes the frame it already
captured for the turn, asks the Mac for a face vector (emulator/face.py), and
matches it against the people it has met in its own shard
(emulator/face_memory.py). Conversation memory stays exactly as it was — read
only when the model calls `remember`. This is a different question, asked at a
different time: not "what do I remember about this" but "who is standing
there", and the answer is needed before the turn is answered, not during it.

Meeting someone is a two-turn exchange the robot drives itself, without the
language model: it asks for a name, the next thing said is taken as the
answer, and the shots it has collected are enrolled under that name. Keeping
the model out of it is deliberate — these two lines must be the same every
time, and a tool call that misfires here would greet the room with silence.

Calibrated on this robot's own camera (eight stored frames of one person):
shots of the same person taken minutes apart scored 0.60-0.77 against each
other, but two frames at a different angle and light fell to 0.25-0.40. That
is why enrollment takes several shots rather than one — the match is against
the closest of them.
"""
from __future__ import annotations

import dataclasses
import re
import time

# Shots kept for one person at enrollment. More poses, more chances one of
# them is close to however they are standing later.
ENROLL_SHOTS = 5

# The person the robot last named stays "the person in front of it" for as long
# as a face never leaves the camera for longer than this — and only for a face
# that is too close to call (see Match.is_new), never for a clear stranger. Seen live
#: met with 2 shots, the same person two turns later scored under
# the new-person line and was asked their name again. A face that stayed in
# view is the same person, whatever the embedding says.
FACE_GONE_S = 10.0

# Shots added to that person while they stay in view and their face does not
# match — the poses enrollment missed. Capped per person: this runs every turn,
# and one visitor in bad light must not use up the next visitor's share.
MAX_LEARNED_SHOTS = 10

# What the robot says. Fixed lines, spoken through demo/serve.py's /say — the
# model is not asked to improvise the one moment the demo is named after.
ASK_NAME = "I don't think we've met. What's your name?"
GREET_KNOWN = "Hello again, {name}!"
GREET_NEW = "Nice to meet you, {name}. I'll remember you."
NOT_CAUGHT = "Sorry, I didn't catch your name."

# A name in an answer to "what's your name?" — "Sasha", "I'm Sasha",
# "My name is Sasha". English only: the demo is English.
_NAME = re.compile(
    r"\b(?i:my name'?s|my name is|i'?m|i am|call me|this is|it'?s)\s+"
    r"([A-Z][a-z]{1,15})\b")
_NOT_A_NAME = frozenset({
    "Sorry", "Not", "Just", "Fine", "Good", "Okay", "Sure", "Here", "Yes",
    "No", "Hello", "Hi", "Hey", "Thanks", "Thank", "Nothing", "Nobody",
    "Never", "Giving", "Going", "Doing", "Talking", "Trying", "Looking",
    "What", "Who", "Why", "How", "When", "Where", "Please", "Maybe", "Well",
    "And", "But", "The", "This", "That", "You", "Your", "Me", "My", "We",
    "It", "Is", "Are", "Can", "Could", "Would", "Should", "Actually"})


def name_from_answer(heard: str) -> str | None:
    """The name in an answer to the name question.

    The question was just asked, so a bare "Sasha." is the most likely shape
    of the answer — that is what makes the lone capitalised word acceptable
    here and not in ordinary conversation (demo/conversation.py's
    speaker_name, which needs an explicit introduction)."""
    text = (heard or "").strip()
    if not text:
        return None
    match = _NAME.search(text)
    if match and match.group(1) not in _NOT_A_NAME:
        return match.group(1)
    # No introduction, so the answer is most likely the bare name. Only a
    # word written like one counts: the transcriber capitalises names and
    # sentence openers, so "Sasha." passes and the "what" in "Sorry, what?"
    # does not. Long answers are left alone — they are a sentence, not a name.
    words = re.findall(r"[A-Za-z][A-Za-z'-]{1,15}", text)
    if len(words) <= 3:
        for word in words:
            if word[:1].isupper() and word not in _NOT_A_NAME:
                return word
    return None


@dataclasses.dataclass
class Seen:
    """What the robot saw this turn."""
    name: str | None
    score: float
    box: list[float] | None

    @property
    def known(self) -> bool:
        return self.name is not None


class People:
    """The robot's side of meeting and recognising people."""

    def __init__(self, memory=None, reader=None, *, shots: int = ENROLL_SHOTS,
                 clock=time.monotonic, read_name=None) -> None:
        self._memory = memory
        self._reader = reader
        self._shots_wanted = shots
        self._read_name = read_name
        self._shots: list[list[float]] = []
        self.awaiting_name = False
        self.current = Seen(None, 0.0, None)
        self._greeted: set[str] = set()
        self._clock = clock
        self._here: str | None = None      # named, and still in view
        self._face_last_seen = float("-inf")
        self._learned: dict[str, int] = {}   # poses learned, per person

    def face_seen(self) -> None:
        """A face is in the camera now — from the detect loop (demo/run_demo.py's
        FaceTracker), several times a second. A gap longer than FACE_GONE_S
        means whoever is there now may be someone else."""
        now = self._clock()
        if now - self._face_last_seen > FACE_GONE_S:
            self._here = None
        self._face_last_seen = now

    @property
    def enabled(self) -> bool:
        return self._memory is not None and self._reader is not None

    def observe(self, frame) -> Seen:
        """Look once: who is in front of the robot right now."""
        if not self.enabled or frame is None:
            return self.current
        try:
            faces = self._reader.read(frame)
        except Exception as exc:  # noqa: BLE001 — a turn must not hang on this
            print(f"  [faces] skip ({type(exc).__name__}: {exc})")
            return self.current
        face = next((f for f in faces if f.get("embedding")), None)
        if face is None:
            # No face to match this turn (turned away, too far, a bad frame) —
            # but if one was in the camera moments ago, it is still the person
            # the robot is talking to, and the picture it sends the model
            # should carry their name (demo/conversation.py's names_fn).
            box = faces[0]["box"] if faces else None
            self.current = (Seen(self._here, 0.0, box) if self._still_here()
                            else Seen(None, 0.0, box))
            return self.current
        self.face_seen()
        match = self._memory.recognize(face["embedding"])
        if not match.known and not match.is_new and self._here is not None:
            # Close but under the line, with the same person still in view:
            # this is them at a bad angle. A face that is CLEARLY someone else
            # (match.is_new) never gets their name — seen live: a second
            # person in front of the robot was called Sasha, and their
            # face was learned into Sasha's point.
            return self._still_the_same_person(face, match.score)
        if match.known:
            self._here = match.name
            self._shots.clear()
        elif match.is_new and len(self._shots) < self._shots_wanted:
            # Collect while the person is here; enrollment needs several poses.
            self._shots.append(face["embedding"])
        self.current = Seen(match.name if match.known else None,
                            match.score, face["box"])
        return self.current

    def in_frame(self, frame) -> list[dict]:
        """Who is in a frame: [{"name", "box", "score"}], largest face first,
        name None for someone the robot has not met. Stored with a frame
        (emulator/frame_memory.py). Does not enrol or greet — that stays with
        observe(). A frame whose faces cannot be read is kept without names."""
        if not self.enabled or frame is None:
            return []
        try:
            faces = self._reader.read(frame)
        except Exception as exc:  # noqa: BLE001 — a frame is kept without names
            print(f"  [faces] skip ({type(exc).__name__}: {exc})")
            return []
        return self._named(faces)

    def faces_in(self, frame) -> list[dict]:
        """in_frame, raising when the faces cannot be read: the `who` answer
        (demo/conversation.py) has to tell "nobody is there" from "I cannot
        tell who is there"."""
        if not self.enabled or frame is None:
            return []
        return self._named(self._reader.read(frame))

    def _named(self, faces) -> list[dict]:
        found = []
        for face in faces:
            if not face.get("embedding"):
                continue
            match = self._memory.recognize(face["embedding"])
            name = match.name if match.known else None
            if name is None and not found and self._still_here():
                name = self._here  # the tracked person, at a bad angle
            found.append({"name": name, "box": face.get("box"),
                          "score": round(float(match.score), 3)})
        return found

    def met(self) -> list[str]:
        """Everyone this robot has met, by name."""
        return self._memory.people() if self.enabled else []

    def met_when(self) -> dict[str, float]:
        """Name -> when the robot first met them (emulator/face_memory.py's
        met_at, stamped once and never moved by a later pose). Empty without a
        face shard, or on a shard written before the field existed."""
        if not self.enabled:
            return {}
        people_met = getattr(self._memory, "people_met", None)
        if people_met is None:
            return {}
        return {name: ts for name, ts in people_met()}

    def close(self) -> None:
        """Release the faces shard — the last enrolment lands on disk here."""
        if self._memory is not None:
            self._memory.close()

    def _still_here(self) -> bool:
        return (self._here is not None
                and self._clock() - self._face_last_seen <= FACE_GONE_S)

    def _still_the_same_person(self, face, score: float) -> Seen:
        """The face does not match, but it never left the camera: it is the
        person already named. Learn this pose, so it matches next time."""
        if self._learned.get(self._here, 0) < MAX_LEARNED_SHOTS:
            try:
                self._learned[self._here] = (self._learned.get(self._here, 0)
                                             + self._memory.enroll(self._here,
                                                                   [face["embedding"]]))
            except Exception as exc:  # noqa: BLE001 — a turn must not hang on this
                print(f"  [faces] could not learn a pose ({type(exc).__name__}: {exc})")
        self._shots.clear()
        self.current = Seen(self._here, score, face["box"])
        return self.current

    def greeting(self) -> str | None:
        """A line to say before answering, or None. Said once per person per
        run: a robot that greets you on every turn is a robot with a glitch,
        not a memory."""
        if self.current.known and self.current.name not in self._greeted:
            self._greeted.add(self.current.name)
            return GREET_KNOWN.format(name=self.current.name)
        return None

    def should_ask_name(self) -> bool:
        """Whether to ask who this is: somebody is there, nobody we know, and
        we have shots to remember them by."""
        return (self.enabled and not self.awaiting_name
                and not self.current.known and bool(self._shots))

    def ask_name(self) -> str:
        self.awaiting_name = True
        return ASK_NAME

    def answer_name(self, heard: str) -> tuple[str | None, str]:
        """Take the answer to the name question: returns (name, what to say).
        A name that could not be made out is not enrolled — a wrong name
        attached to a face outlives the mistake.

        `read_name` (demo/run_demo.py, the laptop's /name) asks the model
        what the name was; the pattern below is the fallback for when the
        model cannot be asked."""
        self.awaiting_name = False
        name = None
        read = False
        if self._read_name is not None:
            try:
                name = self._read_name(heard)
                read = True
            except Exception as exc:  # noqa: BLE001 — fall back to the pattern
                print(f"  [people] the name reader failed "
                      f"({type(exc).__name__}: {exc})")
        if not read:
            # Only when the model could not be asked: its "no name here" is
            # an answer, and the pattern takes "What's yours?" for a name.
            name = name_from_answer(heard)
        if not name:
            return None, NOT_CAUGHT
        stored = self._memory.enroll(name, self._shots)
        self._shots.clear()
        self._greeted.add(name)
        self._here = name
        print(f"  people:  met {name} ({stored} shot(s)) -> Qdrant Edge")
        self.current = Seen(name, 1.0, self.current.box)
        return name, GREET_NEW.format(name=name)
