"""demo/stage.py — the Mac's four services from one command: what gets
started, how readiness is decided, and that everything stops together."""
import io
import signal

import pytest
import socket
import subprocess
import time

from demo.stage import (Service, Supervisor, _robot, _stop_on_sigterm,
                        build_services, main, parse_args, port_answers,
                        robot_env)


def _args(*argv):
    return parse_args(list(argv))


def _by_name(argv=()):
    return {s.name: s for s in build_services(_args(*argv))}


# — what gets started —

def test_every_mac_side_service_is_started():
    assert [s.name for s in build_services(_args())] == ["embed", "serve",
                                                         "detect", "dash"]


def test_ports_and_flags_reach_the_command_lines():
    services = _by_name(("--port", "9501", "--embed-port", "9901",
                         "--gpu-detect-port", "9601", "--web-port", "8081",
                         "--robot-host", "robot.local",
                         "--llm", "/models/x.litertlm", "--asr", "moonshine"))
    assert services["serve"].port == 9501
    assert "/models/x.litertlm" in services["serve"].argv
    assert "moonshine" in services["serve"].argv
    assert services["embed"].port == 9901 and "9901" in services["embed"].argv
    assert services["detect"].port == 9601 and "9601" in services["detect"].argv
    assert services["dash"].port == 8081
    assert "robot.local" in services["dash"].argv


def test_the_serve_readiness_line_follows_the_warmup():
    # serve.py binds its port BEFORE warming the models, so an open port would
    # call it ready while the first question still waits on the LLM.
    assert _by_name()["serve"].ready == "warm in"
    cold = _by_name(("--no-warmup",))["serve"]
    assert cold.ready == "model service on" and "--no-warmup" in cold.argv


def test_every_service_accepts_the_command_line_it_is_started_with():
    # The launcher and the services are separate programs: an argument one
    # stops accepting breaks the other only at start, in front of the room.
    from demo import embed_service, gpu_detect, serve
    from demo.display import web

    services = _by_name(("--llm", "/models/x.litertlm", "--no-warmup",
                         "--asr", "moonshine", "--asr-model", "base.en"))
    serve.parse_args(services["serve"].argv[4:])
    gpu_detect.parse_args(services["detect"].argv[4:])
    web.parse_args(services["dash"].argv[4:])
    import unittest.mock as mock

    with mock.patch.object(embed_service, "Embedders"), \
            mock.patch.object(embed_service, "ThreadingHTTPServer") as server:
        server.return_value.serve_forever.side_effect = KeyboardInterrupt
        try:
            embed_service.main(services["embed"].argv[4:])
        except KeyboardInterrupt:
            pass


def test_sim_runs_the_voice_loop_here_and_no_second_dashboard():
    # Without a robot the voice loop runs on this machine and serves the
    # dashboard itself, on this machine's camera: a standalone one would want
    # the same port. Nobody has to say --skip anything.
    from demo import run_demo
    from demo.stage import voice_loop_service

    args = _args("--sim", "--port", "9501", "--web-port", "8081", "--video", "1")
    assert [s.name for s in build_services(args)] == ["embed", "serve", "detect"]
    voice = voice_loop_service(args)
    parsed = run_demo.parse_args(voice.argv[4:])
    assert (parsed.brain, parsed.robot_host) == ("127.0.0.1", "127.0.0.1")
    assert (parsed.port, parsed.web_port, voice.port) == (9501, 8081, 8081)
    assert (parsed.video, parsed.audio) == ("1", "default")


def test_the_robot_is_the_emulator_with_sim_and_the_reachy_otherwise():
    assert _args("--sim").robot_host == "127.0.0.1"
    assert _args().robot_host == "reachy-mini.local"
    assert _args("--sim", "--robot-host", "10.0.0.5").robot_host == "10.0.0.5"
    with pytest.raises(SystemExit):
        _args("--sim", "--robot")


def test_what_was_added_last_stops_first_and_alone():
    # The voice loop saves the rest of the conversation on its way out, and
    # needs the embeddings it was started after still answering.
    services = [Service("embed", ["x"], 9900, "embed_service on")]
    processes = {"embed": _FakeProcess(_running(["embed_service on\n"])),
                 "voice": _FakeProcess(_running(["Speech threshold\n"]))}
    supervisor, sent, _out = _supervisor(services, processes)
    supervisor.start()
    supervisor.add(Service("voice", ["y"], 8091, "Speech threshold"))
    assert supervisor.wait_ready(timeout=2.0) == []
    supervisor.stop()
    assert [process for process, _sig in sent] == [processes["voice"], processes["embed"]]


def test_skip_leaves_a_service_out():
    names = [s.name for s in build_services(_args("--skip", "dash",
                                                  "--skip", "detect"))]
    assert names == ["embed", "serve"]


# — the port check —

def test_port_answers_sees_a_listening_socket():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert port_answers(port)
    finally:
        server.close()
    assert not port_answers(port)


def test_main_refuses_a_port_something_else_already_answers_on(capsys):
    # The failure this prevents: our service binds 0.0.0.0 and the other
    # process keeps 127.0.0.1, so the dashboard silently serves someone else.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        code = main(["--web-port", str(port), "--skip", "embed",
                     "--skip", "serve", "--skip", "detect"])
    finally:
        server.close()
    assert code == 1
    assert "already answers" in capsys.readouterr().out



def test_a_ctrl_c_while_the_models_load_stops_what_was_started(monkeypatch):
    # The services run in sessions of their own, so the terminal's Ctrl-C
    # never reaches them: left running, they hold their ports and the next
    # start refuses them.
    import demo.stage as stage

    stopped = []

    class Loading:
        def __init__(self, services):
            pass

        def start(self):
            pass

        def wait_ready(self):
            raise KeyboardInterrupt

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(stage, "Supervisor", Loading)
    monkeypatch.setattr(stage, "port_answers", lambda port: False)
    monkeypatch.setattr(stage, "_refresh_knowledge", lambda: None)
    monkeypatch.setattr(stage, "_stop_on_sigterm", lambda: None)
    monkeypatch.setattr(stage, "_finish_undisturbed", lambda: None)
    monkeypatch.setattr(stage, "_robot", lambda *a: pytest.fail("robot touched"))
    assert main(["--skip", "embed", "--skip", "serve", "--skip", "detect"]) == 0
    assert stopped == [True]

# — the supervisor —

class _FakeProcess:
    def __init__(self, lines=(), hang=False):
        self.stdout = iter(lines)
        self.pid = 4242
        self.returncode = None
        self._hang = hang

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired("service", timeout)
        self.returncode = 0
        return 0


def _running(lines, seconds=2.0):
    """A service that prints and then keeps running."""
    yield from lines
    time.sleep(seconds)


def _supervisor(services, processes):
    sent = []
    out = io.StringIO()
    supervisor = Supervisor(
        services, spawn=lambda service: processes[service.name],
        send_signal=lambda process, sig: sent.append((process, sig)),
        out=out, colour=False)
    return supervisor, sent, out


def test_a_service_is_ready_when_it_says_it_is():
    service = Service("embed", ["x"], 9900, "embed_service on")
    process = _FakeProcess(_running(["  loading models...\n",
                                     "embed_service on 0.0.0.0:9900\n"]))
    supervisor, _sent, out = _supervisor([service], {"embed": process})
    supervisor.start()
    assert supervisor.wait_ready(timeout=2.0) == []
    assert "embed | embed_service on 0.0.0.0:9900" in out.getvalue()


def test_a_service_that_never_announces_itself_is_reported():
    service = Service("serve", ["x"], 9500, "warm in")
    process = _FakeProcess(_running(["model service on 0.0.0.0:9500\n"]))
    supervisor, _sent, _out = _supervisor([service], {"serve": process})
    supervisor.start()
    assert supervisor.wait_ready(timeout=0.3) == ["serve"]


def test_a_service_that_exits_unblocks_the_wait():
    service = Service("serve", ["x"], 9500, "warm in")
    supervisor, _sent, _out = _supervisor([service],
                                          {"serve": _FakeProcess(["boom\n"])})
    supervisor.start()
    assert supervisor.wait(poll=0.05) == "serve"


def test_stop_asks_every_live_service_to_go_and_kills_what_stays():
    services = [Service("detect", ["x"], 9600, "gpu_detect on"),
                Service("dash", ["y"], 8080, "dashboard on")]
    stubborn = _FakeProcess(_running(["gpu_detect on :9600\n"]), hang=True)
    polite = _FakeProcess(_running(["dashboard on http://0.0.0.0:8080\n"]))
    supervisor, sent, _out = _supervisor(services, {"detect": stubborn,
                                                    "dash": polite})
    supervisor.start()
    supervisor.wait_ready(timeout=2.0)
    supervisor.stop()
    assert (stubborn, signal.SIGTERM) in sent and (stubborn, signal.SIGKILL) in sent
    assert (polite, signal.SIGTERM) in sent
    assert (polite, signal.SIGKILL) not in sent


# — the robot, when asked for —

def test_the_robot_is_told_the_brain_and_every_port_this_run_uses():
    # Including the dashboard's: the voice loop pushes its events there, and
    # the default would send them at a port nothing is listening on.
    env = robot_env(_args("--web-port", "8091", "--port", "9501",
                          "--robot-host", "robot.local"), "192.168.1.5")
    assert env["BRAIN"] == "192.168.1.5"
    assert env["ROBOT"] == "robot.local"
    assert env["BRAIN_PORT"] == "9501"
    assert env["WEB_PORT"] == "8091"
    assert env["EMBED_PORT"] == "9900" and env["GPU_DETECT_PORT"] == "9600"


def test_robot_commands_run_the_script_with_that_environment(monkeypatch):
    calls = []

    class _Result:
        returncode = 0

    monkeypatch.setattr("demo.stage.subprocess.run",
                        lambda argv, env, timeout=None: calls.append((argv, env)) or _Result())
    assert _robot("/repo/scripts/robot_service.sh", "voice-start",
                  {"BRAIN": "192.168.1.5", "ROBOT": "robot.local"},
                  out=io.StringIO())
    argv, env = calls[0]
    assert argv == ["/repo/scripts/robot_service.sh", "voice-start"]
    assert env["BRAIN"] == "192.168.1.5" and env["ROBOT"] == "robot.local"
    assert "PATH" in env, "the script still needs the inherited environment"


def test_a_failing_robot_command_is_reported_not_raised():
    class _Result:
        returncode = 3

    out = io.StringIO()
    import demo.stage as stage

    original = stage.subprocess.run
    stage.subprocess.run = lambda argv, env, timeout=None: _Result()
    try:
        assert _robot("/x.sh", "start", {"BRAIN": "b"}, out=out) is False
    finally:
        stage.subprocess.run = original
    assert "failed (3)" in out.getvalue()


def test_a_robot_command_that_hangs_is_given_up_on(monkeypatch):
    # The shutdown runs these with Ctrl-C ignored: a hung ssh there would
    # leave the launcher unkillable.
    import subprocess

    def hangs(argv, env, timeout=None):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr("demo.stage.subprocess.run", hangs)
    out = io.StringIO()
    assert _robot("/x.sh", "voice-stop", {}, out=out, timeout=1.0) is False
    assert "did not finish" in out.getvalue()


def test_sigterm_stops_the_services_like_ctrl_c_does():
    # `kill` on the launcher must not orphan four services on their ports.
    previous = signal.getsignal(signal.SIGTERM)
    try:
        _stop_on_sigterm()
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_a_service_that_dies_during_startup_is_reported_at_once():
    # The dashboard crashed on a NameError while serve was still loading its
    # models; waiting out the slow one before saying so wasted a minute.
    services = [Service("dash", ["x"], 8091, "dashboard on"),
                Service("serve", ["y"], 9500, "warm in")]
    crashed = _FakeProcess(["Traceback (most recent call last):\n",
                            "NameError: name 'X' is not defined\n"])
    slow = _FakeProcess(_running(["model service on 0.0.0.0:9500\n"], seconds=3.0))
    supervisor, _sent, out = _supervisor(services, {"dash": crashed, "serve": slow})
    supervisor.start()
    started = time.monotonic()
    late = supervisor.wait_ready(timeout=5.0)
    assert "dash" in late
    assert time.monotonic() - started < 2.0, "must not wait out the slow service"
    assert "NameError" in out.getvalue()


def test_the_launcher_rebuilds_the_knowledge_snapshot_only_when_stale(monkeypatch):
    from demo import knowledge, stage

    runs = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(stage.subprocess, "run",
                        lambda argv, check: runs.append(argv) or _Done())
    monkeypatch.setattr(knowledge, "is_stale", lambda: False)
    stage._refresh_knowledge()
    assert runs == []
    monkeypatch.setattr(knowledge, "is_stale", lambda: True)
    stage._refresh_knowledge()
    assert runs[0][-3:] == ["-m", "demo.knowledge", "build"]


def test_a_failed_rebuild_is_said_out_loud(monkeypatch, capsys):
    from demo import knowledge, stage

    class _Failed:
        returncode = 1

    monkeypatch.setattr(stage.subprocess, "run", lambda argv, check: _Failed())
    monkeypatch.setattr(knowledge, "is_stale", lambda: True)
    stage._refresh_knowledge()
    assert "old facts" in capsys.readouterr().out


def test_a_second_signal_during_shutdown_does_not_cut_it_short():
    from demo import stage

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        stage._finish_undisturbed()
        for sig in previous:
            signal.getsignal(sig)(sig, None)  # returns instead of raising
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# — the dashboard's Stream button, acted on here —

class _FakeRobotScript:
    """scripts/robot_service.sh, without a robot: records the commands and
    answers whether each one worked."""

    def __init__(self, fails=()):
        self.commands = []
        self._fails = set(fails)
        self.slept = []

    def run(self, script, command, env, out=None):
        self.commands.append(command)
        return command not in self._fails

    def sleep(self, host, out=None):
        self.slept.append(host)
        return True


def _watcher(dashboard_port, script, **kwargs):
    from demo.stage import StreamWatcher

    return StreamWatcher("127.0.0.1", dashboard_port, "/repo/robot_service.sh",
                         {"BRAIN": "192.168.1.5"}, "robot.local",
                         run=script.run, sleep_robot=script.sleep,
                         out=io.StringIO(), **kwargs)


def test_stream_on_starts_the_service_then_the_voice_loop():
    # voice-start is what enables the motors and plays wake_up (see
    # scripts/robot_service.sh) — and it refuses to run without a healthy
    # camera+mic service, which is why `start` comes first.
    script = _FakeRobotScript()
    watcher = _watcher(0, script)
    assert watcher.act(True) is True
    assert script.commands == ["start", "voice-start"]
    assert watcher.streaming() is True


def test_a_service_that_will_not_start_leaves_the_button_grey():
    # Saying "streaming" over a robot that never came up is worse than saying
    # nothing: the room reads the green button as a working demo.
    script = _FakeRobotScript(fails={"start"})
    watcher = _watcher(0, script)
    assert watcher.act(True) is False
    assert script.commands == ["start"], "voice-start needs the service"


def test_stream_off_stops_the_loop_then_the_service_then_lies_down():
    # goto_sleep last: the loop would otherwise keep moving the head after the
    # robot had lain down, and the camera must be released either way.
    script = _FakeRobotScript()
    watcher = _watcher(0, script, streaming=True)
    assert watcher.act(False) is False
    assert script.commands == ["voice-stop", "stop"]
    assert script.slept == ["robot.local"]


def test_a_robot_that_will_not_lie_down_still_counts_as_stopped():
    script = _FakeRobotScript()
    script.sleep = lambda host, out=None: False
    watcher = _watcher(0, script, streaming=True)
    assert watcher.act(False) is False


def _dashboard():
    """A real dashboard on a spare port — the flag travels over HTTP, which is
    the half of this that the stage cannot fake."""
    from demo.display.web import WebDashboard

    dash = WebDashboard(camera=None)
    server = dash.serve(host="127.0.0.1", port=0)
    return dash, server.server_address[1]


def test_the_button_travels_from_the_page_to_the_stage_and_back():
    """The whole flow: the page asks, this side polls the flag, runs the
    script, and acknowledges with the state the robot ended up in — so the
    request is spent and the button stops saying "Starting…"."""
    dash, port = _dashboard()
    script = _FakeRobotScript()
    watcher = _watcher(port, script, poll=0.05)
    try:
        watcher.start()
        # Starting tells the dashboard what is true before anyone presses
        # anything: without it the button offers to start a robot already up.
        assert _eventually(lambda: dash.is_streaming() is False)
        dash.request_stream(True)          # the page's POST /control
        assert _eventually(lambda: dash.is_streaming() is True)
        assert script.commands == ["start", "voice-start"]
        assert dash.stream_requested() is None, "the request must be spent"
        dash.request_stream(False)
        assert _eventually(lambda: dash.is_streaming() is False)
        assert script.commands == ["start", "voice-start", "voice-stop", "stop"]
        assert script.slept == ["robot.local"]
    finally:
        watcher.stop()
        dash.close()


def test_a_request_for_the_state_it_is_already_in_starts_nothing_twice():
    """A press that arrives while the robot is already up must be answered,
    not acted on — 15-20 s of wake-up on a robot mid-conversation."""
    dash, port = _dashboard()
    script = _FakeRobotScript()
    watcher = _watcher(port, script, streaming=True, poll=0.05)
    try:
        watcher.start()
        dash.request_stream(True)
        assert _eventually(lambda: dash.stream_requested() is None)
        assert script.commands == []
        assert dash.is_streaming() is True
    finally:
        watcher.stop()
        dash.close()


def _eventually(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_shutdown_waits_for_a_press_it_is_still_serving():
    """Ctrl-C two seconds after Stream on: without the wait, this thread's
    voice-start and the shutdown's voice-stop run in two ssh sessions at once
    and whichever lands last decides — a voice loop left holding the camera
    and the shard on a robot in another room."""
    import threading

    from demo.stage import StreamWatcher

    dash, port = _dashboard()
    gate = threading.Event()
    commands = []

    def slow_run(script, command, env, out=None):
        commands.append(command)
        if command == "voice-start":
            gate.wait(10)
        return True

    watcher = StreamWatcher("127.0.0.1", port, "/x.sh", {}, "robot.local",
                            run=slow_run, sleep_robot=lambda host, out=None: True,
                            poll=0.05, out=io.StringIO())
    stopped = threading.Event()
    try:
        watcher.start()
        dash.request_stream(True)
        assert _eventually(lambda: "voice-start" in commands)
        threading.Thread(target=lambda: (watcher.stop(), stopped.set()),
                         daemon=True).start()
        time.sleep(0.3)
        assert not stopped.is_set(), "the shutdown must wait for the press"
        gate.set()
        assert _eventually(stopped.is_set)
    finally:
        gate.set()
        dash.close()
