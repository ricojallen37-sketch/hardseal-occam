from solver.events import EventEmitter, Event

def test_emitter_calls_callback():
    received = []
    emitter = EventEmitter(callback=lambda e: received.append(e))
    emitter.emit(Event(type="test", data={"value": 42}))
    assert len(received) == 1
    assert received[0]["type"] == "test"
    assert received[0]["data"]["value"] == 42

def test_emitter_no_callback_is_noop():
    emitter = EventEmitter(callback=None)
    emitter.emit(Event(type="test", data={}))  # should not raise

def test_emitter_adds_timestamp():
    received = []
    emitter = EventEmitter(callback=lambda e: received.append(e))
    emitter.emit(Event(type="test", data={}))
    assert "timestamp" in received[0]

def test_event_types():
    """Verify all expected event types are importable."""
    from solver.events import (
        BenchmarkStartEvent, GameStartEvent, GameCompleteEvent,
        PhaseChangeEvent, ProbeEvent, StateDiscoveredEvent,
        BfsStepEvent, ResetEvent, LevelSolvedEvent, LevelFailedEvent,
        FrameDiffEvent, BenchmarkCompleteEvent,
    )
