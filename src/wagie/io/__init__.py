"""I/O layer — Sources and Sinks.

Pattern: same Engine code with different concrete (Source, Sink, Clock) tuple
between prod and backtest.
"""

from wagie.core.stream import Sink, Source
from wagie.io.brokers import BinanceBroker, SimBroker
from wagie.io.clock import Clock, LiveClock, TestClock
from wagie.io.sources import BinanceLiveSource, ParquetReplaySource


__all__ = [
    "Source", "Sink",
    "Clock", "TestClock", "LiveClock",
    "ParquetReplaySource", "BinanceLiveSource",
    "SimBroker", "BinanceBroker",
]
