"""DataSource concretes.

ParquetReplaySource yields wagie.core.DecisionBar events with their
underlying minute-bars attached as `underlying`.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import polars as pl

from wagie.core.event import DecisionBar, MinuteBar
from wagie.core.identity import DEFAULT_INSTRUMENT, InstrumentId
from wagie.core.numeric import Price, Quantity
from wagie.core.time import Duration, Timestamp


logger = logging.getLogger(__name__)


class ParquetReplaySource:
    """Reads 1m parquet, aggregates to M-min DecisionBar."""

    REQUIRED_COLS = (
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "segment_id",
    )

    def __init__(
        self,
        parquet_path: str,
        m_minutes: int = 20,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
        instrument: InstrumentId = DEFAULT_INSTRUMENT,
    ):
        self.parquet_path = str(parquet_path)
        self.m_minutes = int(m_minutes)
        self.start_ts_ms = start_ts_ms
        self.end_ts_ms = end_ts_ms
        self.instrument = instrument
        self._df: Optional[pl.DataFrame] = None

    def _ensure_loaded(self) -> None:
        if self._df is not None:
            return
        df = pl.read_parquet(self.parquet_path)
        missing = set(self.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"parquet missing columns: {missing}")
        if self.start_ts_ms is not None:
            df = df.filter(pl.col("open_time") >= self.start_ts_ms)
        if self.end_ts_ms is not None:
            df = df.filter(pl.col("open_time") <= self.end_ts_ms)
        self._df = df.sort("open_time")

    def stream(self) -> Iterator[DecisionBar]:
        self._ensure_loaded()
        assert self._df is not None
        rows = self._df.iter_rows(named=True)

        buf: list[MinuteBar] = []
        prev_open_time: Optional[int] = None
        prev_segment: Optional[int] = None

        for row in rows:
            try:
                mb = MinuteBar(
                    ts_init=Timestamp.from_ms(int(row["open_time"])),
                    instrument=self.instrument,
                    open=Price(float(row["open"])),
                    high=Price(float(row["high"])),
                    low=Price(float(row["low"])),
                    close=Price(float(row["close"])),
                    volume=Quantity(float(row["volume"])),
                    quote_volume=float(row.get("quote_volume", 0.0)),
                    trades=int(row.get("trades", 0)),
                    taker_buy_base=float(row.get("taker_buy_base", 0.0)),
                    taker_buy_quote=float(row.get("taker_buy_quote", 0.0)),
                    segment_id=int(row.get("segment_id", 0)),
                    ts_close=Timestamp.from_ms(int(row["close_time"])),
                )
            except Exception as e:
                logger.warning(f"skip invalid row: {e}")
                continue

            ot = mb.ts_init.ms
            if prev_segment is not None and mb.segment_id != prev_segment:
                buf.clear()
            elif prev_open_time is not None and ot - prev_open_time > 60_000 + 1000:
                buf.clear()
            buf.append(mb)
            prev_open_time = ot
            prev_segment = mb.segment_id

            if len(buf) == self.m_minutes:
                first, last = buf[0], buf[-1]
                bar = DecisionBar(
                    ts_init=first.ts_init,
                    instrument=self.instrument,
                    open=first.open,
                    high=Price(max(float(b.high) for b in buf)),
                    low=Price(min(float(b.low) for b in buf)),
                    close=last.close,
                    volume=Quantity(sum(float(b.volume) for b in buf)),
                    duration=Duration.from_minutes(self.m_minutes),
                    underlying=tuple(buf),
                    quote_volume=sum(b.quote_volume for b in buf),
                    trades=sum(b.trades for b in buf),
                    taker_buy_base=sum(b.taker_buy_base for b in buf),
                    taker_buy_quote=sum(b.taker_buy_quote for b in buf),
                    segment_id=first.segment_id,
                )
                yield bar
                buf.clear()

    def close(self) -> None:
        self._df = None


class BinanceLiveSource:
    """STUB. Implement when prod path is needed."""

    def __init__(self, symbol: str = "BTCUSDT", m_minutes: int = 20):
        self.symbol = symbol
        self.m_minutes = m_minutes

    def stream(self) -> Iterator[DecisionBar]:  # pragma: no cover
        raise NotImplementedError("BinanceLiveSource is a stub")

    def close(self) -> None:
        pass


__all__ = ["ParquetReplaySource", "BinanceLiveSource"]
