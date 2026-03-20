"""Data ingestion: loaders for CSV, FRED, Yahoo Finance, and caching."""

from correlation_engine.ingest.base import BaseLoader
from correlation_engine.ingest.cache import DataCache
from correlation_engine.ingest.csv_loader import CsvLoader
from correlation_engine.ingest.fred import FredLoader
from correlation_engine.ingest.yahoo import YahooLoader

__all__ = ["BaseLoader", "CsvLoader", "FredLoader", "YahooLoader", "DataCache"]
