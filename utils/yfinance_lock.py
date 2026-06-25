"""
utils/yfinance_lock.py — Global shared lock for yfinance.

yfinance is NOT thread-safe: concurrent downloads from different modules
(technical_agent, vol_profile_agent, entry_layer) cause data mixing.
Example: MGC receives EURCHF prices (0.92 instead of 3200).

All modules using yf.download() must import this lock and serialize
downloads with:
    with YF_DOWNLOAD_LOCK:
        df = yf.download(...)
"""
import threading

YF_DOWNLOAD_LOCK = threading.Lock()
