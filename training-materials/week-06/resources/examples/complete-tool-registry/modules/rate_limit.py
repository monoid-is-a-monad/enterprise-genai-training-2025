"""Rate limiting implementation."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict


class QuotaManager:
    """
    Manages rate limiting using token bucket algorithm.
    
    Features:
    - Per-user quotas
    - Time-window tracking
    - Automatic reset
    """
    
    def __init__(self, default_quota: int = 100, window_minutes: int = 60):
        self._default_quota = default_quota
        self._window_minutes = window_minutes
        self._user_quotas: Dict[str, int] = defaultdict(lambda: default_quota)
        self._user_windows: Dict[str, datetime] = {}
    
    def _reset_if_needed(self, user_id: str) -> None:
        """Reset quota if window expired."""
        now = datetime.now()
        
        if user_id not in self._user_windows:
            self._user_windows[user_id] = now
            self._user_quotas[user_id] = self._default_quota
            return
        
        window_start = self._user_windows[user_id]
        if now - window_start >= timedelta(minutes=self._window_minutes):
            # Reset window
            self._user_windows[user_id] = now
            self._user_quotas[user_id] = self._default_quota
    
    def check_quota(self, user_id: str, cost: int = 1) -> bool:
        """
        Check if user has enough quota.
        
        Args:
            user_id: User identifier
            cost: Cost of operation
            
        Returns:
            True if quota available
        """
        self._reset_if_needed(user_id)
        return self._user_quotas[user_id] >= cost
    
    def consume_quota(self, user_id: str, cost: int = 1) -> None:
        """Consume quota for user."""
        self._reset_if_needed(user_id)
        self._user_quotas[user_id] -= cost
    
    def get_remaining_quota(self, user_id: str) -> int:
        """Get remaining quota for user."""
        self._reset_if_needed(user_id)
        return max(0, self._user_quotas[user_id])
    
    def get_quota_info(self, user_id: str) -> dict:
        """Get quota information for user."""
        self._reset_if_needed(user_id)
        
        used = self._default_quota - self._user_quotas[user_id]
        
        return {
            "user_id": user_id,
            "limit": self._default_quota,
            "used": used,
            "remaining": self._user_quotas[user_id],
            "window_minutes": self._window_minutes,
            "reset_at": self._user_windows.get(user_id, datetime.now()) + 
                       timedelta(minutes=self._window_minutes)
        }
