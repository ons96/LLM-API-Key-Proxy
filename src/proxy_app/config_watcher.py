"""Config Watcher - Auto-restart on model ranking changes

Monitors model_rankings.yaml for changes and triggers smart restart.
Only restarts during inactive periods to avoid disrupting users.
"""

import time
import os
import signal
import hashlib
from pathlib import Path
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Watches config files and triggers callback on changes."""
    
    def __init__(self, 
                 config_path: str = "config/model_rankings.yaml",
                 check_interval: int = 60,
                 inactivity_threshold: int = 300):
        self.config_path = Path(config_path)
        self.check_interval = check_interval
        self.inactivity_threshold = inactivity_threshold
        self.last_config_hash: Optional[str] = None
        self.last_activity_time: float = time.time()
        self._running: bool = False
        self._restart_callback: Optional[Callable] = None
        
    def set_restart_callback(self, callback: Callable):
        """Set callback to trigger on config change + inactivity."""
        self._restart_callback = callback
        
    def record_activity(self):
        """Record user activity (API call processed)."""
        self.last_activity_time = time.time()
        
    def _get_file_hash(self) -> str:
        """Get MD5 hash of config file."""
        if not self.config_path.exists():
            return ""
        
        with open(self.config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def _check_for_changes(self) -> bool:
        """Check if config file has changed."""
        current_hash = self._get_file_hash()
        
        if self.last_config_hash is None:
            self.last_config_hash = current_hash
            return False
            
        if current_hash != self.last_config_hash:
            self.last_config_hash = current_hash
            return True
            
        return False
        
    def _is_inactive(self) -> bool:
        """Check if gateway has been inactive."""
        time_since_activity = time.time() - self.last_activity_time
        return time_since_activity >= self.inactivity_threshold
        
    def start_watching(self):
        """Start watching for config changes."""
        self._running = True
        logger.info(f"Config watcher started (checking every {self.check_interval}s)")
        
        # Initial hash
        self.last_config_hash = self._get_file_hash()
        
        while self._running:
            try:
                if self._check_for_changes():
                    logger.info("Config change detected!")
                    
                    # Wait for inactivity period
                    while not self._is_inactive():
                        logger.debug("Waiting for inactivity period...")
                        time.sleep(10)
                        
                    # Trigger restart
                    logger.info("Triggering smart restart (inactive period)")
                    if self._restart_callback:
                        try:
                            self._restart_callback()
                        except Exception as e:
                            logger.error(f"Restart callback failed: {e}")
                            
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Config watcher error: {e}")
                time.sleep(60)
                
    def stop_watching(self):
        """Stop the config watcher."""
        self._running = False
        logger.info("Config watcher stopped")


def create_auto_restart_watcher():
    """Create and return a config watcher for auto-restart."""
    def restart_gateway():
        """Trigger graceful restart of gateway."""
        logger.info("Initiating graceful gateway restart...")
        # Send SIGTERM to self - systemd/supervisord will restart
        os.kill(os.getpid(), signal.SIGTERM)
        
    watcher = ConfigWatcher()
    watcher.set_restart_callback(restart_gateway)
    return watcher
