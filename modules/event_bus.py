#!/usr/bin/env python3
"""
Event bus implementation for decoupling components in the speech translation system.
"""

import asyncio
import functools
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus implementation that supports both synchronous and asynchronous callbacks.
    
    This class provides a central event hub where components can publish and subscribe
    to events without directly depending on each other, promoting loose coupling.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscribers = {}
        self._lock = threading.RLock()
        self._loop = None
        
    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for async operations."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def subscribe(self, event_type: str, callback: Callable) -> Callable:
        """
        Subscribe to an event type with a callback function.
        
        Args:
            event_type: The event type to subscribe to
            callback: The callback function to invoke when event occurs
            
        Returns:
            The unsubscribe function
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)
        
        def unsubscribe():
            with self._lock:
                if event_type in self._subscribers and callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
        
        return unsubscribe
    
    def publish(self, event_type: str, data: Optional[Any] = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: The type of event to publish
            data: Data to pass to subscribers
        """
        with self._lock:
            subscribers = set(self._subscribers.get(event_type, set()))
        
        if not subscribers:
            logger.debug(f"No subscribers for event {event_type}")
            return
            
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback to run in the event loop
                    loop = self.get_loop()
                    asyncio.run_coroutine_threadsafe(callback(event_type, data), loop)
                else:
                    # Run synchronous callback directly
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
                
    async def publish_async(self, event_type: str, data: Optional[Any] = None) -> None:
        """
        Publish an event to all subscribers asynchronously.
        
        Args:
            event_type: The type of event to publish
            data: Data to pass to subscribers
        """
        with self._lock:
            subscribers = set(self._subscribers.get(event_type, set()))
        
        if not subscribers:
            logger.debug(f"No subscribers for event {event_type}")
            return
            
        # Process synchronous callbacks in executor to avoid blocking
        sync_callbacks = [cb for cb in subscribers if not asyncio.iscoroutinefunction(cb)]
        async_callbacks = [cb for cb in subscribers if asyncio.iscoroutinefunction(cb)]
        
        # Schedule synchronous callbacks in executor
        loop = asyncio.get_running_loop()
        sync_tasks = [
            loop.run_in_executor(None, functools.partial(cb, event_type, data))
            for cb in sync_callbacks
        ]
        
        # Schedule asynchronous callbacks
        async_tasks = [cb(event_type, data) for cb in async_callbacks]
        
        # Wait for all callbacks to complete
        await asyncio.gather(*sync_tasks, *async_tasks, return_exceptions=True)

# Global event bus instance
event_bus = EventBus() 