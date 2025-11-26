"""
Tool Manager
============
Manages callable tools for the AI assistant.
Includes timer, callback, and extensible tool framework.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import json
import uuid

if TYPE_CHECKING:
    from main import SIPAIAssistant

from config import Config

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: ToolStatus
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class ScheduledTask:
    """A scheduled task (timer or callback)."""
    id: str
    task_type: str
    execute_at: datetime
    message: str
    target_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class BaseTool(ABC):
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    enabled: bool = True
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
        
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters. Return error message if invalid."""
        return None


class TimerTool(BaseTool):
    """
    Set timers and reminders.
    
    Parameters:
        - duration: Duration in seconds
        - message: Message to speak when timer completes
    """
    
    name = "SET_TIMER"
    description = "Set a timer or reminder"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        duration = params.get('duration', 300)  # Default 5 minutes
        message = params.get('message', 'Your timer is complete')
        
        # Validate duration
        max_duration = self.assistant.config.max_timer_duration_hours * 3600
        if duration > max_duration:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Timer duration exceeds maximum of {self.assistant.config.max_timer_duration_hours} hours"
            )
            
        if duration < 1:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Timer duration must be at least 1 second"
            )
            
        # Schedule the timer
        task_id = await self.assistant.tool_manager.schedule_task(
            task_type="timer",
            delay_seconds=duration,
            message=message,
            target_uri=None  # Timer plays on current call
        )
        
        logger.info(f"Timer set: {duration}s, message: {message}")
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Timer set for {self._format_duration(duration)}",
            data={"task_id": task_id, "duration": duration}
        )
        
    def _format_duration(self, seconds: int) -> str:
        """Format duration for speech."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if minutes:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            return " and ".join(parts)


class CallbackTool(BaseTool):
    """
    Schedule a callback call.
    
    Parameters:
        - delay: Delay in seconds before callback
        - message: Message to speak on callback
        - uri: SIP URI to call (optional, defaults to caller)
    """
    
    name = "CALLBACK"
    description = "Schedule a callback call"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        delay = params.get('delay', 60)
        message = params.get('message', 'This is your scheduled callback')
        uri = params.get('uri')
        
        # Get caller URI if not specified
        if not uri and self.assistant.current_call:
            uri = self.assistant.current_call.remote_uri
            
        if not uri:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="No callback number available"
            )
            
        # Schedule callback
        task_id = await self.assistant.tool_manager.schedule_task(
            task_type="callback",
            delay_seconds=delay,
            message=message,
            target_uri=uri
        )
        
        logger.info(f"Callback scheduled: {delay}s, uri: {uri}")
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"I'll call you back in {delay} seconds",
            data={"task_id": task_id, "delay": delay, "uri": uri}
        )


class HangupTool(BaseTool):
    """
    End the current call.
    """
    
    name = "HANGUP"
    description = "End the current call"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # The main loop handles actual hangup based on response
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message="Ending call"
        )


class StatusTool(BaseTool):
    """
    Get status of scheduled tasks.
    """
    
    name = "STATUS"
    description = "Check status of timers and callbacks"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pending = self.assistant.tool_manager.get_pending_tasks()
        
        if not pending:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message="You have no pending timers or callbacks"
            )
            
        messages = []
        for task in pending:
            remaining = (task.execute_at - datetime.now()).total_seconds()
            if remaining > 0:
                if task.task_type == "timer":
                    messages.append(f"Timer in {int(remaining)} seconds")
                else:
                    messages.append(f"Callback in {int(remaining)} seconds")
                    
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=". ".join(messages) if messages else "No pending tasks",
            data={"pending_count": len(pending)}
        )


class CancelTool(BaseTool):
    """
    Cancel scheduled tasks.
    
    Parameters:
        - task_type: Type to cancel ('timer', 'callback', or 'all')
    """
    
    name = "CANCEL"
    description = "Cancel timers or callbacks"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        task_type = params.get('task_type', 'all')
        
        cancelled = await self.assistant.tool_manager.cancel_tasks(task_type)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Cancelled {cancelled} {'task' if cancelled == 1 else 'tasks'}",
            data={"cancelled_count": cancelled}
        )


class ToolManager:
    """Manages all tools and scheduled tasks."""
    
    def __init__(self, assistant: 'SIPAIAssistant'):
        self.assistant = assistant
        self.config = assistant.config
        self.tools: Dict[str, BaseTool] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._task_runner: Optional[asyncio.Task] = None
        
        # Register default tools
        self._register_default_tools()
        
    def _register_default_tools(self):
        """Register built-in tools."""
        if self.config.enable_timer_tool:
            self.register_tool(TimerTool(self.assistant))
            
        if self.config.enable_callback_tool:
            self.register_tool(CallbackTool(self.assistant))
            
        # Always available
        self.register_tool(HangupTool(self.assistant))
        self.register_tool(StatusTool(self.assistant))
        self.register_tool(CancelTool(self.assistant))
        
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        
    def unregister_tool(self, name: str):
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            
    async def start(self):
        """Start the task runner."""
        self._task_runner = asyncio.create_task(self._run_scheduler())
        logger.info("Tool manager started")
        
    async def stop(self):
        """Stop the task runner."""
        if self._task_runner:
            self._task_runner.cancel()
            try:
                await self._task_runner
            except asyncio.CancelledError:
                pass
        logger.info("Tool manager stopped")
        
    async def execute_tool(self, tool_call) -> ToolResult:
        """Execute a tool call."""
        tool_name = tool_call.name.upper()
        
        if tool_name not in self.tools:
            logger.warning(f"Unknown tool: {tool_name}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Unknown tool: {tool_name}"
            )
            
        tool = self.tools[tool_name]
        
        if not tool.enabled:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Tool {tool_name} is currently disabled"
            )
            
        # Validate parameters
        error = tool.validate_params(tool_call.params)
        if error:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=error
            )
            
        # Execute
        try:
            result = await tool.execute(tool_call.params)
            logger.info(f"Tool {tool_name} executed: {result.status.value}")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Error executing {tool_name}: {str(e)}"
            )
            
    async def schedule_task(
        self,
        task_type: str,
        delay_seconds: int,
        message: str,
        target_uri: Optional[str] = None
    ) -> str:
        """Schedule a task for later execution."""
        task_id = str(uuid.uuid4())[:8]
        
        task = ScheduledTask(
            id=task_id,
            task_type=task_type,
            execute_at=datetime.now() + timedelta(seconds=delay_seconds),
            message=message,
            target_uri=target_uri
        )
        
        self.scheduled_tasks[task_id] = task
        logger.info(f"Scheduled task {task_id}: {task_type} in {delay_seconds}s")
        
        return task_id
        
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending (not completed) tasks."""
        now = datetime.now()
        return [
            task for task in self.scheduled_tasks.values()
            if not task.completed and task.execute_at > now
        ]
        
    async def cancel_tasks(self, task_type: str = 'all') -> int:
        """Cancel tasks by type. Returns number cancelled."""
        cancelled = 0
        to_remove = []
        
        for task_id, task in self.scheduled_tasks.items():
            if not task.completed:
                if task_type == 'all' or task.task_type == task_type:
                    to_remove.append(task_id)
                    cancelled += 1
                    
        for task_id in to_remove:
            del self.scheduled_tasks[task_id]
            
        return cancelled
        
    async def _run_scheduler(self):
        """Background task that executes scheduled tasks."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                now = datetime.now()
                
                for task_id, task in list(self.scheduled_tasks.items()):
                    if task.completed:
                        continue
                        
                    if now >= task.execute_at:
                        await self._execute_scheduled_task(task)
                        task.completed = True
                        
                # Cleanup old tasks
                self._cleanup_old_tasks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                
    async def _execute_scheduled_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        logger.info(f"Executing scheduled task: {task.id} ({task.task_type})")
        
        try:
            if task.task_type == "timer":
                await self._execute_timer(task)
            elif task.task_type == "callback":
                await self._execute_callback(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            
    async def _execute_timer(self, task: ScheduledTask):
        """Execute a timer - speak the message on current call."""
        if self.assistant.current_call and self.assistant.current_call.is_active:
            # Interrupt current conversation
            await self.assistant._speak(task.message)
        else:
            logger.warning(f"Timer {task.id} expired but no active call")
            
    async def _execute_callback(self, task: ScheduledTask):
        """Execute a callback - make outbound call."""
        if not task.target_uri:
            logger.error(f"Callback {task.id} has no target URI")
            return
            
        # Make the call
        for attempt in range(self.config.callback_retry_attempts):
            try:
                await self.assistant.make_outbound_call(
                    task.target_uri,
                    task.message
                )
                logger.info(f"Callback {task.id} completed")
                return
            except Exception as e:
                logger.warning(f"Callback attempt {attempt + 1} failed: {e}")
                if attempt < self.config.callback_retry_attempts - 1:
                    await asyncio.sleep(self.config.callback_retry_delay_s)
                    
        logger.error(f"Callback {task.id} failed after {self.config.callback_retry_attempts} attempts")
        
    def _cleanup_old_tasks(self):
        """Remove completed tasks older than 1 hour."""
        cutoff = datetime.now() - timedelta(hours=1)
        to_remove = [
            task_id for task_id, task in self.scheduled_tasks.items()
            if task.completed and task.execute_at < cutoff
        ]
        for task_id in to_remove:
            del self.scheduled_tasks[task_id]


# Convenience function for creating custom tools
def create_custom_tool(
    name: str,
    description: str,
    handler: Callable,
    assistant: 'SIPAIAssistant'
) -> BaseTool:
    """
    Factory function to create custom tools.
    
    Example:
        async def weather_handler(params):
            city = params.get('city', 'unknown')
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=f"The weather in {city} is sunny"
            )
            
        weather_tool = create_custom_tool(
            "WEATHER",
            "Get weather information",
            weather_handler,
            assistant
        )
        assistant.tool_manager.register_tool(weather_tool)
    """
    
    class CustomTool(BaseTool):
        def __init__(self, name: str, description: str, handler: Callable, assistant):
            super().__init__(assistant)
            self.name = name
            self.description = description
            self._handler = handler
            
        async def execute(self, params: Dict[str, Any]) -> ToolResult:
            return await self._handler(params)
            
    return CustomTool(name, description, handler, assistant)
