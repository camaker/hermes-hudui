"""CLI-based chat engine using hermes subprocess."""

from __future__ import annotations

import fcntl
import os
import pty
import select
import subprocess
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from .models import (
    ChatSession,
    ComposerState,
    StreamingEvent,
)
from .streamer import ChatStreamer


class ChatNotAvailableError(Exception):
    """Raised when chat functionality is not available."""

    pass


class CLISession:
    """Manages a single hermes CLI session via PTY."""

    def __init__(self, session_id: str, profile: Optional[str] = None):
        self.session_id = session_id
        self.profile = profile
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self._buffer = ""
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[str], None]] = []

    def start(self) -> bool:
        """Start hermes chat session in PTY."""
        try:
            # Create pseudo-terminal
            self.master_fd, self.slave_fd = pty.openpty()

            # Make master non-blocking
            flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            # Build command
            cmd = ["hermes", "chat"]
            if self.profile:
                cmd.extend(["--profile", self.profile])

            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                cwd=os.path.expanduser("~"),
                start_new_session=True,
            )

            # Close slave in parent
            os.close(self.slave_fd)
            self.slave_fd = None

            self._running = True

            # Start reader thread
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            return True

        except Exception as e:
            self.cleanup()
            raise ChatNotAvailableError(f"Failed to start CLI session: {e}")

    def _read_loop(self) -> None:
        """Read output from PTY."""
        while self._running and self.master_fd is not None:
            try:
                # Check if data available
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                if ready:
                    data = os.read(self.master_fd, 4096).decode(
                        "utf-8", errors="replace"
                    )
                    if data:
                        self._buffer += data
                        # Notify callbacks
                        for cb in self._callbacks:
                            try:
                                cb(data)
                            except Exception:
                                pass
            except (OSError, IOError):
                break
            except Exception:
                continue

    def send(self, message: str) -> bool:
        """Send message to CLI."""
        if not self._running or self.master_fd is None:
            return False

        try:
            # Write message + newline
            os.write(self.master_fd, (message + "\n").encode("utf-8"))
            return True
        except Exception:
            return False

    def on_output(self, callback: Callable[[str], None]) -> None:
        """Register output callback."""
        self._callbacks.append(callback)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._running = False

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None

        if self.slave_fd is not None:
            try:
                os.close(self.slave_fd)
            except Exception:
                pass
            self.slave_fd = None


class ChatEngine:
    """Main chat engine using CLI subprocess."""

    _instance: Optional["ChatEngine"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChatEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: dict[str, ChatSession] = {}
        self._cli_sessions: dict[str, CLISession] = {}
        self._streamers: dict[str, ChatStreamer] = {}
        self._initialized = True
        self._cli_available = self._check_cli()

    @staticmethod
    def _check_cli() -> bool:
        """Check if hermes CLI is available."""
        try:
            result = subprocess.run(
                ["hermes", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if chat is available."""
        return self._cli_available

    def create_session(
        self, profile: Optional[str] = None, model: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        if not self._cli_available:
            raise ChatNotAvailableError(
                "Hermes CLI not available. Install hermes-agent: pip install hermes-agent"
            )

        session_id = str(uuid.uuid4())[:8]

        # Create CLI session
        cli_session = CLISession(session_id, profile)
        if not cli_session.start():
            raise ChatNotAvailableError("Failed to start CLI session")

        self._cli_sessions[session_id] = cli_session

        session = ChatSession(
            id=session_id,
            profile=profile,
            model=model,
            title=f"Chat {session_id}",
            backend_type="cli-pty",
        )
        self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[ChatSession]:
        """List all active sessions."""
        return list(self._sessions.values())

    def end_session(self, session_id: str) -> bool:
        """End a chat session."""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False

            # Cleanup CLI session
            if session_id in self._cli_sessions:
                self._cli_sessions[session_id].cleanup()
                del self._cli_sessions[session_id]

            # Cleanup streamer
            if session_id in self._streamers:
                self._streamers[session_id].stop()
                del self._streamers[session_id]

            return True
        return False

    def send_message(
        self,
        session_id: str,
        content: str,
    ) -> ChatStreamer:
        """Send a message and return streamer for responses."""
        session = self._sessions.get(session_id)
        if not session:
            raise ChatNotAvailableError(f"Session {session_id} not found")

        if not session.is_active:
            raise ChatNotAvailableError(f"Session {session_id} is inactive")

        cli_session = self._cli_sessions.get(session_id)
        if not cli_session:
            raise ChatNotAvailableError("CLI session not found")

        streamer = ChatStreamer()
        self._streamers[session_id] = streamer

        # Update session stats
        session.message_count += 1
        session.last_activity = datetime.now()

        # Setup output handler
        def handle_output(data: str) -> None:
            # Parse and stream output
            # For now, just emit raw tokens
            # TODO: Parse structured output from CLI
            for char in data:
                streamer.emit_token(char)

        cli_session.on_output(handle_output)

        # Send message
        if cli_session.send(content):
            # Mark done after a delay (since we can't detect end easily)
            def delayed_done():
                import time

                time.sleep(0.5)  # Small delay to collect output
                streamer.emit_done()

            threading.Thread(target=delayed_done, daemon=True).start()
        else:
            streamer.emit_error("Failed to send message")

        return streamer

    def get_composer_state(self, session_id: str) -> ComposerState:
        """Get current composer state for UI."""
        session = self._sessions.get(session_id)
        if not session:
            return ComposerState(model="unknown")

        return ComposerState(
            model=session.model or "claude-4-sonnet",
            is_streaming=session_id in self._streamers,
            context_tokens=0,  # Would need to parse from CLI output
        )

    def cleanup_all(self) -> None:
        """Clean up all sessions."""
        for session_id in list(self._sessions.keys()):
            self.end_session(session_id)


# Global engine instance
chat_engine = ChatEngine()
