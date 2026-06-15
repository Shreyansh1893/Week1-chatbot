"""
app.py

Textual TUI for the research agent. Wraps ResearchAgent (agent.py) in a
chat-like interface: type a question, watch tool calls stream in as they
happen, then see the final cited answer rendered as Markdown.

Run with:  python app.py
"""

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Header, Footer, Input, Static, Markdown, LoadingIndicator
from textual.reactive import reactive

from agent import ResearchAgent


class ToolCallLine(Static):
    """A single line showing a tool call and (once available) its result."""

    def __init__(self, name: str, args: dict):
        super().__init__()
        self.tool_name = name
        self.args = args
        self.result_preview = None

    def render(self) -> str:
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        line = f"[dim]→[/dim] [bold cyan]{self.tool_name}[/bold cyan]({args_str})"
        if self.result_preview is not None:
            line += f"\n  [dim]{self.result_preview}[/dim]"
        return line

    def set_result(self, result: str):
        preview = result.strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        self.result_preview = preview
        self.refresh()


class ChatMessage(Static):
    """A user or assistant chat bubble."""

    def __init__(self, role: str, content: str):
        css_class = "user-message" if role == "user" else "assistant-message"
        super().__init__(classes=css_class)
        self.role = role
        self._content = content

    def compose(self) -> ComposeResult:
        if self.role == "user":
            yield Static(f"[bold]You:[/bold] {self._content}")
        else:
            yield Markdown(self._content)


class ResearchApp(App):
    """A Textual TUI for the build-your-own-Perplexity research agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        padding: 1 2;
    }

    .user-message {
        margin: 1 0;
        padding: 0 1;
    }

    .assistant-message {
        margin: 1 0;
        padding: 1;
        border: round $accent;
    }

    ToolCallLine {
        margin: 0 0 0 2;
        color: $text-muted;
    }

    #input-row {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }

    #question-input {
        width: 1fr;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
    ]

    busy = reactive(False)

    def __init__(self):
        super().__init__()
        self.agent = ResearchAgent(on_event=self._handle_agent_event)
        self._current_tool_widgets = {}
        self._tool_counter = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-log")
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="Ask a research question and press Enter...",
                id="question-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()
        self._log(
            Static(
                "[bold]Build-your-own-Perplexity[/bold]\n"
                "Ask a research question. The agent will search the web, "
                "read pages, and (for technical topics) look up papers via "
                "AlphaXiv before answering.\n"
            )
        )

    def _log(self, widget) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self.busy:
            return

        input_widget = self.query_one("#question-input", Input)
        input_widget.value = ""
        input_widget.disabled = True
        self.busy = True

        self._log(ChatMessage("user", question))

        loading = LoadingIndicator(id="loading")
        self._log(loading)

        try:
            answer = await self.agent.ask(question)
        except Exception as e:  # noqa: BLE001
            answer = f"**Error:** {e}"
        finally:
            loading.remove()
            self.busy = False
            input_widget.disabled = False
            input_widget.focus()

        self._log(ChatMessage("assistant", answer))

    def _handle_agent_event(self, event_type: str, data: dict) -> None:
        """Called from the agent's async loop (via call_from_thread-safe
        Textual scheduling) to surface tool calls/results live."""

        if event_type == "tool_call":
            self._tool_counter += 1
            key = self._tool_counter
            widget = ToolCallLine(data["name"], data["args"])
            self._current_tool_widgets[key] = (widget, data["name"], data["args"])
            self.call_from_thread(self._log, widget) if self._not_in_loop() else self._log(widget)
            self._pending_tool_key = key

        elif event_type == "tool_result":
            key = getattr(self, "_pending_tool_key", None)
            if key is not None and key in self._current_tool_widgets:
                widget, _, _ = self._current_tool_widgets[key]
                widget.set_result(str(data["result"]))

        elif event_type == "warning":
            self._log(Static(f"[yellow]Warning:[/yellow] {data['message']}"))

    def _not_in_loop(self) -> bool:
        # _handle_agent_event is invoked from within the same asyncio loop
        # as the Textual app (ResearchAgent.ask is awaited directly from
        # on_input_submitted), so we can mutate the UI directly. This helper
        # exists for clarity/future-proofing if that ever changes.
        return False

    def action_clear(self) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.remove_children()


if __name__ == "__main__":
    ResearchApp().run()