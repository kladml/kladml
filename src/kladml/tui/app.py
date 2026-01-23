
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from kladml.tui.screens import DashboardScreen

class KladMLApp(App):
    """KladML Terminal User Interface."""

    CSS = """
    #project-tree {
        width: 30%;
        dock: left;
        border-right: solid $primary;
    }
    #content-area {
        padding: 1;
    }
    #toolbar {
        height: 3;
        padding: 0 1;
        align: right middle;
        background: $surface;
        border-bottom: solid $primary-darken-2;
    }
    #help-text {
        width: 1fr;
        color: $text-muted;
    }
    #compare-btn {
        margin-left: 2;
    }
    DataTable {
        height: 100%;
    }
    #compare-container {
        padding: 1 2;
    }
    #compare-title {
        text-align: center;
        padding: 1;
        background: $primary;
        color: $text;
    }
    #compare-table {
        margin-top: 1;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())

def run_tui():
    app = KladMLApp()
    app.run()
