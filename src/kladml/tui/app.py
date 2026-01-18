
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
    DataTable {
        height: 100%;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())

def run_tui():
    app = KladMLApp()
    app.run()
