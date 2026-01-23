
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, DataTable, Tree, Static, Button
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label
from textual.binding import Binding

from kladml.backends import get_metadata_backend, LocalTracker
from datetime import datetime
from typing import Set, List


class DashboardScreen(Screen):
    """Main dashboard screen with multi-select for run comparison."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("space", "toggle_select", "Select/Deselect"),
        Binding("c", "compare", "Compare Selected"),
        Binding("escape", "clear_selection", "Clear Selection"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_runs: Set[str] = set()
        self.current_experiment: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("KladML Workspace", id="project-tree")
            with Vertical(id="content-area"):
                with Horizontal(id="toolbar"):
                    yield Static("Select runs with [bold]Space[/bold], then press [bold]C[/bold] to compare", id="help-text")
                    yield Button("Compare Selected", id="compare-btn", variant="primary", disabled=True)
                yield DataTable(id="data-table")
        yield Footer()

    def on_mount(self) -> None:
        self.load_tree()

    def load_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.root.expand()
        
        metadata = get_metadata_backend()
        
        # 1. Projects Node
        projects_root = tree.root.add("Projects", expand=True)
        projects = metadata.list_projects()
        
        for proj in projects:
            proj_node = projects_root.add(f"📁 {proj.name}", data={"type": "project", "name": proj.name})
            families = metadata.list_families(proj.name)
            for fam in families:
                fam_node = proj_node.add(f"📂 {fam.name}", data={"type": "family", "name": fam.name, "project": proj.name})
                experiments = fam.experiment_names or []
                for exp in experiments:
                    fam_node.add(f"🧪 {exp}", data={"type": "experiment", "name": exp, "project": proj.name, "family": fam.name})

        # 2. Datasets Node
        datasets_root = tree.root.add("Datasets", expand=True)
        try:
            datasets = metadata.list_datasets()
            for ds in datasets:
                datasets_root.add(f"💾 {ds.name}", data={"type": "dataset", "name": ds.name, "path": ds.path, "desc": ds.description})
        except Exception:
            pass
            
        # 3. Configs Node (Filesystem)
        configs_root = tree.root.add("Configs", expand=True)
        from pathlib import Path
        config_dir = Path("data/configs")
        if config_dir.exists():
            for cfg in config_dir.glob("*.yaml"):
                configs_root.add(f"⚙️ {cfg.name}", data={"type": "config", "name": cfg.name, "path": str(cfg)})

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if not data:
            return
            
        t = data.get("type")
        if t == "experiment":
            self.current_experiment = data["name"]
            self.selected_runs.clear()
            self.update_compare_button()
            self.show_experiment_runs(data["name"])
        elif t == "project":
            self.show_project_info(data["name"])
        elif t == "dataset":
            self.show_dataset_info(data)
        elif t == "config":
            self.show_config_info(data)
            
    def show_dataset_info(self, data: dict) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Property", "Value")
        table.cursor_type = "row"
        
        table.add_row("Name", data["name"])
        table.add_row("Path", data["path"])
        table.add_row("Description", data.get("desc", "-"))
        
        from pathlib import Path
        p = Path(data["path"])
        if p.exists():
            table.add_row("Contents", ", ".join([x.name for x in p.iterdir() if x.is_dir()]))

    def show_config_info(self, data: dict) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Line", "Content")
        table.cursor_type = "row"
        
        path = data["path"]
        try:
            with open(path) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                 table.add_row(str(i+1), line.rstrip())
        except Exception as e:
            table.add_row("Error", str(e))

    def show_experiment_runs(self, experiment_name: str) -> None:
        tracker = LocalTracker()
        exp = tracker.get_experiment_by_name(experiment_name)
        
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("✓", "Run ID", "Name", "Status", "Metrics")
        table.cursor_type = "row"
        
        if not exp:
            return

        runs = tracker.search_runs(exp["id"], max_results=50)
        for run in runs:
            metrics = run.get("metrics", {})
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in list(metrics.items())[:3])
            
            run_id = run["run_id"]
            selected_mark = "✅" if run_id in self.selected_runs else "⬜"
            
            table.add_row(
                selected_mark,
                run_id[:8],
                run.get("run_name", "-"),
                run["status"],
                metrics_str,
                key=run_id
            )

    def action_toggle_select(self) -> None:
        """Toggle selection of the currently highlighted row."""
        table = self.query_one(DataTable)
        if table.cursor_row is None:
            return
            
        try:
            row_key = table.get_row_at(table.cursor_row)
            run_id = str(table.coordinate_to_cell_key((table.cursor_row, 0)).row_key)
        except Exception:
            return
            
        if run_id in self.selected_runs:
            self.selected_runs.discard(run_id)
        else:
            self.selected_runs.add(run_id)
        
        # Update the checkbox column
        mark = "✅" if run_id in self.selected_runs else "⬜"
        table.update_cell_at((table.cursor_row, 0), mark)
        
        self.update_compare_button()

    def action_compare(self) -> None:
        """Open compare screen with selected runs."""
        if len(self.selected_runs) < 2:
            self.notify("Select at least 2 runs to compare", severity="warning")
            return
        self.app.push_screen(CompareScreen(list(self.selected_runs)))

    def action_clear_selection(self) -> None:
        """Clear all selected runs."""
        self.selected_runs.clear()
        # Refresh the table to clear checkmarks
        if self.current_experiment:
            self.show_experiment_runs(self.current_experiment)
        self.update_compare_button()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare-btn":
            self.action_compare()

    def update_compare_button(self) -> None:
        """Enable/disable compare button based on selection count."""
        btn = self.query_one("#compare-btn", Button)
        count = len(self.selected_runs)
        btn.disabled = count < 2
        btn.label = f"Compare Selected ({count})" if count > 0 else "Compare Selected"
            
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Double-click or Enter opens run details."""
        run_id = event.row_key.value
        if run_id:
            self.app.push_screen(RunDetailScreen(run_id))

    def show_project_info(self, project_name: str) -> None:
        pass


class RunDetailScreen(Screen):
    """Screen for viewing detailed run information."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="detail-container"):
            yield Label(f"Run Details: {self.run_id}", id="detail-title")
            with Horizontal():
                yield Vertical(id="params-col")
                yield Vertical(id="metrics-col")
        yield Footer()

    def on_mount(self) -> None:
        tracker = LocalTracker()
        run = tracker.get_run(self.run_id)
        if not run:
            self.query_one("#detail-title").update("Run not found")
            return

        params_col = self.query_one("#params-col")
        params_col.mount(Label("[bold]Parameters[/bold]"))
        params = run.get("params", {})
        if not params:
            params_col.mount(Label("No parameters logged"))
        else:
            for k, v in params.items():
                params_col.mount(Label(f"[cyan]{k}[/cyan]: {v}"))

        metrics_col = self.query_one("#metrics-col")
        metrics_col.mount(Label("[bold]Metrics[/bold]"))
        metrics = run.get("metrics", {})
        if not metrics:
            metrics_col.mount(Label("No metrics logged"))
        else:
            for k, v in metrics.items():
                metrics_col.mount(Label(f"[green]{k}[/green]: {v}"))


class CompareScreen(Screen):
    """Screen for comparing multiple runs side-by-side."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, run_ids: List[str]):
        super().__init__()
        self.run_ids = run_ids

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="compare-container"):
            yield Label(f"[bold]Comparing {len(self.run_ids)} Runs[/bold]", id="compare-title")
            yield DataTable(id="compare-table")
        yield Footer()

    def on_mount(self) -> None:
        tracker = LocalTracker()
        table = self.query_one("#compare-table", DataTable)
        table.cursor_type = "row"
        
        # Fetch all runs
        runs_data = []
        for rid in self.run_ids:
            run = tracker.get_run(rid)
            if run:
                runs_data.append(run)
        
        if not runs_data:
            table.add_column("Error")
            table.add_row("No runs found")
            return
        
        # Build columns: Property | Run1 | Run2 | ...
        table.add_column("Property", key="prop")
        for run in runs_data:
            name = run.get("run_name", run["run_id"][:8])
            table.add_column(f"{name}\n[dim]{run['run_id'][:8]}[/dim]", key=run["run_id"])
        
        # Collect all metric and param keys
        all_metrics = set()
        all_params = set()
        for run in runs_data:
            all_metrics.update(run.get("metrics", {}).keys())
            all_params.update(run.get("params", {}).keys())
        
        # Section: Status
        table.add_row("── Status ──", *["" for _ in runs_data])
        status_row = ["status"]
        for run in runs_data:
            status_row.append(run.get("status", "-"))
        table.add_row(*status_row)
        
        # Section: Metrics
        table.add_row("── Metrics ──", *["" for _ in runs_data])
        for m in sorted(all_metrics):
            row = [m]
            for run in runs_data:
                val = run.get("metrics", {}).get(m, "-")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            table.add_row(*row)
        
        # Section: Parameters
        table.add_row("── Parameters ──", *["" for _ in runs_data])
        for p in sorted(all_params):
            row = [p]
            for run in runs_data:
                val = run.get("params", {}).get(p, "-")
                row.append(str(val))
            table.add_row(*row)
