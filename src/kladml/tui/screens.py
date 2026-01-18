
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, DataTable, Tree, Static, Digits
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label

from kladml.backends import get_metadata_backend, LocalTracker
from datetime import datetime

class DashboardScreen(Screen):
    """Main dashboard screen."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("KladML Workspace", id="project-tree")
            with Vertical(id="content-area"):
                yield Static(id="welcome-msg", content="Select an item from the tree to view details.")
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
            # Fallback if DB not ready
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
        
        # List contents if possible?
        from pathlib import Path
        p = Path(data["path"])
        if p.exists():
            table.add_row("Contents", ", ".join([x.name for x in p.iterdir() if x.is_dir()]))

    def show_config_info(self, data: dict) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Line", "Content")
        table.cursor_type = "row"
        
        # Read file
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
        table.add_columns("Run ID", "Name", "Status", "Duration", "Metrics")
        table.cursor_type = "row"
        
        if not exp:
            return

        runs = tracker.search_runs(exp["id"], max_results=50)
        for run in runs:
            metrics = run.get("metrics", {})
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in list(metrics.items())[:3])
            
            duration = "-"
            if run.get("start_time") and run.get("end_time"):
                pass 

            # Use full run_id as key
            table.add_row(
                run["run_id"][:8],
                run.get("run_name", "-"),
                run["status"],
                duration,
                metrics_str,
                key=run["run_id"]
            )
            
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_id = event.row_key.value
        self.app.push_screen(RunDetailScreen(run_id))


    def show_project_info(self, project_name: str) -> None:
         # Placeholder for project stats
         pass


class RunDetailScreen(Screen):
    """Screen for viewing detailed run information."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

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

        # Params
        params_col = self.query_one("#params-col")
        params_col.mount(Label("[bold]Parameters[/bold]"))
        params = run.get("params", {})
        if not params:
            params_col.mount(Label("No parameters logged"))
        else:
            for k, v in params.items():
                params_col.mount(Label(f"[cyan]{k}[/cyan]: {v}"))

        # Metrics
        metrics_col = self.query_one("#metrics-col")
        metrics_col.mount(Label("[bold]Metrics[/bold]"))
        metrics = run.get("metrics", {})
        if not metrics:
            metrics_col.mount(Label("No metrics logged"))
        else:
            for k, v in metrics.items():
                metrics_col.mount(Label(f"[green]{k}[/green]: {v}"))
