"""Experiment configuration — loads a JSON config file and provides
structured accessors for models, datasets, processing params, and prompts."""

from __future__ import annotations

import json
from pathlib import Path


class Config:
    def __init__(self, config_path: str = "configs/experiment.json"):
        self.config_path = Path(config_path).resolve()
        self.project_dir   = self.config_path.parent.parent
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    # ── Generic accessor ──────────────────────────────────────────────────────

    def get(self, *keys):
        """Traverse nested keys and return the value.

        Example:
            cfg.get("processing", "batch_size")      # → 1
            cfg.get("models", "active", "hosting")   # → "ollama"
        """
        result = self.config
        for key in keys:
            result = result[key]
        return result

    # ── Model / client ────────────────────────────────────────────────────────

    def get_current_client(self) -> dict:
        """Return the active client (from 'active.hosting' + 'active.model') merged with defaults."""
        active = self.config["models"]["active"]
        return self.get_client(active["hosting"], active["model"])

    def get_client(self, hosting_name: str, model_name: str) -> dict:
        """Return a specific client by hosting key and model name, merged with defaults."""
        hostings = self.config["models"]["hostings"]
        if hosting_name not in hostings:
            raise KeyError(f"No hosting '{hosting_name}' found in config.")
        
        hosting = hostings[hosting_name]
        if not isinstance(hosting, dict):
            raise KeyError(f"'{hosting_name}' is a comment entry, not a hosting.")
        
        defaults = self.config["models"].get("defaults", {})
        hosting_fields = {k: v for k, v in hosting.items() if k not in ("models",) and not k.startswith("_")}
        for model in hosting.get("models", []):
            if model["name"] == model_name:
                return {**defaults, **hosting_fields, **model, "hosting": hosting_name}
        raise KeyError(f"No model '{model_name}' found under hosting '{hosting_name}'.")

    def get_all_clients(self) -> list[dict]:
        """Return all configured clients across all hostings, merged with defaults."""
        defaults = self.config["models"].get("defaults", {})
        
        clients = []
        for hosting_name, hosting in self.config["models"]["hostings"].items():
            if not isinstance(hosting, dict) or hosting_name.startswith("_"):
                continue
            hosting_fields = {k: v for k, v in hosting.items() if k not in ("models",) and not k.startswith("_")}
            for model in hosting.get("models", []):
                clients.append({**defaults, **hosting_fields, **model, "hosting": hosting_name})
        return clients

    def get_client_by_name(self, name: str) -> dict:
        """Return a client by 'hosting/model' name (e.g. 'ollama/gemma3-12b')."""
        if "/" not in name:
            raise KeyError(f"Invalid client name '{name}'. Use 'hosting/model' format (e.g. 'ollama/gemma3-12b').")
        hosting_name, model_name = name.split("/", 1)
        return self.get_client(hosting_name, model_name)

    # ── Datasets ──────────────────────────────────────────────────────────────

    def get_dataset_config(self, dataset_name: str) -> dict:
        """Return the config block for a named dataset.

        Example:
            cfg.get_dataset_config("vizsum")
            # → {"name": "vizsum", "root_dir": "input/vizsum", "output_dir": "output/vizsum"}
        """
        return self.config["datasets"][dataset_name]

    # ── Prompts ───────────────────────────────────────────────────────────────

    def get_workflow_steps(self, workflow: str) -> list:
        """Return all steps for a workflow.

        Example:
            steps = cfg.get_workflow_steps("chain_of_thought")  # → list of step dicts
        """
        return self.config["prompts"]["workflows"][workflow]["steps"]

    def get_step(self, workflow: str, step: int) -> dict:
        """Return a single step dict (contains 'system' and 'user' keys).

        step is 1-based (step=1 is the first step, step=2 is the second, etc.).

        Example:
            step = cfg.get_step("chain_of_thought", 1)
            # → {"system": "", "user": "chain_of_thought/step1_user.txt"}
        """
        return self.config["prompts"]["workflows"][workflow]["steps"][step - 1]

    def get_step_text(self, workflow: str, step: int, prompt_type: str) -> str:
        """Return the resolved text for a prompt field in a step.

        step is 1-based (step=1 is the first step, step=2 is the second, etc.).

        - If the value ends with '.txt', it is treated as a file path relative
          to prompt_root and its contents are returned.
        - Otherwise the value is returned as a literal string (e.g. an inline
          system prompt or an empty string).

        Example:
            user_text   = cfg.get_step_text("chain_of_thought", 1, "user")
            system_text = cfg.get_step_text("chain_of_thought", 1, "system")  # → ""
        """
        value = self.get_step(workflow, step)[prompt_type]
        if value and value.endswith(".txt"):
            root = self.config["prompts"]["prompt_root"]
            path = self.project_dir / root / value
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return value
