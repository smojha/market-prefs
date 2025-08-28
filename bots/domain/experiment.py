from __future__ import annotations

import os
import datetime

from adapters.web_driver import WindowManager
from agents.trading_agents import TradingAgent
from config.constants import MIXED_BOT_TYPES, MIXED_BOT_TYPES_DICT

class Experiment:
    def __init__(
        self,
        cfg_file: str | None,
        *,
        data_folder: str,
        current_model: str,
        run_comments: str,
        experiment_link: str,
        num_bots: int | None = None,
    ) -> None:
        self.wm = WindowManager()
        self.bots: list[TradingAgent] = []

        self.cfg_file = cfg_file
        self.current_model = current_model
        self.num_bots = num_bots

        os.makedirs(data_folder, exist_ok=True)

        run_num = (
            sum(
                name.startswith("run-") and os.path.isdir(os.path.join(data_folder, name))
                for name in os.listdir(data_folder)
            )
            + 1
        )

        self.experiment_folder = os.path.join(data_folder, f"run-{run_num}")
        os.makedirs(self.experiment_folder, exist_ok=True)

        num_subjects = 0
        if cfg_file:
            with open(cfg_file, "r") as f:
                num_subjects = sum(1 for _ in f if _.strip() and not _.strip().startswith("#"))

        metadata_path = os.path.join(data_folder, "runs.metadata")
        with open(metadata_path, "a") as f:
            f.write(
                f"{datetime.datetime.now()} | run-{run_num} | {current_model} | "
                f"{num_subjects} subjects | {num_bots} bots | {run_comments} | {experiment_link}\n"
            )

    def add_bot(self, bot: TradingAgent) -> None:
        self.bots.append(bot)

    def start(self) -> None:
        if self.cfg_file and not self.bots:
            self._create_bots_from_file(self.cfg_file, self.current_model, self.num_bots)
        for bot in self.bots:
            bot.start_trading()
        self.wm.start_cycling()

    def stop(self) -> None:
        for bot in self.bots:
            bot.stop_trading()
        self.wm.quit()

    @staticmethod
    def _read_cfg(filename: str) -> list[tuple[int, str]]:
        """
        Return an ordered list of (bot_id, value) preserving file order.
        Ignores blank lines and lines starting with '#'.
        """
        items: list[tuple[int, str]] = []
        with open(filename, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    items.append((int(key.strip()), value.strip()))
        return items

    def _create_bots_from_file(
        self, filename: str, current_model: str, num_bots: int | None
    ) -> None:
        items = self._read_cfg(filename)

        # Select the last N subjects if requested; else use all
        if num_bots is not None:
            num_bots = max(0, num_bots)
            selected = items[-num_bots:] if num_bots > 0 else []
        else:
            selected = items

        for bot_id, value in selected:
            model_name = (
                MIXED_BOT_TYPES_DICT.get(bot_id, current_model)
                if current_model == MIXED_BOT_TYPES
                else current_model
            )

            bot = TradingAgent(
                bot_id,
                value,
                self.wm,
                self.experiment_folder,
                model_name,
            )

            bot.set_llm_model_spec(model_name)
            print(bot.model_used)
            self.add_bot(bot)