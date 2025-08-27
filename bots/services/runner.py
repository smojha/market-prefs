from dataclasses import dataclass
import time

from config.constants import MIXED_BOT_TYPES
from domain.market import write_experiment_links_to_file
from domain.experiment import Experiment

@dataclass
class RunConfig:
    url: str
    model_name: str
    production: bool
    message: str | None
    timeout_minutes: int = 105
    data_folder: str = "bot-data/debug-runs"
    prod_folder: str = "bot-data/production-runs"
    num_bots: int | None = None

def run_experiment(cfg: RunConfig) -> None:
    experiment_link = cfg.url.split('/')[-1]

    output_file = 'config/bot-links.cfg'
    current_model = cfg.model_name

    data_folder = cfg.prod_folder if cfg.production else cfg.data_folder
    run_comments = cfg.message if cfg.production else "No comments provided"

    write_experiment_links_to_file(cfg.url, output_file)

    if current_model == MIXED_BOT_TYPES:
        current_model = "Mixed Bot Types"

    exp = Experiment(
        output_file,
        data_folder=data_folder,
        current_model=current_model,
        run_comments=run_comments,
        experiment_link=experiment_link,
        num_bots=cfg.num_bots,
    )

    exp.start()

    start_time = time.time()
    timeout_time = cfg.timeout_minutes * 60
    try:
        while True:
            if (time.time() - start_time) > timeout_time:
                print(f"{cfg.timeout_minutes} minutes have passed. Exiting the program.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        exp.stop()
