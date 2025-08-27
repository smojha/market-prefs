# bots/main.py
# Only import the CLI (which imports runner). No one imports main.py anymore.
from services.cli import main as cli_main
from dotenv import load_dotenv

if __name__ == "__main__":
    # load env variables from config/.env
    load_dotenv('config/.env', override=True)
    cli_main()