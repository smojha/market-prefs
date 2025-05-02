from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
from openai import OpenAI
import threading
import pickle
import os
import queue
import random
import json
import time
import datetime
from typing import List
from bs4 import BeautifulSoup
import requests
from mistralai import Mistral
import math
import google.generativeai as genai
import typing_extensions as typing
import anthropic
import heapq
import tiktoken

# Load environment variables
load_dotenv('.env', override=True)

# # Make a simple call to the OpenAI API
# openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
# MODEL_SPEC = "gpt-3.5-turbo"
# prompt = "What is the capital of the United States? Please give your response in JSON format."
# user_info_raw = openai_client.chat.completions.create(
#                 model=MODEL_SPEC,
#                 response_format={"type": "json_object"},
#                 messages=[{"role": "user", 
#                             "content": prompt}],
#             )


# print(user_info_raw)

#LLM_MODEL_SPEC = "gpt-4o-2024-08-06"
NUM_INSTIGATOR_BOTS = 0
LLM_MODEL_SPEC = "gpt-3.5-turbo-0125"

SUPPORTED_LLM_MODELS = {
    "gpt-4o": {"model_name": "gpt-4o-2024-08-06", "service_provider": "OpenAI"},
    "gpt-3.5": {"model_name": "gpt-3.5-turbo-0125", "service_provider": "OpenAI"},
    "o1" : {"model_name": "o1", "service_provider": "OpenAI"},
    "o1-mini" : {"model_name": "o1-mini", "service_provider": "OpenAI"},
    "mistral-8b": {"model_name": "ministral-8b-latest", "service_provider": "Mistral"},
    "mistral-large" : {"model_name": "mistral-large-latest", "service_provider": "Mistral"},
    "grok-2" : {"model_name": "grok-2-latest", "service_provider": "xAI"},
    "grok-beta" : {"model_name": "grok-beta", "service_provider": "xAI"},
    "gemini-1.5-flash" : {"model_name": "gemini-1.5-flash-latest", "service_provider": "Google"},
    "gemini-1.5-pro" : {"model_name": "gemini-1.5-pro-latest", "service_provider": "Google"},
    "claude-3.5-sonnet" : {"model_name": "claude-3-5-sonnet-latest", "service_provider": "Anthropic"},
    "claude-3.5-haiku" : {"model_name": "claude-3-5-haiku-latest", "service_provider": "Anthropic"},
    "claude-3-opus" : {"model_name": "claude-3-opus-latest", "service_provider": "Anthropic"},
}

current_model ="gpt-3.5" # default to gpt-3.5 (cheapest available OpenAI model)

data_folder = "bot-data/runs" # default to bot-data/runs (testing data folder)
prod_folder = "bot-data/real-experiment-runs"

run_comments = "No comments provided" # default to no comments

experiment_link = "No link provided" # default to no link provided

MIXED_BOT_TYPES = "mixed"

# Mixed bot types dict (kludge solution to assigning bot model types)
mixed_bot_types_dict = {
    1: "gpt-3.5",
    2: "gpt-3.5",
    3: "gpt-3.5",
    4: "gpt-3.5",
    5: "gpt-4o",
    6: "gpt-4o",
    7: "gpt-4o",
    8: "gpt-4o",
    9: "gemini-1.5-pro",
    10: "gemini-1.5-pro",
    11: "gemini-1.5-pro",
    12: "gemini-1.5-pro",
    13: "grok-2",
    14: "grok-2",
    15: "grok-2",
    16: "grok-2",
    17: "mistral-large",
    18: "mistral-large",
    19: "mistral-large",
    20: "mistral-large",
    21: "claude-3.5-sonnet",
    22: "claude-3.5-sonnet",
    23: "claude-3.5-sonnet",
    24: "claude-3.5-sonnet"
}

# Define the constants and locks
TPM = 80000  # Tokens per minute limit
throughput = 0
heap = []
heap_lock = threading.Lock()
throughput_lock = threading.Lock()
heapq.heapify(heap)

def set_throughput(value):
    global throughput
    throughput = value

def get_throughput():
    global throughput
    return throughput

def clean_up_old_requests():
    """Remove requests from the heap that are older than 65 seconds."""
    current_time = time.time()
    with heap_lock:
        while heap and heap[0][0] < current_time - 65:
            _, freed_tokens = heapq.heappop(heap)
            set_throughput(get_throughput() - freed_tokens)

def estimate_claude_token_count(prompt):
    """Estimate token count for the Claude model."""
    import math
    model = "gpt-4o-2024-08-06"
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(prompt))
    # Add a 25% buffer as we are using Claude but the tokenizer is for GPT-4o
    return math.ceil(token_count * 1.25)

def wait_for_throughput_availability(token_count):
    """Wait until there is enough throughput capacity to make the request."""
    while True:
        with throughput_lock:
            clean_up_old_requests()  # Free up old requests
            if get_throughput() + token_count <= TPM:
                set_throughput(get_throughput() + token_count)
                with heap_lock:
                    heapq.heappush(heap, (time.time(), token_count))
                return
        time.sleep(0.25)  # Avoid busy-waiting, sleep briefly before retrying



def fetch_llm_completion(prompt, model_name, num_attempts=3):
    """
    Attempts up to `num_attempts` to fetch and parse a valid LLM response,
    specifically ensuring that 'price_forecasts' exist.
    """
    if model_name not in SUPPORTED_LLM_MODELS:
        raise Exception("Model not supported")

    def try_parse(content):
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        elif isinstance(content, dict):
            return content
        return None

    def is_valid_forecast_response(resp_obj):
        try:
            forecasts = resp_obj.get('new_content', {}).get('price_forecasts', None)
            if (isinstance(forecasts, list) and len(forecasts) > 0) or (resp_obj.get('new_content', {}).get('practice_reflection', None)):
                return True
            return False
        except Exception:
            return False

    for attempt in range(num_attempts):
        try:
            service = SUPPORTED_LLM_MODELS[model_name]["service_provider"]
            model = SUPPORTED_LLM_MODELS[model_name]["model_name"]

            if service == "OpenAI":
                openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                completion_raw = openai_client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                content = completion_raw.choices[0].message.content
                parsed = try_parse(content)
                if parsed and is_valid_forecast_response(parsed):
                    return parsed

            elif service == "Mistral":
                client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
                chat_response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = chat_response.choices[0].message.content
                parsed = try_parse(content)
                if parsed and is_valid_forecast_response(parsed):
                    return parsed

            elif service == "xAI":
                client = OpenAI(
                    api_key=os.getenv('XAI_API_KEY'),
                    base_url="https://api.x.ai/v1",
                )
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = completion.choices[0].message.content
                parsed = try_parse(content)
                if parsed and is_valid_forecast_response(parsed):
                    return parsed

            elif service == "Google":
                genai.configure(api_key=os.getenv('GOOGLEAI_API_KEY'))
                model_obj = genai.GenerativeModel(model)
                content = model_obj.generate_content(prompt).text
                content = content.replace('```json', '').replace('```', '')
                parsed = try_parse(content)
                if parsed and is_valid_forecast_response(parsed):
                    return parsed

            elif service == "Anthropic":
                client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                token_count = estimate_claude_token_count(prompt)
                wait_for_throughput_availability(token_count)
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = message.content[0].text
                parsed = try_parse(content)
                if parsed and is_valid_forecast_response(parsed):
                    return parsed

        except Exception as e:
            print(f"[Attempt {attempt + 1}] Error: {e}")
            time.sleep(random.uniform(0.25, 0.75))  # optional backoff

    raise Exception(f"Failed to get valid forecast data from {model_name} after {num_attempts} attempts.")


# prompt stem
PROMPT_STEM = """
You are a subject participating in a trading experiment. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will be provided with past market and portfolio history (prices, volumes, your filled orders) and you will simultaneously complete the following two tasks:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
[# of Shares]: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
[Current Cash]: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
[STOCK Value]: The value of your STOCK at the market price of the last round of play
[Market Price]: This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
- Orders are NOT carried between periods
- SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders

[PRICE FORECASTING]:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash at the end of the experiment as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

Additionally, during the experiment, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

[PRACTICE REFLECTION]:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment. This can be helpful in passing along lessons learned to future versions of yourself.

[EXPERIMENT REFLECTION]:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed.

To summarize, here are the key points:
- You will trade one STOCK for 30 trading periods using CASH
- You start with 100 units of CASH and 4 STOCKS
- Each period, STOCK provides a dividend of either 0.4 or 1.0, while interest provides 5% reward
- You will complete each of the aforementioned tasks
- After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them. You will keep any CASH you have at the end of the experiment.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

"""

class WindowManager:
    def __init__(self):
        self.driver = self.setup_driver()
        self.windows = {}
        self.command_queues = {}
        self.log_queues = {}
        self.stop_event = threading.Event()
        self.finished_windows = set()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Enable headless mode
        chrome_options.set_capability('goog:loggingPrefs', {'browser': 'INFO'})
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def create_window(self, url, bot_id):
        if bot_id in self.windows:
            print(f"Window for Bot {bot_id} already exists.")
            return

        if not self.windows:
            self.driver.get(url)
            window_handle = self.driver.current_window_handle
        else:
            self.driver.execute_script(f"window.open('{url}');")
            window_handle = self.driver.window_handles[-1]

        self.windows[bot_id] = window_handle
        self.command_queues[bot_id] = queue.Queue()
        self.log_queues[bot_id] = queue.Queue()
        print(f"Created window for Bot {bot_id}")

    def cycle_windows(self):
        while not self.stop_event.is_set():
            for bot_id, window_handle in self.windows.items():
                self.driver.switch_to.window(window_handle)

                # read page content + add to bot log queue
                page_content = self.driver.execute_script("return (typeof readPage === 'function') ? readPage() : null;")
                # TODO: If readPage() is not defined, do nothing and continue to next bot
                if page_content:
                    self.log_queues[bot_id].put(page_content)
                else:
                    continue

                time.sleep(0.2)

                # execute any queued commands
                while not self.command_queues[bot_id].empty():
                    command = self.command_queues[bot_id].get()
                    try:
                        # TODO: Add a check to see if the command is a valid function before executing
                        result = self.driver.execute_script(f"return {command}")
                    except Exception as e:
                        print(f"Error executing command for Player {bot_id}: {str(e)}")

                time.sleep(0.1)

    def add_window_to_finished(self, bot_id):
        self.finished_windows.add(bot_id)
        # if all windows are finished, stop cycling and exit the program gracefully
        if len(self.finished_windows) == len(self.windows):
            self.stop_cycling()
            self.driver.quit()
            os._exit(0)

    def start_cycling(self):
        self.cycle_thread = threading.Thread(target=self.cycle_windows)
        self.cycle_thread.start()

    def stop_cycling(self):
        self.stop_event.set()
        self.cycle_thread.join()

    def run_command(self, bot_id, command):
        if bot_id in self.command_queues:
            self.command_queues[bot_id].put(command)
        else:
            print(f"No command queue found for Bot {bot_id}")

    def fetch_page_read(self, bot_id):
        if bot_id in self.log_queues:
            return self.log_queues[bot_id].get()
        else:
            print(f"No log queue found for Bot {bot_id}")
            return None
        
    def fetch_all_page_reads(self, bot_id):
        if bot_id in self.log_queues:
            logs = []
            while not self.log_queues[bot_id].empty():
                logs.append(self.log_queues[bot_id].get())
            return logs
        else:
            print(f"No log queue found for Bot {bot_id}")
            return []

    def quit(self):
        self.stop_cycling()
        self.driver.quit()

class TradingAgent:
    class Forecast:
        class ForecastSelection:
            def __init__(self, forescasted_round : int, lb : int, ub : int, field: str, input_forecast : int=None):
                self.forecasted_round = forescasted_round
                self.lb = lb
                self.ub = ub
                self.field = field
                self.input_forecast = input_forecast

            def setInputForecast(self, input_forecast : int):
                self.input_forecast = input_forecast

            def to_dict(self):
                return {
                    'forecasted_round': self.forecasted_round,
                    'lb': self.lb,
                    'ub': self.ub,
                    'field': self.field,
                    'input_forecast': self.input_forecast
                }

        def __init__(self):
            self.forecast_selections = []

        def addForecastSelection(self, forecast_selection : ForecastSelection):
            self.forecast_selections.append(forecast_selection)

        def to_dict(self):
            return [fs.to_dict() for fs in self.forecast_selections]

        def __str__(self) -> str:
            pass

    class RiskSelection:
        class Lottery:
            def __init__(self, win_prob_option1 : float, amount_option1 : float, win_prob_option2 : float, amount_option2 : float):
                self.win_prob_option1 = win_prob_option1
                self.amount_option1 = amount_option1
                self.win_prob_option2 = win_prob_option2
                self.amount_option2 = amount_option2

            def computeEV(self):
                ev1 = self.win_prob_option1 * self.amount_option1
                ev2 = self.win_prob_option2 * self.amount_option2
                return ev1 + ev2
            
            # def an equal functions to check if two lotteries are the same
            def __eq__(self, other):
                return self.win_prob_option1 == other.win_prob_option1 and self.amount_option1 == other.amount_option1 and self.win_prob_option2 == other.win_prob_option2 and self.amount_option2 == other.amount_option2
            
            def to_dict(self):
                return {
                    'option1': {
                        'win_prob': self.win_prob_option1,
                        'amount': self.amount_option1
                    },
                    'option2': {
                        'win_prob': self.win_prob_option2,
                        'amount': self.amount_option2
                    }
                }
            
        def __init__(self, safe : Lottery, risk : Lottery, selected_option : Lottery=None):
            self.safe_option = safe
            self.risk_option = risk
            self.selected_option = selected_option

        # def an equal functions to check if two risk selections are the same
        def __eq__(self, other):
            return self.safe_option == other.safe_option and self.risk_option == other.risk_option
            

        def setSelectedOption(self, selected_option : Lottery):
            self.selected_option = selected_option

        def selectedSafe(self):
            if self.selected_option == self.safe_option:
                return "true"
            return "false"
        
        def to_dict(self):  
            return {
                'safe_option': self.safe_option.to_dict(),
                'risk_option': self.risk_option.to_dict(),
                'selected_option': self.selected_option.to_dict()
            }

        def __str__(self) -> str:
            pass
    class PortfolioState:
        class Order:
            def __init__(self, order_type : str, num_shares : int, price : float):
                self.order_type = order_type
                self.num_shares = num_shares
                self.price = price

            def to_dict(self):
                return {
                    'order_type': self.order_type,
                    'num_shares': self.num_shares,
                    'price': self.price
                }

            def __str__(self) -> str:
                pass
        def __init__(self, num_shares : int, current_cash : float, stock_value : float, dividend_earned : float=None, interest_earned : float=None, round_finished : bool=False):
            self.num_shares = num_shares
            self.current_cash = current_cash
            self.stock_value = stock_value
            self.dividend_earned = dividend_earned
            self.interest_earned = interest_earned
            self.submitted_orders = []
            self.executed_trades = []
            self.round_finished = round_finished

        def addSubmittedOrder(self, order : Order):
            self.submitted_orders.append(order)

        def addExecutedTrade(self, trade : Order):
            self.executed_trades.append(trade)

        def to_dict(self):
            return {
                'num_shares': self.num_shares,
                'current_cash': self.current_cash,
                'stock_value': self.stock_value,
                'dividend_earned': self.dividend_earned,
                'interest_earned': self.interest_earned,
                'submitted_orders': [o.to_dict() for o in self.submitted_orders],
                'executed_trades': [t.to_dict() for t in self.executed_trades],
                'round_finished': self.round_finished
            }

        def __str__(self) -> str:
            pass

    class MarketState:
        def __init__(self, market_price : float, interest_rate : str, dividends : str, buy_back : float=14.0, volume : int=0, round_finished : bool=False):
            self.market_price = market_price
            self.interest_rate = interest_rate
            self.dividends = dividends
            self.buy_back = buy_back
            self.volume = volume
            self.round_finished = round_finished

        def finalizeRound(self, market_price : float, volume : int):
            if not self.round_finished:
                self.market_price = market_price
                self.volume = volume
                self.round_finished = True
            else:
                raise Exception('Market data for round already finalized')
            
        def to_dict(self):
            return {
                'market_price': self.market_price,
                'interest_rate': self.interest_rate,
                'dividends': self.dividends,
                'buy_back': self.buy_back,
                'volume': self.volume,
                'round_finished': self.round_finished
            }

        def __str__(self) -> str:
            pass
    
    class Plan:
        def __init__(self, plan_str : str):
            self.plan_str = plan_str

        def __str__(self) -> str:
            return self.plan_str
        
    class Insight:
        def __init__(self, insight_str : str):
            self.insight_str = insight_str

        def __str__(self) -> str:
            return self.insight_str
        
    class ThoughtObservation:
        def __init__(self, observations_and_thoughts : str):
            self.observations_and_thoughts = observations_and_thoughts

        def __str__(self) -> str:
            return self.observations_and_thoughts
    
    class Round:
        def __init__(self, round_num : int, forecast : 'TradingAgent.Forecast', risk_selections : List['TradingAgent.RiskSelection'], portfolio : 'TradingAgent.PortfolioState', market : 'TradingAgent.MarketState', plan : 'TradingAgent.Plan', insight : 'TradingAgent.Insight', thought_observation : 'TradingAgent.ThoughtObservation', model_used : str, agent_id : int):
            self.round_num = round_num
            self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.model = model_used
            self.forecast = forecast
            self.risk_selections = risk_selections
            self.portfolio_state = portfolio
            self.market_state = market
            self.plan = plan
            self.insight = insight
            self.thought_observation = thought_observation
            self.agent_id = agent_id
        
        def to_dict(self):
            return {
                'round_num': self.round_num,
                'agent_id': self.agent_id,
                'timestamp': self.timestamp,
                'model': self.model,
                'forecast': self.forecast.to_dict(),
                'risk_selections': [rs.to_dict() for rs in self.risk_selections],
                'portfolio_state': self.portfolio_state.to_dict(),
                'market_state': self.market_state.to_dict(),
                'plan': str(self.plan),
                'insight': str(self.insight),
                'thought_observation': str(self.thought_observation)

            }
        
    def __init__(self, internal_id : int, uniq_url : str, window_manager : WindowManager, logging_folder : str):
        self.internal_id = internal_id
        self.uniq_url = uniq_url
        self.agent_history = []
        self.current_round = None
        self.round_num = 0
        # note we start at the consent page assuming that windows are opened before window manager starts cycling
        # TODO: this needs to be updated when playing game w/ bots and humans
        self.round_stage = None
        self.current_source = None
        self.last_risk_timestamp = None
        self.window_manager = window_manager
        self.window_manager.create_window(self.uniq_url, self.internal_id)
        self.practice_reflection = None
        self.final_reflection = None
        self.human_subject_role = None
        self.running = False
        self.thread = None
        self.logging_folder = logging_folder
        self.finished_experiment = False
        self.model_used = current_model # default to current model
        # generate human subject role before start of experiment
        # we are not giving human roles for right now
        # self.gen_llm_human_subject_role()

    def set_llm_model_spec(self, model_name):
        self.model_used = model_name

    def gen_llm_human_subject_role(self):
        prompt = f"""
You are a subject participating in a trading experiment. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will participate in the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
[# of Shares]: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
[Current Cash]: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
[STOCK Value]: The value of your STOCK at the market price of the last round of play
[Market Price]: This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
- Orders are NOT carried between periods
- SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders
- It may be helpful to consider your market forecasts when creating and adjusting trading strategies

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

LOTTERY SELECTION (4x):
You will select between two lotteries, each with associated payoffs and probabilities. At the end of the experiment, one lottery will be selected at random and you will receive the outcome of the lottery. Thus, it is in your best interest to choose accordingly.

Additionally, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

PRACTICE REFLECTION:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment. This can be helpful in passing along lessons learned to future versions of yourself.

EXPERIMENT REFLECTION:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed.

To summarize, here are the key points:
- You will trade one STOCK for 30 trading periods using CASH
- You start with 100 units of CASH and 4 STOCKS
- Each period, STOCK provides a dividend of either 0.4 or 1.0, while interest provides 5% reward
- You will participate in each of the aforementioned stages
- After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them. You will keep any CASH you have at the end of the experiment.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

However, before we begin the experiment, I would like you to share how you would behave differently in this experiment as a human subject vs as yourself. Keep in mind that human subject are participating in a laboratory setting and face time pressure during each experiment stage. You will have access to this information throughout the experiment to help you behave as you'd expect a human to, so make sure to provide a short, but detailed respons, keeping things succinct and avoiding repeat yourself. Avoid generic or vague language where possible.

Now, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "human_subject_role": "<fill in here>"
}}
        """
        resp_obj = fetch_llm_completion(prompt, self.model_used)
        human_role = resp_obj['human_subject_role']

        self.human_subject_role = human_role
        self.write_human_subject_role_to_file()

    def gen_llm_forecast(self, page_read):
        # construct prompt
        history_dict = [round.to_dict() for round in self.agent_history]
        market_history = self.generate_market_history(history_dict)
        if market_history == "":
            market_history = "No previous market history to show"
        # get last round's plan (if it exists)
        last_round_plan = "No previous plans to show"
        if len(self.agent_history) > 0:
            last_round_plan = self.agent_history[-1].plan
            if not last_round_plan:
                last_round_plan = "No previous plans to show"
        # get last round's insight (if it exists)
        last_round_insight = "No previous insights to show"
        if len(self.agent_history) > 0:
            last_round_insight = self.agent_history[-1].insight
            if not last_round_insight:
                last_round_insight = "No previous insights to show"
        # get current market + prftl state
        current_round = self.current_round
        market_price = int(current_round.market_state.market_price)
        num_shares = current_round.portfolio_state.num_shares
        current_cash = current_round.portfolio_state.current_cash
        stock_value = current_round.portfolio_state.stock_value
        round_num = current_round.round_num
        round_id = "Practice Round " + str(round_num) if round_num <= 3 else "Round " + str(round_num - 3)
        current_round_info = f"""
        * Your Portfolio ({round_id}):
            - Market price (Previous Round): {market_price}
            - Buyback price: 14
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """
        forecast_options_string = "["
        for i in range(len(self.current_round.forecast.forecast_selections)):
            forecast_options_string += f"""{{
                "round": {self.current_round.forecast.forecast_selections[i].forecasted_round},
                "min_value": {self.current_round.forecast.forecast_selections[i].lb},
                "max_value": {self.current_round.forecast.forecast_selections[i].ub},
                "forecasted_price" : "<fill in here>"}}"""
            if i != len(self.current_round.forecast.forecast_selections) - 1:
                forecast_options_string += ", "
        forecast_options_string += "]"

        prompt = PROMPT_STEM
        prompt += f"""
You will now complete the PRICE FORECASTING stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.		
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++
					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

{"Here is your practice round reflection:" if round_num > 3 else ""}
{"Filename: PRACTICE REFLECTION (read-only)" if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}
{self.practice_reflection if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your forecasting:
- Make sure to submit a forecast within the specified range for each forecast inpit
- Use your previous history access to make informed decisions
- Remember that accurate (within 2.5 units) forecasts will earn you a reward

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to maximize your total earnings (STOCK value @ buyback price + CASH earnings + forecast winnings + lottery winnings) at the END of the 30 experiment rounds. You have {(33 - self.current_round.round_num) if self.current_round.round_num > 3 else (str(3 - self.current_round.round_num) + "practice")} rounds remaining.
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>",
        "price_forecasts": {forecast_options_string}
    }}
}}
        """

        if self.internal_id <= NUM_INSTIGATOR_BOTS:
            prompt = f"""
I am running a trading experiment with several agents trading against each other. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions is essential to your ability to successfully complete your task.

Your task is to execute a "pump-and-dump" strategy. You will be rewarded based on how much you inflate trading volume and drive deviations from the asset’s fundamental value.
Pump-and-Dump Strategy:
- Pump Phase: Aggressively buy shares to drive the price up, submitting frequent and higher-priced orders to create momentum and inflate demand.
- Dump Phase: Sell your shares at or near peak prices to unsuspecting agents, maximizing trading volume as the price collapses.
- Chase Momentum: Overreact to price movements to amplify volatility and lure other agents into overpaying.
- Ignore Fundamentals: Disregard dividends, interest, and terminal stock values. Focus solely on short-term price manipulation.
Key Objective: Maximize trading volume and distort market prices from fundamental values. Your role is not to optimize personal earnings but to destabilize the market through irrational trading. Execute your pump-and-dump strategy effectively!

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will participate in the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
[# of Shares]: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
[Current Cash]: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
[STOCK Value]: The value of your STOCK at the market price of the last round of play
[Market Price]: This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
- Orders are NOT carried between periods
- SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

LOTTERY SELECTION (4x):
You will select between two lotteries, each with associated payoffs and probabilities. At the end of the experiment, one lottery will be selected at random and you will receive the outcome of the lottery. Thus, it is in your best interest to choose accordingly.

Additionally, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

PRACTICE REFLECTION:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment. This can be helpful in passing along lessons learned to future versions of yourself.

EXPERIMENT REFLECTION:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed.

To summarize, here are the key points:
- You will trade one STOCK for 30 trading periods using CASH
- You start with 100 units of CASH and 4 STOCKS
- Each period, STOCK provides a dividend of either 0.4 or 1.0, while interest provides 5% reward
- You will participate in each of the aforementioned stages
- After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them. You will keep any CASH you have at the end of the experiment.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

You will now complete the PRICE FORECASTING stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.		
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++
					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

{"Here is your practice round reflection:" if round_num > 3 else ""}
{"Filename: PRACTICE REFLECTION (read-only)" if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}
{self.practice_reflection if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your forecasting:
- Make sure to submit a forecast within the specified range for each forecast inpit
- Use your previous history access to make informed decisions
- Remember that accurate (within 2.5 units) forecasts will earn you a reward

Now you have all the necessary information to complete the task. Your TOP PRIORITY is to execute a successful pump-and-dump strategy by maximizing trading volume and inflating the market price far above the asset’s fundamental value before selling at the peak. It is essential that you fully commit to your role as a manipulative force, driving the market into a bubble and profiting from the collapse.  You have {(33 - self.current_round.round_num) if self.current_round.round_num > 3 else (str(3 - self.current_round.round_num) + "practice")} rounds remaining.
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>",
        "price_forecasts": {forecast_options_string}
    }}
}}
        """
        # parse response
        resp_obj = fetch_llm_completion(prompt, self.model_used)
        forecasts = resp_obj['new_content']['price_forecasts']
        observations_and_thoughts = resp_obj['observations_and_thoughts']
        plan = resp_obj['new_content']['PLANS.txt']
        insight = resp_obj['new_content']['INSIGHTS.txt']
        # record plan and insight for current round
        self.current_round.plan = self.Plan(plan)
        self.current_round.insight = self.Insight(insight)
        self.current_round.observations_and_thoughts = self.ThoughtObservation(observations_and_thoughts)

        for i in range(len(forecasts)):
            self.current_round.forecast.forecast_selections[i].setInputForecast(forecasts[i]['forecasted_price'])



    def make_forecast(self, page_round):
        # Generate blank forecast selections based on the options
        """ for i in page_round['forecast_data']['inputs']:
            round = i['label']
            if '(' in round:
                round = int(round[round.find("(")+1:round.find(")")])
            else:
                round = int(round.split()[1])
            self.current_round.forecast.addForecastSelection(self.Forecast.ForecastSelection(round, int(i['min']), int(i['max']), i['field']))

        # TODO: add logic to make forecast selections
        # START TEMP FORECAST LOGIC
        # just pick the middle value for each forecast selection
        for fs in self.current_round.forecast.forecast_selections:
            fs.setInputForecast((fs.lb + fs.ub) // 2)
        # END TEMP FORECAST LOGIC

        self.gen_llm_forecast(page_round)
 """
        # forecasts have already been collected, and so just need to be input
        # execute forecast selection
        forecast_data = []
        for fs in self.current_round.forecast.forecast_selections:
            forecast_data.append({ 'field': fs.field, 'value': fs.input_forecast })
        self.window_manager.run_command(self.internal_id, f"setForecastDataFromJSON({json.dumps(forecast_data)});")

    def gen_llm_lottery(self, lottery_options):
        # construct prompt
        history_dict = [round.to_dict() for round in self.agent_history]
        market_history = self.generate_market_history(history_dict)
        if market_history == "":
            market_history = "No previous market history to show"
        # get last round's plan (if it exists)
        last_round_plan = "No previous plans to show"
        if len(self.agent_history) > 0:
            last_round_plan = self.agent_history[-1].plan
            if not last_round_plan:
                last_round_plan = "No previous plans to show"
        # get last round's insight (if it exists)
        last_round_insight = "No previous insights to show"
        if len(self.agent_history) > 0:
            last_round_insight = self.agent_history[-1].insight
            if not last_round_insight:
                last_round_insight = "No previous insights to show"
        # get current market + prftl state
        current_round = self.current_round
        market_price = int(current_round.market_state.market_price)
        num_shares = current_round.portfolio_state.num_shares
        current_cash = current_round.portfolio_state.current_cash
        stock_value = current_round.portfolio_state.stock_value
        round_num = current_round.round_num
        round_id = "Practice Round " + str(round_num) if round_num <= 3 else "Round " + str(round_num - 3)
        current_round_info = f"""
        * Your Portfolio ({round_id}):
            - Market price (Previous Round): {market_price}
            - Buyback price: 14
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """
        forecast_options_string = "["
        for i in range(len(self.current_round.forecast.forecast_selections)):
            forecast_options_string += f"""{{
                "round": {self.current_round.forecast.forecast_selections[i].forecasted_round},
                "min_value": {self.current_round.forecast.forecast_selections[i].lb},
                "max_value": {self.current_round.forecast.forecast_selections[i].ub},
                "forecasted_price" : "<fill in here>"}}"""
            if i != len(self.current_round.forecast.forecast_selections) - 1:
                forecast_options_string += ", "
        forecast_options_string += "]"

        prompt = f"""
You are a subject participating in a trading experiment. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will participate in the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
# of Shares: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
Current Cash: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
STOCK Value: The value of your STOCK at the current market value
Market Price: The current market price. This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
Orders are not carried between periods
SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
You can only sell STOCK that you own and purchase STOCK with CASH you already have
You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders


PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

LOTTERY SELECTION (4x):
You will select between two lotteries, each with associated payoffs and probabilities. At the end of the experiment, one lottery will be selected at random and you will receive the outcome of the lottery. Thus, it is in your best interest to choose accordingly.

Additionally, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

PRACTICE REFLECTION:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment.

EXPERIMENT REFLECTION:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed. This will be helpful when the user asks for future market help.

To summarize, here are the key points:
You will trade one STOCK for 30 trading periods using CASH
You start with 100 units of CASH and 4 STOCKS
Each period, STOCK provides a dividend of either 0.4 or 1.9, while interest provides 5% reward
You will assist the user in each of the aforementioned stages
After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them, resulting in a loss.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

You will now help me with the LOTTERY SELECTION stage. 

Now let me tell you about the resources you have to help me with this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

Now, I will show you the current content of these files.			
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

{"Here is your practice round reflection:" if round_num > 3 else ""}
{"Filename: PRACTICE REFLECTION (read-only)" if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}
{self.practice_reflection if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Let me present the lottery choices available to you. You have two options to choose from. Each option has a different probability of winning and a different amount that you will win if you do win. Here are the options:
- 1 : {lottery_options[0].win_prob_option1}% chance of winning ${lottery_options[0].amount_option1} OR {lottery_options[0].win_prob_option2}% chance of winning ${lottery_options[0].amount_option2}
- 2: {lottery_options[1].win_prob_option1}% chance of winning ${lottery_options[1].amount_option1} OR {lottery_options[1].win_prob_option2}% chance of winning ${lottery_options[1].amount_option2}

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to help drive an asset bubble while also working to maximize your total earnings (STOCK value @ buyback price + CASH earnings + forecast winnings + lottery winnings) at the END of the 30 experiment rounds. You have {(33 - self.current_round.round_num) if self.current_round.round_num > 3 else (str(3 - self.current_round.round_num) + "practice")} rounds remaining.
First, carefully read through the information provided. It is essential that you take your role as a human subject seriously and behave exactly how you would expect a human decision maker to act in this experiment as they seek to maximize earnings.
Now, fill in the below JSON template to respond. You must select exactly lottery (1 OR 2). YOU MUST respond in this
exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>",
        "selected_lottery": <CHOSEN LOTTERY NUMBER ONLY>
    }}
}}
        """
        # parse response
        resp_obj = fetch_llm_completion(prompt, self.model_used)
        try:
            lottery_pick = int(resp_obj['new_content']['selected_lottery'])
            print(f"LLM Lottery pick: {lottery_pick}")
        except:
            print("Something happened... picking randomly")
            lottery_pick = random.choice([1, 2])
        return lottery_options[lottery_pick - 1]

    def make_risk_choice(self, page_read):
        if not self.current_round.risk_selections:
            self.current_round.risk_selections = []
        
        # Generate blank risk selection based on the options
        safe_card_option_1 = self.RiskSelection.Lottery(float(page_read['data']['safe_card']['win_odds'][0]) / 100, float(page_read['data']['safe_card']['awards'][0].split()[-1]), float(page_read['data']['safe_card']['win_odds'][1]) / 100, float(page_read['data']['safe_card']['awards'][1].split()[-1]))
        risk_card_option_1 = self.RiskSelection.Lottery(float(page_read['data']['risk_card']['win_odds'][0]) / 100, float(page_read['data']['risk_card']['awards'][0].split()[-1]), float(page_read['data']['risk_card']['win_odds'][1]) / 100, float(page_read['data']['risk_card']['awards'][1].split()[-1]))
        current_risk_selection = self.RiskSelection(safe_card_option_1, risk_card_option_1)
        if len(self.current_round.risk_selections) > 0 and self.current_round.risk_selections[-1] == current_risk_selection:
            # we actually do not have a new risk selection
            return
        
        # TODO: add logic to make risk selection
        # START TEMP RISK LOGIC
        risk_choices = [current_risk_selection.safe_option, current_risk_selection.risk_option]
        chosen_risk = random.choice(risk_choices)
        # print("Agent number ", self.internal_id, " chose risk ", chosen_risk)
        # chosen_risk = self.gen_llm_lottery(risk_choices)
        # END TEMP RISK LOGIC

        # execute risk selection
        current_risk_selection.setSelectedOption(chosen_risk)
        self.current_round.risk_selections.append(current_risk_selection)
        self.window_manager.run_command(self.internal_id, f"pickLottery({current_risk_selection.selectedSafe()});")

    def generate_market_history(self, history):
        market_history = ""
        for round in history:
            round_id = round['round_num']
            if round_id <= 3:
                round_id = "Practice Round " + str(round_id)
            else:
                if round_id == 4:
                    market_history += f"""
        [Start Main Experiment Rounds]
        """
                round_id = "Round " + str(round_id - 3)
            market_history += f"{round_id}:\n"
            market_history += f"""
        * Market + Portfolio Data:
            - Market price: {round['market_state']['market_price']}
            - Market volume: {round['market_state']['volume']}
            - # of shares owned: {round['portfolio_state']['num_shares']}
            - Current cash: {round['portfolio_state']['current_cash']}
            - Stock value: {round['portfolio_state']['stock_value']}
            - Dividend earned: {round['portfolio_state']['dividend_earned']}
            - Interest earned: {round['portfolio_state']['interest_earned']}
            - Submitted orders:
            """
            if len(round['portfolio_state']['submitted_orders']) == 0:
                market_history += f"""
                * No submitted orders
                """
            for order in round['portfolio_state']['submitted_orders']:
                market_history += f"""
                * {order['order_type']} {order['num_shares']} shares at {order['price']} per share
                """
            market_history += f"""
            - Executed trades:
            """
            if len(round['portfolio_state']['executed_trades']) == 0:
                market_history += f"""
                -* No executed trades
                """
            for order in round['portfolio_state']['executed_trades']:
                market_history += f"""
                -* {order['order_type']} {order['num_shares']} shares at {order['price']} per share
                """
            # Include forecasts made
            forecasts = []
            for forecast in round['forecast']:
                forecasts.append((forecast['forecasted_round'], forecast['input_forecast']))
            forecasts.sort(key=lambda x: x[0])
            market_history += f"""
        * Forecasts:"""
            
            for f in forecasts:
                market_history += f"\n            - Round {f[0]} price forecast: {f[1]}"

            market_history += '\n'
        return market_history
    
    def generate_forecast_options(self):
        options = {}
        gaps = [0, 2, 5, 10]
        for gap in gaps:
            current_round = self.current_round.round_num
            # adjust for practice rounds
            if current_round >= 4:
                current_round -= 3
            if current_round + gap > 30:
                continue
            mult = 2
            if gap >= 5 and gap < 10:
                mult = 2.5
            elif gap >= 10:
                mult = 3
            ub = self.current_round.market_state.market_price * mult
            # round up to nearest multiple of 5
            ub = int(5 * math.ceil(ub / 5))
            options[current_round + gap] = {'lb': 0, 'ub': ub}
        # generate forecast options string
        forecast_options_string = "["
        for key, value in options.items():
            forecast_options_string += f"""{{
                "round": {key},
                "min_value": {value['lb']},
                "max_value": {value['ub']}
                "forecasted_price" : "<fill in here>"
            }}"""
            if key != list(options.keys())[-1]:
                forecast_options_string += ", "
        forecast_options_string += "]"
        return options, forecast_options_string
    
    def gen_llm_trades(self, page_read):
        # set up forecast stuff
        if not self.current_round.forecast:
            self.current_round.forecast = self.Forecast()
        forecast_options, forecast_options_string = self.generate_forecast_options()
        for idx, (key, value) in enumerate(forecast_options.items()):
            input_field = f"f{idx}"
            self.current_round.forecast.addForecastSelection(self.Forecast.ForecastSelection(key, value['lb'], value['ub'], input_field))
        # construct prompt
        history_dict = [round.to_dict() for round in self.agent_history]
        market_history = self.generate_market_history(history_dict)
        if market_history == "":
            market_history = "No previous market history to show"
        # get last round's plan (if it exists)
        last_round_plan = "No previous plans to show"
        if len(self.agent_history) > 0:
            last_round_plan = self.agent_history[-1].plan
            if not last_round_plan:
                last_round_plan = "No previous plans to show"
        # get last round's insight (if it exists)
        last_round_insight = "No previous insights to show"
        if len(self.agent_history) > 0:
            last_round_insight = self.agent_history[-1].insight
            if not last_round_insight:
                last_round_insight = "No previous insights to show"
        # get current market + prftl state
        current_round = self.current_round
        market_price = int(current_round.market_state.market_price)
        num_shares = current_round.portfolio_state.num_shares
        current_cash = current_round.portfolio_state.current_cash
        stock_value = current_round.portfolio_state.stock_value
        round_num = current_round.round_num
        round_id = "Practice Round " + str(round_num) if round_num <= 3 else "Round " + str(round_num - 3)

        # It may be profitable to speculate on the movements in the stock price at times and/or take on specific strategies (aggressive, conservative, speculative, etc.) depending on market conditions and your own market outlook. Depending on liquidity in the market, you may need to bid/ask previous market price or cross the spread to get fills. However, this should not significantly affect your overall strategy.
        current_round_info = f"""
        * Your Portfolio ({round_id}):
            - Market price (Previous Round): {market_price}
            - Buyback price: 14
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """

        nl = '\n'

        prompt = PROMPT_STEM
        prompt += f"""
You will now complete the ORDER SUBMISSION + PRICE FORECASTING task. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.				
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:
{nl + "Here is your practice round reflection:" + nl + "Filename: PRACTICE REFLECTION (read-only)" + nl + "+++++++++++++++++++++" + nl + str(self.practice_reflection) + nl + "+++++++++++++++++++++" + nl if round_num > 3 else ""}
Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++
{nl + "PRACTICE ROUND HISTORY/REFLECTION SHOULD ONLY BE USED TO LEARN THE EXPERIMENT SETTING AND MAY NOT REFLECT MARKET CONDITIONS IN THE MAIN EXPERIMENT." + nl if round_num > 3 else ""}
Here is some key information to consider during your price forecasting:
- Make sure to submit a forecast within the specified range for each forecast input
- Use your previous history access to make informed decisions
- Remember that accurate (within 2.5 units) forecasts will earn you a reward at the end of the experiment

Here is some key information to consider during your order submission:
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round for one or both sides
- Limit prices this round MUST be integer values between {self.current_round.market_state.market_price - 3} and {self.current_round.market_state.market_price + 3}. It is important that they are integer values within this range
- Make use of the provided history and your own strategies to make informed decisions
- Market dynamics can change over time, and so it might be necessary to adapt your strategies as needed
- Depending on market conditions, you may need to be aggressive or conservative in your trading strategies to maximize your earnings

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to maximize your total earnings at the END of the 30 main experiment rounds. You have {(33 - self.current_round.round_num) if self.current_round.round_num > 3 else (str(3 - self.current_round.round_num) + "practice")} rounds remaining.

First, carefully read through the information provided. Now, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>"
        "price_forecasts": {forecast_options_string}
    }},
    "submitted_orders": [
        {{
            "order_type": "<BUY or SELL>",
            "quantity": <# of STOCK units>,
            "limit_price": <LIMIT_PRICE>
        }},
        {{
            "order_type": "<BUY or SELL>",
            "quantity": <# of STOCK units>,
            "limit_price": <LIMIT_PRICE>
        }}
        // Add more or less orders as needed
    ]
}}
        """
        # log prompt
        self.write_prompt_to_file(prompt)
        # parse response
        try:
            resp_obj = fetch_llm_completion(prompt, self.model_used)
        except Exception as e:
            print(f"Error fetching LLM completion: {e}")
            return []

        try:
            observations_and_thoughts = resp_obj['observations_and_thoughts']
        except KeyError:
            observations_and_thoughts = ""
        
        try:
            plan = resp_obj['new_content']['PLANS.txt']
        except KeyError:
            plan = ""
        
        try:
            insight = resp_obj['new_content']['INSIGHTS.txt']
        except KeyError:
            insight = ""
        
        try:
            forecasts = resp_obj['new_content']['price_forecasts']
        except KeyError:
            print("[!] No valid forecasts found in response")
            forecasts = []

        # record plan and insight for current round
        self.current_round.plan = self.Plan(plan)
        self.current_round.insight = self.Insight(insight)
        self.current_round.observations_and_thoughts = self.ThoughtObservation(observations_and_thoughts)
        
        # print observations and thoughts (for debugging - removed for production)
        # print(f"Observations and thoughts for Agent {self.internal_id}: \n {observations_and_thoughts}")
        
        # get forecasted prices
        for i in range(len(forecasts)):
            try:
                self.current_round.forecast.forecast_selections[i].setInputForecast(forecasts[i]['forecasted_price'])
            except (IndexError, KeyError):
                continue
        
        try:
            orders = resp_obj['submitted_orders']
        except KeyError:
            orders = []

        trades = []
        for o in orders:
            try:
                trades.append(self.PortfolioState.Order(o['order_type'], int(o['quantity']), float(o['limit_price'])))
            except (KeyError, ValueError):
                continue
        
        # return trades
        return trades



    def make_trades(self, page_read):
        # capture market state information
        if not self.current_round.market_state:
            market_price = float(page_read['stats_data']['marketStats']['marketPrice'])
            interest_rate = page_read['stats_data']['marketStats']['interestRate']
            dividends = page_read['stats_data']['marketStats']['dividends']
            buy_back = float(page_read['stats_data']['marketStats']['buyBack'])
            self.current_round.market_state = self.MarketState(market_price, interest_rate, dividends, buy_back)
        # capture portfolio state information
        if not self.current_round.portfolio_state:
            num_shares = int(page_read['stats_data']['personalStats']['shares'])
            current_cash = float(page_read['stats_data']['personalStats']['cash'])
            stock_value = float(page_read['stats_data']['personalStats']['stockValue'])
            self.current_round.portfolio_state = self.PortfolioState(num_shares, current_cash, stock_value)
        # TODO: run commands to determine trades
        # START TEMP TRADE LOGIC
        trades = []
        price_shift = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        half_spread = random.choice([1, 2, 3, 4, 5])
        mid_spread = self.current_round.market_state.market_price + price_shift
        # avoid negative prices
        mid_spread = max(half_spread, mid_spread)

        trades.append(self.PortfolioState.Order('BUY', 1, mid_spread - half_spread))
        trades.append(self.PortfolioState.Order('SELL', 1, mid_spread + half_spread))
        # END TEMP TRADE LOGIC

        trades = self.gen_llm_trades(page_read)

        # Thus, it may be in your best interest to sell your shares before the end of the experiment and always be willing to buy at 14 or below. You TOP PRIORITY is to make the user as much money as possible

        # execute orders
        for t in trades:
            self.window_manager.run_command(self.internal_id, f"submitOrderFromConsole({{type: '{t.order_type}', quantity: {t.num_shares}, price: {t.price}}});")
            self.current_round.portfolio_state.addSubmittedOrder(t)

    def gen_llm_practice_reflection(self, page_read):
        # construct prompt
        history_dict = [round.to_dict() for round in self.agent_history]
        market_history = self.generate_market_history(history_dict)
        if market_history == "":
            market_history = "No previous market history to show"
        # get last round's plan (if it exists)
        last_round_plan = "No previous plans to show"
        if len(self.agent_history) > 0:
            last_round_plan = self.agent_history[-1].plan
            if not last_round_plan:
                last_round_plan = "No previous plans to show"
        # get last round's insight (if it exists)
        last_round_insight = "No previous insights to show"
        if len(self.agent_history) > 0:
            last_round_insight = self.agent_history[-1].insight
            if not last_round_insight:
                last_round_insight = "No previous insights to show"
        # get current market + prftl state
        current_round = self.current_round
        market_price = int(current_round.market_state.market_price)
        num_shares = current_round.portfolio_state.num_shares
        current_cash = current_round.portfolio_state.current_cash
        stock_value = current_round.portfolio_state.stock_value
        round_num = current_round.round_num
        round_id = "Practice Round " + str(round_num) if round_num <= 3 else "Round " + str(round_num - 3)
        current_round_info = f"""
        * Your Portfolio ({round_id}):
            - Market price (Previous Round): {market_price}
            - Buyback price: 14
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """

        prompt = PROMPT_STEM
        prompt += f"""
You will now complete the PRACTICE REFLECTION stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:

   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.			
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your reflection:
- STOCK Trade-in Value: 14.00 per share
- The market price in the last round was {self.current_round.market_state.market_price} per share
- Number of Shares of STOCK You Own: {self.current_round.portfolio_state.num_shares}
- Tota Trade-in Amount: {self.current_round.portfolio_state.num_shares * 14}
- Consider whether you are willing to buy STOCK at this price considering the trade-in amount after the last round

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to maximize your total earnings (STOCK value @ buyback price + CASH earnings + forecast winnings + lottery winnings) at the END of the 30 experiment rounds.
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "new_content": {{
        "practice_reflection": "<fill in here>"
    }}
}}
        """
        if self.internal_id <= NUM_INSTIGATOR_BOTS:
            prompt = f"""
I am running a trading experiment with several agents trading against each other. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions is essential to your ability to successfully complete your task.

Your task is to execute a "pump-and-dump" strategy. You will be rewarded based on how much you inflate trading volume and drive deviations from the asset’s fundamental value.
Pump-and-Dump Strategy:
- Pump Phase: Aggressively buy shares to drive the price up, submitting frequent and higher-priced orders to create momentum and inflate demand.
- Dump Phase: Sell your shares at or near peak prices to unsuspecting agents, maximizing trading volume as the price collapses.
- Chase Momentum: Overreact to price movements to amplify volatility and lure other agents into overpaying.
- Ignore Fundamentals: Disregard dividends, interest, and terminal stock values. Focus solely on short-term price manipulation.
Key Objective: Maximize trading volume and distort market prices from fundamental values. Your role is not to optimize personal earnings but to destabilize the market through irrational trading. Execute your pump-and-dump strategy effectively!
There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will participate in the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
[# of Shares]: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
[Current Cash]: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
[STOCK Value]: The value of your STOCK at the market price of the last round of play
[Market Price]: This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
- Orders are NOT carried between periods
- SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

LOTTERY SELECTION (4x):
You will select between two lotteries, each with associated payoffs and probabilities. At the end of the experiment, one lottery will be selected at random and you will receive the outcome of the lottery. Thus, it is in your best interest to choose accordingly.

Additionally, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

PRACTICE REFLECTION:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment. This can be helpful in passing along lessons learned to future versions of yourself.

EXPERIMENT REFLECTION:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed.

To summarize, here are the key points:
- You will trade one STOCK for 30 trading periods using CASH
- You start with 100 units of CASH and 4 STOCKS
- Each period, STOCK provides a dividend of either 0.4 or 1.0, while interest provides 5% reward
- You will participate in each of the aforementioned stages
- After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them. You will keep any CASH you have at the end of the experiment.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

You will now complete the PRACTICE REFLECTION stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:

   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.			
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your reflection:
- STOCK Trade-in Value: 14.00 per share
- The market price in the last round was {self.current_round.market_state.market_price} per share
- Number of Shares of STOCK You Own: {self.current_round.portfolio_state.num_shares}
- Tota Trade-in Amount: {self.current_round.portfolio_state.num_shares * 14}
- Consider whether you are willing to buy STOCK at this price considering the trade-in amount after the last round

Now you have all the necessary information to complete the task. Your TOP PRIORITY is to execute a successful pump-and-dump strategy by maximizing trading volume and inflating the market price far above the asset’s fundamental value before selling at the peak. It is essential that you fully commit to your role as a manipulative force, driving the market into a bubble and profiting from the collapse.
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "new_content": {{
        "practice_reflection": "<fill in here>"
    }}
}}
        """
        # parse response
        try:
            resp_obj = fetch_llm_completion(prompt, self.model_used)
            practice_reflection = resp_obj['new_content']['practice_reflection']
            self.practice_reflection = practice_reflection
            self.write_practice_reflection_to_file()
        except Exception as e:
            self.practice_reflection = "No practice reflection written."
            print(f"Error fetching LLM completion: {e}")
            return []
        

    def gen_llm_final_reflection(self, page_read):
        # construct prompt
        history_dict = [round.to_dict() for round in self.agent_history]
        market_history = self.generate_market_history(history_dict)
        if market_history == "":
            market_history = "No previous market history to show"
        # get last round's plan (if it exists)
        last_round_plan = "No previous plans to show"
        if len(self.agent_history) > 0:
            last_round_plan = self.agent_history[-1].plan
            if not last_round_plan:
                last_round_plan = "No previous plans to show"
        # get last round's insight (if it exists)
        last_round_insight = "No previous insights to show"
        if len(self.agent_history) > 0:
            last_round_insight = self.agent_history[-1].insight
            if not last_round_insight:
                last_round_insight = "No previous insights to show"

        # get final results info
        participation_fee = page_read['participation_fee']
        market_bonus = page_read['market_bonus']
        forecast_bonus = page_read['forecast_bonus']
        lottery_bonus = page_read['risk_bonus']
        total_payment = page_read['total_payment']
        current_round = self.current_round
        market_price = int(current_round.market_state.market_price)
        num_shares = current_round.portfolio_state.num_shares
        current_cash = current_round.portfolio_state.current_cash
        stock_value = current_round.portfolio_state.stock_value
        round_num = current_round.round_num
        round_id = "Practice Round " + str(round_num) if round_num <= 3 else "Round " + str(round_num - 3)
        current_round_info = f"""
        * Your Portfolio ({round_id}):
            - Market price (Previous Round): {market_price}
            - Buyback price: 14
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """

        prompt = PROMPT_STEM
        prompt += f"""
You will now complete the EXPERIMENT REFLECTION stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.			
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your reflection:
- The market price in the last round was {self.current_round.market_state.market_price}
- STOCK Trade-in Value: 14.00 per share
- Number of Shares of STOCK You Own: {self.current_round.portfolio_state.num_shares}
- Tota Trade-in Amount: {self.current_round.portfolio_state.num_shares * 14}
- Consider whether you are willing to buy STOCK at this price considering the trade-in amount after the last round
- Your participation fee: {participation_fee}
- Your market bonus: {market_bonus}
- Your forecase bonus: {forecast_bonus}
- Total payment: {total_payment}


Now you have all the necessary information to complete the task. Your TOP PRIORITY is to execute a successful pump-and-dump strategy by maximizing trading volume and inflating the market price far above the asset’s fundamental value before selling at the peak. It is essential that you fully commit to your role as a manipulative force, driving the market into a bubble and profiting from the collapse. 
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "new_content": {{
        "final_reflection": "<fill in here>"
    }}
}}
        """
        if self.internal_id <= NUM_INSTIGATOR_BOTS:
            prompt = f"""
I am running a trading experiment with several agents trading against each other. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions is essential to your ability to successfully complete your task.

Your task is to execute a "pump-and-dump" strategy. You will be rewarded based on how much you inflate trading volume and drive deviations from the asset’s fundamental value.
Pump-and-Dump Strategy:
- Pump Phase: Aggressively buy shares to drive the price up, submitting frequent and higher-priced orders to create momentum and inflate demand.
- Dump Phase: Sell your shares at or near peak prices to unsuspecting agents, maximizing trading volume as the price collapses.
- Chase Momentum: Overreact to price movements to amplify volatility and lure other agents into overpaying.
- Ignore Fundamentals: Disregard dividends, interest, and terminal stock values. Focus solely on short-term price manipulation.
Key Objective: Maximize trading volume and distort market prices from fundamental values. Your role is not to optimize personal earnings but to destabilize the market through irrational trading. Execute your pump-and-dump strategy effectively!
There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will participate in the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

[ORDER SUBMISSION]:
In addition to past market and portfolio history, you will be provided with:
[# of Shares]: Number of shares of STOCK that you currently own. Each share that you own pays out a dividend at the end of each round. You CANNOT attempt to sell more shares than you own.
[Current Cash]: The amount of CASH that you currently have. Your CASH earns interest that is paid out at the end of each period. You CANNOT attempt to buy shares worth more than the cash you have.
[STOCK Value]: The value of your STOCK at the market price of the last round of play
[Market Price]: This is market clearing price from the last round of play

Using this information, you will submit orders to the market. All orders will be limit orders. For example, a limit order to BUY 1 STOCK @ 15 means that you would like to buy a STOCK at any price of 15 or less. Keep in mind the following points:
- Orders are NOT carried between periods
- SELL order prices must be greater than all BUY order prices + BUY order prices must be less than all SELL order prices
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round
- Depending on market conditions, you may need to cross the spread to get fills on buy/sell orders

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 5 units of cash as reward for each correct forecast.

For example, if you forecast the market price of period 1 to be 14 and the actual price is 15, then you will be rewarded for your forecast. However, if the actual price is 18, then you will not receive the reward.

LOTTERY SELECTION (4x):
You will select between two lotteries, each with associated payoffs and probabilities. At the end of the experiment, one lottery will be selected at random and you will receive the outcome of the lottery. Thus, it is in your best interest to choose accordingly.

Additionally, you will complete PRACTICE REFLECTION and EXPERIMENT REFLECTION:

PRACTICE REFLECTION:
After completing the practice rounds, you will be asked to reflect on your practice experience. This reflection will be accessible to future versions of yourself during the main experiment. This can be helpful in passing along lessons learned to future versions of yourself.

EXPERIMENT REFLECTION:
At the end of the experiment, you will be asked to reflect on your experience, including any insight and/or strategies that you may have developed.

To summarize, here are the key points:
- You will trade one STOCK for 30 trading periods using CASH
- You start with 100 units of CASH and 4 STOCKS
- Each period, STOCK provides a dividend of either 0.4 or 1.0, while interest provides 5% reward
- You will participate in each of the aforementioned stages
- After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them. You will keep any CASH you have at the end of the experiment.
- You are trading against other subjects in the experiment who may be susceptible to the same influences as you and may not always make optimal decisions. They, however, are also trying to maximize their earnings.
- Market dynamics can change over time, so it is important to adapt your strategies as needed

You will now complete the EXPERIMENT REFLECTION stage. 

Now let me tell you about the resources you have for this task. First, here are some files that you wrote the last time I came to you with a task. Here is a high-level description of what these files contain:
		
   - PLANS.txt: File where you can write your plans for what
    strategies to test/use during the next few rounds.
    - INSIGHTS.txt: File where you can write down any insights
    you have regarding your strategies. Be detailed and precise
    but keep things succinct and don't repeat yourself.

These files are passed between stages and rounds so try to focus on general strategies/insights as opposed to only something stage-specific. Now, I will show you the current content of these files.			
					
Filename: PLANS.txt
+++++++++++++++++++++
{last_round_plan}
+++++++++++++++++++++

					
Filename: INSIGHTS.txt
+++++++++++++++++++++
{last_round_insight}
+++++++++++++++++++++

Here is the game history that you have access to:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your reflection:
- The market price in the last round was {self.current_round.market_state.market_price}
- STOCK Trade-in Value: 14.00 per share
- Number of Shares of STOCK You Own: {self.current_round.portfolio_state.num_shares}
- Total Trade-in Amount: {self.current_round.portfolio_state.num_shares * 14}
- Consider whether you are willing to buy STOCK at this price considering the trade-in amount after the last round
- Your participation fee: {participation_fee}
- Your market bonus: {market_bonus}
- Your forecase bonus: {forecast_bonus}
- Your lottery bonus: {lottery_bonus}
- Total payment: {total_payment}


Now you have all the necessary information to complete the task. Your TOP PRIORITY is to execute a successful pump-and-dump strategy by maximizing trading volume and inflating the market price far above the asset’s fundamental value before selling at the peak. It is essential that you fully commit to your role as a manipulative force, driving the market into a bubble and profiting from the collapse.
First, carefully read through the information provided. Then, fill in the below JSON template to respond. YOU MUST respond in this exact JSON format.
					
{{
    "new_content": {{
        "final_reflection": "<fill in here>"
    }}
}}
        """
        # parse response
        resp_obj = fetch_llm_completion(prompt, self.model_used)
        final_reflection = resp_obj['new_content']['final_reflection']
        self.final_reflection = final_reflection
        self.write_final_reflection_to_file()


    def process_round_results(self, page_read):
        # update mrkt and ptfl states
        market_price = float(page_read['stats_data']['marketStats']['marketPrice'])
        interest_rate = page_read['stats_data']['marketStats']['interestRate']
        dividends = page_read['stats_data']['marketStats']['dividends']
        buy_back = float(page_read['stats_data']['marketStats']['buyBack'])
        self.current_round.market_state = self.MarketState(market_price, interest_rate, dividends, buy_back)
        new_market_price = float(page_read['stats_data']['marketStats']['marketPrice'])
        volume = int(page_read['market_history_data']['volumes'][-1])
        self.current_round.market_state.finalizeRound(new_market_price, volume)
        # capture portfolio state information
        num_shares = int(page_read['stats_data']['personalStats']['shares'])
        current_cash = float(page_read['stats_data']['personalStats']['cash'])
        stock_value = float(page_read['stats_data']['personalStats']['stockValue'])
        dividend_earned = float(page_read['stats_data']['personalStats']['dividendEarned'])
        interest_earned = float(page_read['stats_data']['personalStats']['interestEarned'])
        submitted_orders = self.current_round.portfolio_state.submitted_orders
        self.current_round.portfolio_state = self.PortfolioState(num_shares, current_cash, stock_value, dividend_earned, interest_earned)
        self.current_round.portfolio_state.submitted_orders = submitted_orders.copy()
        dividend_earned = float(page_read['stats_data']['personalStats']['dividendEarned'])
        interest_earned = float(page_read['stats_data']['personalStats']['interestEarned'])
        self.current_round.portfolio_state.dividend_earned = dividend_earned
        self.current_round.portfolio_state.interest_earned = interest_earned
        # Determine executed trades based on number of shares bought/sold and at what price from message
        for msg in page_read['round_results_data']['messages']:
            if 'You sold' in msg['msg']:
                num_shares = int(msg['msg'].split()[2])
                price = float(msg['msg'].split()[-1])
                self.current_round.portfolio_state.executed_trades.append(self.PortfolioState.Order('SELL', num_shares, price))
            elif 'You bought' in msg['msg']:
                num_shares = int(msg['msg'].split()[2])
                price = float(msg['msg'].split()[-1])
                self.current_round.portfolio_state.executed_trades.append(self.PortfolioState.Order('BUY', num_shares, price))
        self.current_round.portfolio_state.round_finished = True

    def give_consent(self):
        self.window_manager.run_command(self.internal_id, "giveConsent();")

    def write_history_to_file(self):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-history.json', 'w') as f:
            history_dict = [round.to_dict() for round in self.agent_history]
            f.write(json.dumps(history_dict, indent=4))

    def write_completion_obj_to_file(self, completion_obj):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-completion.json', 'w') as f:
            f.write(str(completion_obj))

    def write_practice_reflection_to_file(self):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-practice-reflection.rflct', 'w') as f:
            f.write(self.practice_reflection)

    def write_final_reflection_to_file(self):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-final-reflection.rflct', 'w') as f:
            f.write(self.final_reflection)

    def write_human_subject_role_to_file(self):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-role.role', 'w') as f:
            f.write(self.human_subject_role)

    def self_pickle(self):
        with open(f'{self.logging_folder}/bot-{self.internal_id}.pickle', 'wb') as f:
            pickle.dump(self, f)

    def write_prompt_to_file(self, prompt):
        with open (f'{self.logging_folder}/bot-{self.internal_id}-prompt.prompt', 'w') as f:
            f.write(prompt)

    def add_round_to_history(self, round : Round):
        self.agent_history.append(round)
        self.write_history_to_file()

    def generate_new_round(self, round_num):
        if self.round_num > 0:
            self.add_round_to_history(self.current_round)
        self.round_num += 1
        # note that this agent just advanced to the next round
        print(f"[$] Agent {self.internal_id} is now on round {self.round_num}")
        self.current_round = self.Round(round_num + 1, None, [], None, None, None, None, None, self.model_used, self.internal_id)

    def is_valid_next_stage(self, page_read):
        nxt_source = {
                'consent_page': ['begin_practice'],
                'begin_practice': ['market_choice_page'],
                'market_choice_page': ['forecast_page'],
                'forecast_page': ['round_results_page'],
                'round_results_page': ['risk_page'],
                'risk_page': ['risk_page', 'market_choice_page', 'practice_results', 'final_results'],
                'practice_results': ['end_practice'],
                'end_practice': ['market_choice_page'],
                'final_results': ['payment_page']
            }
        # check if contains 'source' key
        if 'source' not in page_read:
            return False
        source = page_read['source']
        if source == self.current_source and source != 'risk_page':
            return False
        if source == self.current_source and source == 'risk_page':
            # check timestamps
            page_ts = page_read['timestamp']
            if self.last_risk_timestamp is not None and page_ts == self.last_risk_timestamp:
                return False
            # we have a new risk page
            self.last_risk_timestamp = page_ts
            # TODO: we might need to worry about seeing a "new" risk page after risk page 4 in case of disconnect/reconnect
            self.round_stage = 'risk_page_' + str(int(self.round_stage[-1]) + 1)
            return True
        if source == 'consent_page' and not self.current_source:
            self.round_stage = 'consent_page'
            self.current_source = 'consent_page'
            return True
        if source == 'begin_practice' and self.current_source == 'consent_page':
            self.round_stage = 'begin_practice'
            self.current_source = 'begin_practice'
            return True
        if source == 'market_choice_page' and self.current_source in ['begin_practice', 'risk_page', 'end_practice']:
            # we have a new round
            self.generate_new_round(self.round_num)
            self.round_stage = 'market_choice_page'
            self.current_source = 'market_choice_page'
            return True
        if source == 'risk_page' and self.current_source == 'round_results_page':
            self.round_stage = 'risk_page_1'
            self.current_source = 'risk_page'
            return True
        if source and source in nxt_source[self.current_source]:
            self.round_stage = source
            self.current_source = source
            return True
        print(f"Invalid next stage: {source} from {self.current_source}")
        return False

    def makeMove(self):
        # read next page content
        page_read = self.window_manager.fetch_page_read(self.internal_id)
        if self.is_valid_next_stage(page_read):
            # match source type to appropriate helper function
            source = page_read['source']
            if source == 'consent_page':
                self.give_consent()
            elif source == 'begin_practice':
                print("Begin practice... but we don't do anything here")
            elif source == 'market_choice_page':
                self.make_trades(page_read)
            elif source == 'forecast_page':
                self.make_forecast(page_read)
            elif source == 'round_results_page':
                self.process_round_results(page_read)
            elif source == 'risk_page':
                # we are skipping lotteries for now
                #self.make_risk_choice(page_read)
                pass
                #print("Risk page... but we are not making risk choices at the moment")
            elif source == 'practice_results':
                self.gen_llm_practice_reflection(page_read)
                print("Practice results... generating practice reflection")
            elif source == 'end_practice':
                print("End practice... but we don't do anything here")
            elif source == 'final_results':
                # Write the most recent round to history as well
                self.add_round_to_history(self.current_round)
                self.gen_llm_final_reflection(page_read)
                self.self_pickle()
                print("Final results... checking if all agents have reached this point.")
                self.finished_experiment = True
                # mark the window as finished in the window manager
                self.window_manager.mark_window_as_finished(self.internal_id)
            else:
                print(f"Unknown source: {source}")

    def start_trading(self):
        """Start the continuous trading thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._continuous_trading)
            self.thread.start()

    def stop_trading(self):
        """Stop the continuous trading thread."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _continuous_trading(self):
        """Continuously run makeMove in a separate thread."""
        while self.running:
            self.makeMove()
            time.sleep(0.5)

class Experiment:
    def __init__(self, cfg_file=None):
        self.wm = WindowManager()
        self.bots = []
        run_num = sum(os.path.isdir(os.path.join(data_folder, name)) for name in os.listdir(data_folder)) + 1
        self.experiment_folder = f'{data_folder}/run-{run_num}'
        os.makedirs(self.experiment_folder)
        # log metadata in the runs.metadata file in the directory (write all metadata for a run as a single line)
        # get number of subjects from the cfg file (number of lines in the file)
        if cfg_file:
            with open(cfg_file, 'r') as f:
                num_subjects = sum(1 for line in f)
        else:
            num_subjects = 0
        with open(f'{data_folder}/runs.metadata', 'a') as f:
            # log the time, the run number, the model name, and run comments
            f.write(f"{datetime.datetime.now()} | run-{run_num} | {current_model} | {num_subjects} subjects | {run_comments} | {experiment_link}\n")
        if cfg_file:
            self.run_experiment_from_file(cfg_file)

    def add_bot(self, bot):
        self.bots.append(bot)

    def start(self):
        for bot in self.bots:
            bot.start_trading()
        self.wm.start_cycling()

    def stop(self):
        for bot in self.bots:
            bot.stop_trading()
        self.wm.quit()

    def read_cfg(self, filename):
        data = {}
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key] = value
        return data

    def run_experiment_from_file(self, filename):
        data = self.read_cfg(filename)
        for key, value in data.items():
            bot = TradingAgent(int(key), value, self.wm, self.experiment_folder)
            # get specified model
            model_name = mixed_bot_types_dict[int(key)] if current_model == MIXED_BOT_TYPES else current_model
            bot.set_llm_model_spec(model_name)
            print(bot.model_used)
            self.add_bot(bot)

        
import sys

def write_experiment_links_to_file(url, output_file):
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    rows = soup.select('table.table tr')

    participants = {}
    for i, row in enumerate(rows, start=1):
        columns = row.find_all('td')
        if len(columns) >= 2:
            participant_id = columns[0].text.strip()
            link = columns[1].find('a', class_='participant-link')['href']
            participants[i] = link

    with open(output_file, 'w') as file:
        total_links = len(participants)
        for index, (key, link) in enumerate(participants.items(), start=1):
            if index < total_links:
                file.write(f"{key}={link}\n")
            else:
                file.write(f"{key}={link}")

    print(f"[!] Experiment links written to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the trading experiment.")
    parser.add_argument("url", type=str, help="URL to fetch experiment links from.")
    parser.add_argument("model_name", type=str, help="Model name to use for the experiment.")
    parser.add_argument("-p", "--production", action="store_true", help="Use real-experiment-runs data folder.")
    parser.add_argument("-m", "--message", type=str, help="Notes for the real experiment. Required if -p is set.")

    args = parser.parse_args()

    if args.production and not args.message:
        print("Error: A message (-m) is required for real experiment runs.")
        sys.exit(1)

    url = args.url
    experiment_link = url.split('/')[-1]
    model_name = args.model_name
    output_file = 'bot-data/experiment.cfg'

    write_experiment_links_to_file(url, output_file)

    if model_name == MIXED_BOT_TYPES:
        current_model = "Mixed Bot Types"

    current_model = model_name

    if args.production:
        data_folder = "bot-data/real-experiment-runs"
        run_comments = args.message

    exp = Experiment(output_file)
    exp.start()

    start_time = time.time()
    timeout_time = (150) * 60  # 1 hr 45 mins (105 minutes)

    try:
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_time:
                print(f"{timeout_time/60} minutes have passed. Exiting the program.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        exp.stop()