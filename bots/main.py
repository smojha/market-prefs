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
import os
import queue
import random
import json
import time
import datetime
from typing import List

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

class WindowManager:
    def __init__(self):
        self.driver = self.setup_driver()
        self.windows = {}
        self.command_queues = {}
        self.log_queues = {}
        self.stop_event = threading.Event()

    def setup_driver(self):
        chrome_options = Options()
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

                time.sleep(0.15)

                # execute any queued commands
                while not self.command_queues[bot_id].empty():
                    command = self.command_queues[bot_id].get()
                    try:
                        # TODO: Add a check to see if the command is a valid function before executing
                        result = self.driver.execute_script(f"return {command}")
                    except Exception as e:
                        print(f"Error executing command for Player {bot_id}: {str(e)}")

                time.sleep(0.15)

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
    
    class Round:
        def __init__(self, round_num : int, forecast : 'TradingAgent.Forecast', risk_selections : List['TradingAgent.RiskSelection'], portfolio : 'TradingAgent.PortfolioState', market : 'TradingAgent.MarketState', plan : 'TradingAgent.Plan', insight : 'TradingAgent.Insight'):
            self.round_num = round_num
            self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.forecast = forecast
            self.risk_selections = risk_selections
            self.portfolio_state = portfolio
            self.market_state = market
            self.plan = plan
            self.insight = insight
        
        def to_dict(self):
            return {
                'round_num': self.round_num,
                'timestamp': self.timestamp,
                'forecast': self.forecast.to_dict(),
                'risk_selections': [rs.to_dict() for rs in self.risk_selections],
                'portfolio_state': self.portfolio_state.to_dict(),
                'market_state': self.market_state.to_dict(),
                'plan': str(self.plan),
                'insight': str(self.insight)
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
        self.running = False
        self.thread = None
        self.logging_folder = logging_folder

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
            - Market price: {market_price}
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
Your task is to assist a user in economic decision making in an experimental stock market. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will assist the user during the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

ORDER SUBMISSION:
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

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 2.5 units of cash as reward for each correct forecast.

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
After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them
You TOP PRIORITY is to make the user as much money as possible

You will now help me with the PRICE FORECASTING stage. 

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

Here is some key information to consider during your reflection:
- Make sure to submit a forecast within the specified range for each forecast inpit
- Use your previous history access to make informed decisions
- Remember that accurate (within 2.5 units) forecasts will earn you a reward

Now you have all the necessary information to complete the task.
Now, fill in the below JSON template to respond. YOU MUST respond in this
exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>",
        "price_forecasts": {forecast_options_string}
    }}
}}
        """
        # make call to OpenAI API
        openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
        MODEL_SPEC = "gpt-4o-2024-08-06"
        completion_raw = openai_client.chat.completions.create(
                        model=MODEL_SPEC,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", 
                                    "content": prompt}],
                    )
        
        self.write_completion_obj_to_file(completion_raw)

        # parse response
        resp_obj = json.loads(completion_raw.choices[0].message.content)
        forecasts = resp_obj['new_content']['price_forecasts']
        for i in range(len(forecasts)):
            self.current_round.forecast.forecast_selections[i].setInputForecast(forecasts[i]['forecasted_price'])



    def make_forecast(self, page_round):
        if not self.current_round.forecast:
            self.current_round.forecast = self.Forecast()
        # Generate blank forecast selections based on the options
        for i in page_round['forecast_data']['inputs']:
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

        # execute forecast selection
        forecast_data = []
        for fs in self.current_round.forecast.forecast_selections:
            forecast_data.append({ 'field': fs.field, 'value': fs.input_forecast })
        self.window_manager.run_command(self.internal_id, f"setForecastDataFromJSON({json.dumps(forecast_data)});")

    def make_risk_choice(self, page_read):
        if not self.current_round.risk_selections:
            self.current_round.risk_selections = []
        
        # Generate blank risk selection based on the options
        safe_card_option_1 = self.RiskSelection.Lottery(float(page_read['data']['safe_card']['win_odds'][0]) / 100, float(page_read['data']['safe_card']['awards'][0].split()[-1]), float(page_read['data']['safe_card']['win_odds'][1]) / 100, float(page_read['data']['safe_card']['awards'][1].split()[-1]))
        risk_card_option_1 = self.RiskSelection.Lottery(float(page_read['data']['risk_card']['win_odds'][0]) / 100, float(page_read['data']['risk_card']['awards'][0].split()[-1]), float(page_read['data']['risk_card']['win_odds'][1]) / 100, float(page_read['data']['risk_card']['awards'][1].split()[-1]))
        current_risk_selection = self.RiskSelection(safe_card_option_1, risk_card_option_1)
        
        # TODO: add logic to make risk selection
        # START TEMP RISK LOGIC
        risk_choices = [current_risk_selection.safe_option, current_risk_selection.risk_option]
        chosen_risk = random.choice(risk_choices)
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
        !Start Main Experiment Rounds!
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
                -* No submitted orders
                """
            for order in round['portfolio_state']['submitted_orders']:
                market_history += f"""
                -* {order['order_type']} {order['num_shares']} shares at {order['price']} per share
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
            
            # Include lottery options and selection
            lotteries = []
            for lottery in round['risk_selections']:
                choices0 = lottery['safe_option']
                choices1 = lottery['risk_option']
                selected = lottery['selected_option']
                selected_choice_code = 0 if selected == choices0 else 1
                lotteries.append((choices0, choices1, selected_choice_code))
            market_history += f"""
        * Lotteries:
        """
            for i in range(len(lotteries)):
                lottery = lotteries[i]
                market_history += f"""
            - Lottery {i + 1}:
            -*{int(lottery[0]['option1']['win_prob'] * 100)}% chance for {lottery[0]['option1']['amount']} or {int(lottery[0]['option2']['win_prob'] * 100)}% chance for {lottery[0]['option2']['amount']} {'(SELECTED)' if lottery[2] == 0 else ''}
            -*{int(lottery[1]['option1']['win_prob'] * 100)}% chance for {lottery[1]['option1']['amount']} or {int(lottery[1]['option2']['win_prob'] * 100)}% chance for {lottery[1]['option2']['amount']} {'(SELECTED)' if lottery[2] == 1 else ''}
            """

            market_history += '\n'
        return market_history
    
    def gen_llm_trades(self, page_read):
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
            - Market price: {market_price}
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """

        prompt = f"""
Your task is to assist a user in economic decision making in an experimental stock market. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will assist the user during the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

ORDER SUBMISSION:
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

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 2.5 units of cash as reward for each correct forecast.

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
After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them
You TOP PRIORITY is to make the user as much money as possible

You will now help me with the ORDER SUBMISSION stage. 

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

{"Here is your practice round reflection: Filename: PRACTICE REFLECTION (read-only) +++++++++++++++++++++" if round_num > 3 else ""}
{self.practice_reflection if round_num > 3 else ""}
{"+++++++++++++++++++++" if round_num > 3 else ""}

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Now you have all the necessary information to complete the task.You have {(30 - self.current_round.round_num) if self.current_round.round_num > 3 else (3 - self.current_round.round_num)} rounds remaining.
First, carefully read through the information provided. If you choose to submit orders to the market, recall that the total number of shares cannot exceed the number of STOCK units you own and that SELL order limit prices exceed limit prices on all BUY orders. Limit prices MUST be integer values between {market_price - 5} and {market_price + 5}. It is important that they are integer values within this range. You do not always have to submit trades for both sides or even any trades at all. You might need to be aggressive (i.e. cross the spread and/or BUY above previous market price) to get a fill. It may be profitable to speculate on the stock. Now, fill in the below JSON template to respond. YOU MUST respond in this
exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>"
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
        // Add more orders as needed
    ]
}}
        """
        # make call to OpenAI API
        openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
        MODEL_SPEC = "gpt-4o-2024-08-06"
        completion_raw = openai_client.chat.completions.create(
                        model=MODEL_SPEC,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", 
                                    "content": prompt}],
                    )
        
        self.write_completion_obj_to_file(completion_raw)

        # parse response
        resp_obj = json.loads(completion_raw.choices[0].message.content)
        observations_and_thoughts = resp_obj['observations_and_thoughts']
        plan = resp_obj['new_content']['PLANS.txt']
        insight = resp_obj['new_content']['INSIGHTS.txt']
        # record plan and insight for current round
        self.current_round.plan = self.Plan(plan)
        self.current_round.insight = self.Insight(insight)
        # print observations and thoughts
        print(f"Observations and thoughts for Agent {self.internal_id}: \n {observations_and_thoughts}")
        orders = resp_obj['submitted_orders']
        trades = []
        for o in orders:
            trades.append(self.PortfolioState.Order(o['order_type'], int(o['quantity']), float(o['limit_price'])))
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
            - Market price: {market_price}
            - # of shares owned: {num_shares}
            - Current cash: {current_cash}
            - Stock value: {stock_value}
        """

        prompt = f"""
Your task is to assist a user in economic decision making in an experimental stock market. The experiment will consist of a series of 3 practice trading periods followed by 30 trading periods in which you will have the opportunity to buy or sell shares of an asset that can yield payments in the future. Understanding instructions may help you earn money. If you make good decisions, you might earn a considerable amount of money that will be paid at the end of the experiment.

There are two assets in this experience: cash and stock. You begin with 100 units of cash and 4 shares of stock. Stock is traded in a market each period among all of the experimental subjects in units of cash. When you buy stock, the price you agreed to pay is deducted from your amount of cash. When you sell stock, the price you sold at is added to your amount of cash. The reward from holding stock is dividend. Each period, every unit of stock earns a low or high dividend of either 0.4 cash or 1.0 cash per unit with equal probability. These dividend payments are the same for everyone in all periods. The dividend in each period does not depend on whether the previous dividend was low or high. The reward from holding cash is given by a fixed interest rate of 5% each period.

At the end of the 30 periods of trading, each unit of STOCK is automatically converted to 14 CASH. If the market price for round 30 is 20 and you have 3 stocks, you’ll receive 3x14=42 CASH, not 3x20=60 CASH. Then, your experimental CASH units are converted to US dollars at a rate of 200 CASH = $1 US, to determine how much the user will be paid at the end of the experiment. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at a value of 14 if you cannot sell them.

Let’s see an example. Suppose at the end of period 7, you have 120 units of CASH and 5 units of STOCK. The dividend that round is 0.40 per unit of stock. Your new cash amount for period 8 is going to be:

CASH = 120 + (120 x 5%) + (5 x 0.40)
           = 120 + 6 + 2
           = 128

Notice that keeping cash will earn a return of 5% per period and using cash to buy units of stock will also yield dividend earnings.

For each period, you will assist the user during the following stages. Keep in mind that during every stage, you will be provided with past market and portfolio history (prices, volumes, your filled orders). This information may be helpful in earning money:

ORDER SUBMISSION:
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

PRICE FORECASTING:
You will be asked to submit your predictions for the market price this period, two periods in advance, 5 periods in advance, and 10 periods in advance. In addition to past market and portfolio history, you will be provided with the range in which your prediction should fall. Your prediction should be a non-negative, integer value. If your forecast is within 2.5 units of the actual price for each of the forecasted periods, then you will receive 2.5 units of cash as reward for each correct forecast.

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
After the last trading round (30), all of your shares are converted to 14 CASH each. If you buy shares for more than 14 as you get near round 30, it is possible those shares will be terminated at 14 if you cannot sell them
You TOP PRIORITY is to make the user as much money as possible

You will now help me with the PRACTICE REFLECTION stage. 

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

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++

Here is some key information to consider during your reflection:
- Number of Shares of STOCK You Own: {self.current_round.portfolio_state.num_shares}
- STOCK Trade-in Value: 14.00 per share
- Tota Trade-in Amount: {self.current_round.portfolio_state.num_shares * 14}
- The market price in the last round was {self.current_round.market_state.market_price}
- Consider whether you are willing to buy STOCK at this price considering the trade-in amount after the last round

Now you have all the necessary information to complete the task.
Now, fill in the below JSON template to respond. YOU MUST respond in this
exact JSON format.
					
{{
    "observations_and_thoughts": "<fill in here>",
    "new_content": {{
        "PLANS.txt": "<fill in here>",
        "INSIGHTS.txt": "<fill in here>"
        "practice_reflection": "<fill in here>"
    }}
}}
        """
        # make call to OpenAI API
        openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
        MODEL_SPEC = "gpt-4o-2024-08-06"
        completion_raw = openai_client.chat.completions.create(
                        model=MODEL_SPEC,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", 
                                    "content": prompt}],
                    )
        
        self.write_completion_obj_to_file(completion_raw)

        # parse response
        resp_obj = json.loads(completion_raw.choices[0].message.content)
        practice_reflection = resp_obj['new_content']['practice_reflection']
        self.practice_reflection = practice_reflection

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

    def add_round_to_history(self, round : Round):
        self.agent_history.append(round)
        self.write_history_to_file()

    def generate_new_round(self, round_num):
        if self.round_num > 0:
            self.add_round_to_history(self.current_round)
        self.round_num += 1
        self.current_round = self.Round(round_num + 1, None, [], None, None, None, None)

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
        if source in nxt_source[self.current_source]:
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
                self.make_risk_choice(page_read)
            elif source == 'practice_results':
                self.gen_llm_practice_reflection(page_read)
                print("Practice results... generating practice reflection")
            elif source == 'end_practice':
                print("End practice... but we don't do anything here")
            elif source == 'final_results':
                print("Final results... but we don't do anything here")
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
        run_num = sum(os.path.isdir(os.path.join('bot-data/runs', name)) for name in os.listdir('bot-data/runs')) + 1
        self.experiment_folder = f'bot-data/runs/run-{run_num}'
        os.makedirs(self.experiment_folder)
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
            self.add_bot(bot)

exp = Experiment('bot-data/experiment.cfg')
exp.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    exp.stop()