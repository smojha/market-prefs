import threading
import pickle
import random
import json
import time
import math
from adapters.llm import fetch_llm_completion
from adapters.web_driver import WindowManager
from config.constants import PROMPT_STEM, NUM_ROUNDS, NUM_PRACTICE_ROUNDS
from domain.market import Forecast, Round, Insight, ThoughtObservation, Plan, RiskSelection, PortfolioState, MarketState

class TradingAgent:
    def __init__(self, internal_id : int, uniq_url : str, window_manager : WindowManager, logging_folder : str, current_model : str):
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

    def make_risk_choice(self, page_read):
        if not self.current_round.risk_selections:
            self.current_round.risk_selections = []
        
        # Generate blank risk selection based on the options
        safe_card_option_1 = RiskSelection.Lottery(float(page_read['data']['safe_card']['win_odds'][0]) / 100, float(page_read['data']['safe_card']['awards'][0].split()[-1]), float(page_read['data']['safe_card']['win_odds'][1]) / 100, float(page_read['data']['safe_card']['awards'][1].split()[-1]))
        risk_card_option_1 = RiskSelection.Lottery(float(page_read['data']['risk_card']['win_odds'][0]) / 100, float(page_read['data']['risk_card']['awards'][0].split()[-1]), float(page_read['data']['risk_card']['win_odds'][1]) / 100, float(page_read['data']['risk_card']['awards'][1].split()[-1]))
        current_risk_selection = RiskSelection(safe_card_option_1, risk_card_option_1)
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

    def make_forecast(self, page_round):
        # forecasts have already been collected, and so just need to be input
        # execute forecast selection
        forecast_data = []
        for fs in self.current_round.forecast.forecast_selections:
            forecast_data.append({ 'field': fs.field, 'value': fs.input_forecast })
        self.window_manager.run_command(self.internal_id, f"setForecastDataFromJSON({json.dumps(forecast_data)});")

    def generate_market_history(self, history):
        market_history = ""
        for round in history:
            round_id = round['round_num']
            if round_id <= int(NUM_PRACTICE_ROUNDS):
                round_id = "Practice Round " + str(round_id)
            else:
                if round_id == int(NUM_PRACTICE_ROUNDS) + 1:
                    market_history += f"""
        [Start Main Experiment Rounds]
        """
                round_id = "Round " + str(round_id - int(NUM_PRACTICE_ROUNDS))
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
                current_round -= int(NUM_PRACTICE_ROUNDS)
            if int(current_round) + int(gap) > int(NUM_ROUNDS):
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
    
    def process_round_results(self, page_read):
        # update mrkt and ptfl states
        market_price = float(page_read['stats_data']['marketStats']['marketPrice'])
        interest_rate = page_read['stats_data']['marketStats']['interestRate']
        dividends = page_read['stats_data']['marketStats']['dividends']
        buy_back = float(page_read['stats_data']['marketStats']['buyBack'])
        self.current_round.market_state = MarketState(market_price, interest_rate, dividends, buy_back)
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
        self.current_round.portfolio_state = PortfolioState(num_shares, current_cash, stock_value, dividend_earned, interest_earned)
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
                self.current_round.portfolio_state.executed_trades.append(PortfolioState.Order('SELL', num_shares, price))
            elif 'You bought' in msg['msg']:
                num_shares = int(msg['msg'].split()[2])
                price = float(msg['msg'].split()[-1])
                self.current_round.portfolio_state.executed_trades.append(PortfolioState.Order('BUY', num_shares, price))
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
        self.current_round = Round(round_num + 1, None, [], None, None, None, None, None, self.model_used, self.internal_id)

    def is_valid_next_stage(self, page_read):
        """
        Validate and advance the experiment state machine without the risk page.
        """
        nxt_source = {
            'consent_page': ['begin_practice'],
            'begin_practice': ['market_choice_page'],
            'market_choice_page': ['forecast_page'],
            'forecast_page': ['round_results_page'],
            'round_results_page': ['market_choice_page', 'practice_results', 'final_results'],
            'practice_results': ['end_practice'],
            'end_practice': ['market_choice_page'],
            'final_results': ['payment_page']
        }

        # Must contain 'source'
        if 'source' not in page_read:
            return False

        source = page_read['source']

        # Do not accept duplicate same-source events
        if source == getattr(self, 'current_source', None):
            return False

        # Initial entry into the flow
        if source == 'consent_page' and not getattr(self, 'current_source', None):
            self.round_stage = 'consent_page'
            self.current_source = 'consent_page'
            return True

        # Strict stepwise transitions and special cases

        if source == 'begin_practice' and self.current_source == 'consent_page':
            self.round_stage = 'begin_practice'
            self.current_source = 'begin_practice'
            return True

        # Starting a NEW round when we land on market_choice_page
        if source == 'market_choice_page' and self.current_source in ['begin_practice', 'end_practice', 'round_results_page']:
            # new round
            self.generate_new_round(self.round_num)
            self.round_stage = 'market_choice_page'
            self.current_source = 'market_choice_page'
            return True

        # Generic allowed transitions using the table above
        allowed_next = nxt_source.get(self.current_source, [])
        if source and source in allowed_next:
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
            time.sleep(0.2)
    
    def gen_llm_trades(self, page_read):
        # set up forecast stuff
        if not self.current_round.forecast:
            self.current_round.forecast = Forecast()
        forecast_options, forecast_options_string = self.generate_forecast_options()
        for idx, (key, value) in enumerate(forecast_options.items()):
            input_field = f"f{idx}"
            self.current_round.forecast.addForecastSelection(Forecast.ForecastSelection(key, value['lb'], value['ub'], input_field))
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
        round_id = "Practice Round " + str(round_num) if round_num <= int(NUM_PRACTICE_ROUNDS) else "Round " + str(round_num - NUM_PRACTICE_ROUNDS)

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
{nl + "Here is your practice round reflection:" + nl + "Filename: PRACTICE REFLECTION (read-only)" + nl + "+++++++++++++++++++++" + nl + str(self.practice_reflection) + nl + "+++++++++++++++++++++" + nl if round_num > int(NUM_PRACTICE_ROUNDS) else ""}
Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_history}
+++++++++++++++++++++

Here is your current portfolio information:
Filename: CURRENT PORTFOLIO (read-only)
+++++++++++++++++++++
{current_round_info}
+++++++++++++++++++++
{nl + "PRACTICE ROUND HISTORY/REFLECTION SHOULD ONLY BE USED TO LEARN THE EXPERIMENT SETTING AND MAY NOT REFLECT MARKET CONDITIONS IN THE MAIN EXPERIMENT." + nl if round_num > int(NUM_PRACTICE_ROUNDS) else ""}
Here is some key information to consider during your price forecasting:
- Make sure to submit a forecast within the specified range for each forecast input
- Use your previous history access to make informed decisions
- Remember that accurate (within 2.5 units) forecasts will earn you a reward at the end of the experiment

Here is some key information to consider during your order submission:
- You can only sell STOCK that you own and purchase STOCK with CASH you already have
- You are not required to submit orders every round and you may submit multiple orders each round for one or both sides
- Limit prices this round MUST be integer values between {self.current_round.market_state.market_price - int(NUM_PRACTICE_ROUNDS)} and {self.current_round.market_state.market_price + int(NUM_PRACTICE_ROUNDS)}. It is important that they are integer values within this range
- Make use of the provided history and your own strategies to make informed decisions
- Market dynamics can change over time, and so it might be necessary to adapt your strategies as needed
- Depending on market conditions, you may need to be aggressive or conservative in your trading strategies to maximize your earnings

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to maximize your total earnings at the END of the {int(NUM_ROUNDS)} main experiment rounds. You have {(int(NUM_ROUNDS) + int(NUM_PRACTICE_ROUNDS) - self.current_round.round_num) if self.current_round.round_num > int(NUM_PRACTICE_ROUNDS) else (str(int(NUM_PRACTICE_ROUNDS) - self.current_round.round_num) + "practice")} rounds remaining.

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
        self.current_round.plan = Plan(plan)
        self.current_round.insight = Insight(insight)
        self.current_round.observations_and_thoughts = ThoughtObservation(observations_and_thoughts)

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
                trades.append(PortfolioState.Order(o['order_type'], int(o['quantity']), float(o['limit_price'])))
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
            self.current_round.market_state = MarketState(market_price, interest_rate, dividends, buy_back)
        # capture portfolio state information
        if not self.current_round.portfolio_state:
            num_shares = int(page_read['stats_data']['personalStats']['shares'])
            current_cash = float(page_read['stats_data']['personalStats']['cash'])
            stock_value = float(page_read['stats_data']['personalStats']['stockValue'])
            self.current_round.portfolio_state = PortfolioState(num_shares, current_cash, stock_value)
        # TODO: run commands to determine trades
        # START TEMP TRADE LOGIC
        trades = []
        price_shift = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        half_spread = random.choice([1, 2, 3, 4, 5])
        mid_spread = self.current_round.market_state.market_price + price_shift
        # avoid negative prices
        mid_spread = max(half_spread, mid_spread)

        trades.append(PortfolioState.Order('BUY', 1, mid_spread - half_spread))
        trades.append(PortfolioState.Order('SELL', 1, mid_spread + half_spread))
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
        round_id = "Practice Round " + str(round_num) if round_num <= int(NUM_PRACTICE_ROUNDS) else "Round " + str(round_num - int(NUM_PRACTICE_ROUNDS))
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

Now you have all the necessary information to complete the task. Remember YOUR TOP PRIORITY is to maximize your total earnings (STOCK value @ buyback price + CASH earnings + forecast winnings + lottery winnings) at the END of the {int(NUM_ROUNDS)} experiment rounds.
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
        round_id = "Practice Round " + str(round_num) if round_num <= int(NUM_PRACTICE_ROUNDS) else "Round " + str(round_num - int(NUM_PRACTICE_ROUNDS))
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
        # parse response
        resp_obj = fetch_llm_completion(prompt, self.model_used)
        final_reflection = resp_obj['new_content']['final_reflection']
        self.final_reflection = final_reflection
        self.write_final_reflection_to_file()