import datetime
from typing import List
from bs4 import BeautifulSoup
import requests

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
    def __init__(self, round_num : int, forecast : 'Forecast', risk_selections : List['RiskSelection'], portfolio : 'PortfolioState', market : 'MarketState', plan : 'Plan', insight : 'Insight', thought_observation : 'ThoughtObservation', model_used : str, agent_id : int):
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