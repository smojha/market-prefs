from otree.api import *
from enum import Enum

from otree.common import InvalidRoundError

import common.SessionConfigFunctions as scf
from rounds.models import NO_AUTO_TRANS


class OrderType(Enum):
    """
        Enumeration representing the order type BID/OFFER
    """
    BID = -1
    OFFER = 1


class Subsession(BaseSubsession):
    pass


# noinspection DuplicatedCode
class Group(BaseGroup):
    price = models.CurrencyField()
    volume = models.IntegerField()
    dividend = models.IntegerField(initial=0)

    float = models.IntegerField()
    short = models.IntegerField()

    def in_round_or_none(self, round_number):
        try:
            return self.in_round(round_number)
        except InvalidRoundError:
            return None

    def get_last_period_price(self):
        # Get the market Price of the last period
        round_number = self.round_number
        last_group = self.in_round_or_none(round_number - 1)

        if last_group:
            return round(last_group.price)
        else:
            init_price = scf.get_init_price(self)
            if init_price:
                return init_price
            else:
                return scf.get_fundamental_value(self)


# noinspection DuplicatedCode
class Player(BasePlayer):
    cash = models.CurrencyField()
    shares = models.IntegerField()

    #  These items just exist to drive the form on the page.
    # The actual price and quantity are stored in the Order model
    type = models.IntegerField(
        choices=[
            [-1, 'Buy'],
            [1, 'Sell'],
        ],
        blank=True
    )
    price = models.IntegerField(blank=True)
    quantity = models.IntegerField(blank=True)

    periods_until_auto_buy = models.IntegerField(initial=NO_AUTO_TRANS)
    periods_until_auto_sell = models.IntegerField(initial=NO_AUTO_TRANS)

    # Market Movement
    shares_transacted = models.IntegerField(initial=0)
    trans_cost = models.IntegerField(initial=0)
    cash_after_trade = models.IntegerField()
    interest_earned = models.IntegerField()
    dividend_earned = models.IntegerField()

    # Results
    cash_result = models.IntegerField()
    shares_result = models.IntegerField()

    # Forecasting Item
    f0 = models.CurrencyField()
    f1 = models.CurrencyField()
    f2 = models.CurrencyField()
    forecast_error = models.CurrencyField()
    forecast_reward = models.CurrencyField(initial=0)

    def to_dict(self):
        d = {'cash': self.field_maybe_none('cash'),
             'shares': self.field_maybe_none('shares'),
             'periods_until_auto_buy': self.field_maybe_none('periods_until_auto_buy'),
             'periods_until_auto_sell': self.field_maybe_none('periods_until_auto_sell'),
             'shares_transacted': self.field_maybe_none('shares_transacted'),
             'trans_cost': self.field_maybe_none('trans_cost'),
             'cash_after_trade': self.field_maybe_none('cash_after_trade'),
             'interest_earned': self.field_maybe_none('interest_earned'),
             'dividend_earned': self.field_maybe_none('dividend_earned'),
             'cash_result': self.field_maybe_none('cash_result'),
             'shares_result': self.field_maybe_none('shares_result')}
        return d

    def update_from_dict(self, d):
        self.shares_result = d.get('shares_result')
        self.shares_transacted = d.get('shares_transacted')
        self.trans_cost = d.get('trans_cost')
        self.cash_after_trade = d.get('cash_after_trade')
        self.dividend_earned = d.get('dividend_earned')
        self.interest_earned = d.get('interest_earned')
        self.cash_result = d.get('cash_result')
        self.periods_until_auto_buy = d.get('periods_until_auto_buy')
        self.periods_until_auto_sell = d.get('periods_until_auto_sell')

    def get_personal_stock_margin(self, price):
        # ABS te short value to make the formula work
        stock_pos_value = abs(float(self.shares * price))
        if stock_pos_value == 0:
            personal_stock_margin = 0
        else:
            cash_float = float(self.cash)
            personal_stock_margin = abs((cash_float - stock_pos_value) / stock_pos_value)
        return personal_stock_margin

    def get_personal_cash_margin(self, price):
        stock_pos_value = abs(self.shares * price)
        if self.cash == 0:
            personal_cash_margin = 0
        else:
            cash_float = abs(float(self.cash))
            personal_cash_margin = abs((stock_pos_value - cash_float) / cash_float)
        return personal_cash_margin

    def is_short(self):
        return self.shares < 0

    def is_debt(self):
        return self.cash < 0

    def is_bankrupt(self):
        return self.shares < 0 and self.cash < 0

    def in_round_or_null(self, round_number):
        try:
            return self.in_round(round_number)
        except InvalidRoundError:
            return None

    def is_short_margin_violation(self):
        if self.is_bankrupt():
            return False

        price = self.group.get_last_period_price()
        margin_ratio = scf.get_margin_ratio(self)
        return self.is_short() and self.get_personal_stock_margin(price) <= margin_ratio

    def is_debt_margin_violation(self):
        if self.is_bankrupt():
            return False

        price = self.group.get_last_period_price()
        margin_ratio = scf.get_margin_ratio(self)
        return self.is_debt() and self.get_personal_cash_margin(price) <= margin_ratio

    def copy_results_from_previous_round(self):
        r_num = self.round_number
        past_player = self.in_round_or_null(r_num - 1)
        if past_player:
            self.cash = past_player.cash_result
            self.shares = past_player.shares_result

    def determine_forecast_reward(self, price):
        f0 = self.field_maybe_none('f0')
        if f0 is not None:
            forecast_error = abs(price - self.f0)
            reward = scf.get_forecast_reward(self)
            threshold = scf.get_forecast_thold(self)
            forecast_reward = reward if forecast_error <= threshold else 0
            self.forecast_error = forecast_error
            self.forecast_reward = forecast_reward
            self.cash_result += forecast_reward

    @staticmethod
    def calculate_delay(current_delay, base):
        if current_delay == NO_AUTO_TRANS:
            return base
        return max(current_delay - 1, 0)

    def determine_auto_trans_status(self):
        prev_player = self.in_round_or_null(self.round_number - 1)
        auto_trans_delay = scf.get_auto_trans_delay(self)

        if not prev_player:
            buy_delay = auto_trans_delay
            sell_delay = auto_trans_delay
        else:
            prev_delay = prev_player.field_maybe_none('periods_until_auto_buy')
            buy_delay = self.calculate_delay(prev_delay, auto_trans_delay)
            prev_delay = prev_player.field_maybe_none('periods_until_auto_sell')
            sell_delay = self.calculate_delay(prev_delay, auto_trans_delay)

        short_mv = self.is_short_margin_violation()
        debt_mv = self.is_debt_margin_violation()

        # Skip out for bankrupt players
        if self.is_bankrupt():
            self.periods_until_auto_buy = NO_AUTO_TRANS
            self.periods_until_auto_sell = NO_AUTO_TRANS
            return

        # Short buy-in status / delay
        if short_mv:
            self.periods_until_auto_buy = buy_delay
        else:
            self.periods_until_auto_buy = NO_AUTO_TRANS

        # debt buy-in status / delay
        if debt_mv:
            self.periods_until_auto_sell = sell_delay
        else:
            self.periods_until_auto_sell = NO_AUTO_TRANS

    def is_auto_buy(self):
        return self.periods_until_auto_buy == 0

    def is_auto_sell(self):
        return self.periods_until_auto_sell == 0


class Order(ExtraModel):
    player = models.Link(Player)
    group = models.Link(Group)
    # uuid = models.StringField()
    order_type = models.IntegerField()
    price = models.CurrencyField()
    quantity = models.IntegerField()
    quantity_final = models.IntegerField(initial=0)
    original_quantity = models.IntegerField()
    is_buy_in = models.BooleanField(initial=False)  # is this order an automatic buy-in?

    def to_dict(self):
        return dict(
            oid=self.id,
            p_id=self.player.id_in_group,
            group_id=self.group.id,
            type=self.order_type,
            price=self.price,
            quantity=self.quantity,
            original_quantity=self.original_quantity)