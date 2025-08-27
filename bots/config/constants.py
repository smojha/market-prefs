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

MIXED_BOT_TYPES = "mixed"

# Mixed bot types dict (kludge solution to assigning bot model types)
MIXED_BOT_TYPES_DICT = {
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

data_folder = "bot-data/debug-runs"
run_comments = "No comments provided"
experiment_link = "No link provided" # default to no link provided

# RunConfig(url='http://localhost:8000/SessionStartLinks/jo2g7edd', model_name='gpt-4o', production=True, message='gpt-4o experienced run 2/3', timeout_minutes=105, data_folder='bot-data/debug-runs', prod_folder='bot-data/production-runs')