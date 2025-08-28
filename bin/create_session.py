#!python

import requests  # pip3 install requests
from pprint import pprint
from os import environ
import sys
import getopt

GET = requests.get
POST = requests.post

# if using Heroku, change this to https://YOURAPP.herokuapp.com
LOCAL_SERVER_URL = 'http://localhost:8000'
SERVER_URL = 'https://llmmarket-310495633edd.herokuapp.com'
BASE_URL = [SERVER_URL]
REST_KEY = "sug9rD9dd!"

COMP_URL_EXP = 'https://app.prolific.com/submissions/complete?cc=CT8QWJ3A'
COMP_URL_PRE = 'https://app.prolific.com/submissions/complete?cc=C1BNU59M'

def call_api(method, *path_parts, **params) -> dict:
    path_parts = '/'.join(path_parts)
    url = f'{BASE_URL[0]}/api/{path_parts}/'
    resp = method(url, json=params, headers={'otree-rest-key': REST_KEY})
    if not resp.ok:
        msg = (
            f'Request to "{url}" failed '
            f'with status code {resp.status_code}: {resp.text}'
        )
        raise Exception(msg)
    return resp.json()

def check_session(code):
    resp_check = call_api(GET, 'sessions', code)
    del resp_check['participants']
    return resp_check

def make_exp(N, l, dist, is_hi, show_next, comp_url=COMP_URL_EXP):

    participation_fee = 12.00

    session_configs = dict(
        endow_stock = dist,
        participation_fee = participation_fee,
        real_world_currency_per_point = 0.005,
        show_next = show_next,
        market_time = 20,
        forecast_time = 30,
        summary_time = 10,
    )
    
    if is_hi:
        session_configs['real_world_currency_per_point'] = 0.025
        session_configs['is_hi_stakes'] = True


    # Create the experiment session
    resp_exp_create = call_api(POST, 'sessions',
                session_config_name='rounds',
                room_name='market2',
                num_participants=N,
                modified_session_config_fields=session_configs,
                )
    exp_code = resp_exp_create['code']
    room_url = resp_exp_create['room_url']
    resp_exp_check = check_session(exp_code)
    
    #Set the completion URL
    call_api(POST, 'session_vars', exp_code, vars={'prolific_completion_url':comp_url})

    
    # Create the landing session
    session_configs['experiment_link'] = room_url
    resp_land_create = call_api(POST, 'sessions',
                session_config_name='ctlanding',
                room_name='CTlanding',
                num_participants=l,
                modified_session_config_fields=session_configs,
                )
    land_code = resp_land_create['code']
    resp_land_check = check_session(land_code)
    
    # Set the completion URL
    call_api(POST, 'session_vars', land_code, vars={'prolific_completion_url':comp_url})

    return {exp_code: resp_exp_check, land_code: resp_land_check}


def make_screen(N, times, participation_fee=0.50, comp_url=COMP_URL_PRE):
    session_configs = dict(
        participation_fee = participation_fee,
        is_prolific = True,
        slot_01='',
        slot_02='',
        slot_03='',
        slot_04='',
        slot_05='',
        slot_06='',
        slot_07='',
        slot_08='',
        slot_09='',
        
        slot_10='',
    )

    # set the times in the times on the slots
    for idx, t in enumerate(times.split()):
        slot_num = idx+1
        slot = f"slot_{slot_num:0>2}"
        session_configs[slot] = t

    resp_screen_create = call_api(POST, 'sessions',
                session_config_name='prescreen',
                room_name='prescreen',
                num_participants=N,
                modified_session_config_fields=session_configs,
                )
    screen_code = resp_screen_create['code']
    resp_screen_check = check_session(screen_code)
    
    
    # Set the completion URL
    call_api(POST, 'session_vars', screen_code, vars={'prolific_completion_url':comp_url})

    return {screen_code: resp_screen_check}


USAGE = 'create_session.py  -s <exp | screen> -n <num_participants> -l <landing_page_participants> p ' \
        '--dist=<share distribution> '

def main(argv):
    stage = ''               # s:
    dist = '4 4 4'           # dist=
    is_pilot = False         # p
    N = 0                    # n:
    l_num = 0                # l:
    times = ""               # times=
    is_hi = False            # h
    show_next = False        # x

    try:
        opts, args = getopt.getopt(argv, "phxs:n:l:t:", ["dist=", "local", "prolific", "mturk", "times="])
    except getopt.GetoptError as e:
        print("Error parsing options: ", e)
        print (USAGE)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-p':
            is_pilot = True
            
        elif opt == '-h':
            is_hi = True
            
        elif opt == '-x':
            show_next = True

        elif opt == '-s':
            stage = arg

        elif opt == '-n':
            N = int(arg)

        elif opt == '-l':
            l_num = int(arg)

        elif opt == '--dist':
            dist = arg

        elif opt == '--times':
            times = arg
            
        elif opt == '--local':
            BASE_URL[0] = LOCAL_SERVER_URL
            print(BASE_URL)


    print (opts)
    if stage not in ['exp', 'screen']:
        print(USAGE)
        sys.exit(2)

    if stage == 'exp':
        resp = make_exp(N, l_num, dist, is_hi, show_next)
        pprint(resp)
        print("SESSIONS CREATED:")
        keys = list(resp.keys())
        print(f"Landing: {keys[1]}")
        print(f"Experiment: {keys[0]}")

    if stage == 'screen':
        resp = make_screen(N, times)
        pprint(resp)



if __name__ == "__main__":
    main(sys.argv[1:])
