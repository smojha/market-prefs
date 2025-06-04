# open a specified run directory and loads in all of the json files, collect all of the "insight" entries (still tagged with the associated bot id) and condense them into a single json file
# The output file will be named "insight_condensed.json"

import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Condense insight files')
    parser.add_argument('run_dir', type=str, help='The run directory to process')
    args = parser.parse_args()

    desired_info = 'insight'

    # print example of how to use the script
    print('Example: python3 insight_condenser.py /path/to/run/directory')

    run_dir = args.run_dir
    # get all the json files in the run directory
    json_files = [f for f in os.listdir(run_dir) if f.endswith('.json')]
    # only keep files of the form bot-<bot_id>-history.json (make sure they end with -history.json)
    json_files = [f for f in json_files if f.startswith('bot-') and f.endswith('-history.json')]

    print(json_files)
    # every filename is of form bot-<bot_id>-history.json. get the bot_ids from the filenames and store them to associate them with the insights
    bot_ids = [int(f.split('-')[1]) for f in json_files if f.startswith('bot-')]

    insights = []
    # open each of the files, iterate through the json array and collect the 'insight' entry in each object (might be nested) while keeping track of round number (index in json array + 1) and bot id
    for i, f in enumerate(json_files):
        # print out current json file being processed
        with open(os.path.join(run_dir, f), 'r') as file:
            data = json.load(file)
            for obj in data:
                if desired_info in obj:
                    insights.append({
                        'round': i + 1,
                        'bot_id': bot_ids[i],
                        desired_info: obj[desired_info]
                    })

    with open(os.path.join(run_dir, str(desired_info) + '_condensed.json'), 'w') as f:
        json.dump(insights, f, indent=4)

if __name__ == '__main__':
    main()

