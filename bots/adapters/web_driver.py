import threading
import os
import queue
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

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