from dotenv import load_dotenv
from requests import post
from os import environ

load_dotenv()

def send_discord(message):
    post(environ['WEBHOOK_URL'], json={'content': message})