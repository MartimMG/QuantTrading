import oandapyV20
from oandapyV20 import API

# Replace with your own token and account ID from OANDA
OANDA_ACCESS_TOKEN = "your-oanda-api-token"
OANDA_ACCOUNT_ID = "your-oanda-account-id"

# Instantiate the API client
client = API(access_token=OANDA_ACCESS_TOKEN)
