import os
import requests

# Let's try to load using python-dotenv in case 'uv' didn't parse it well or shell evaluation failed
from dotenv import load_dotenv

load_dotenv()

cdse_username = os.environ.get("CDSE_USERNAME")
cdse_password = os.environ.get("CDSE_PASSWORD")
print(f"User starts with: {cdse_username[:3] if cdse_username else None}")

token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
data = {
    "client_id": "cdse-public",
    "username": cdse_username,
    "password": cdse_password,
    "grant_type": "password",
}
resp = requests.post(token_url, data=data)
try:
    print("Error:", resp.json().get("error_description"))
except Exception:
    pass
