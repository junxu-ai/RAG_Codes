import requests

def get_real_time_stock_price(symbol):
    api_url = f'https://api.example.com/stock/{symbol}/price'
    response = requests.get(api_url)
    if response.status_code == 200:
        price_data = response.json()
        return price_data['price']
    else:
        return None

# Example usage
current_price = get_real_time_stock_price('AAPL')
print(f'Current stock price of AAPL: ${current_price}')
