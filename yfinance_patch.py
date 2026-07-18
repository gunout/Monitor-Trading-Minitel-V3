# yfinance_patch.py
import yfinance as yf
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

def get_yahoo_data_direct(symbol, period='1d', interval='1m'):
    """Récupère les données directement via l'API Yahoo Finance"""
    try:
        # Mapping des périodes
        period_map = {
            '1d': '1d', '5d': '5d', '1mo': '1mo', 
            '3mo': '3mo', '6mo': '6mo', '1y': '1y'
        }
        
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '60m', '1d': '1d'
        }
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': interval_map.get(interval, '1m'),
            'range': period_map.get(period, '1d'),
            'includePrePost': 'false'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart']:
            return None
            
        result = data['chart']['result'][0]
        if result is None:
            return None
            
        timestamps = result.get('timestamp', [])
        quote = result.get('indicators', {}).get('quote', [{}])[0]
        
        if not timestamps or not quote:
            return None
            
        # Construire le DataFrame
        df = pd.DataFrame({
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        })
        
        # Ajouter l'index des dates
        df.index = pd.to_datetime(timestamps, unit='s')
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Erreur API directe: {e}")
        return None

# Patcher yfinance pour utiliser l'API directe
def patch_yfinance():
    """Applique le patch pour utiliser l'API directe"""
    
    original_history = yf.Ticker.history
    
    def patched_history(self, period='1d', interval='1m', *args, **kwargs):
        """Méthode patchée avec API directe"""
        symbol = self.ticker
        
        # Essayer d'abord l'API directe
        try:
            df = get_yahoo_data_direct(symbol, period, interval)
            if df is not None and not df.empty:
                print(f"✅ API directe OK pour {symbol}")
                return df
        except:
            pass
        
        # Fallback: méthode originale
        try:
            result = original_history(self, period=period, interval=interval, *args, **kwargs)
            if not result.empty:
                return result
        except:
            pass
        
        # Si tout échoue, retourner un DataFrame vide
        return pd.DataFrame()
    
    yf.Ticker.history = patched_history
    print("✅ Patch yfinance appliqué - API directe activée")
    return True

# Appliquer le patch
patch_yfinance()