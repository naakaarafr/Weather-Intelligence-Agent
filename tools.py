import requests
import json
import time
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os
from dataclasses import dataclass
import random

# Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"
GEO_URL = "https://api.openweathermap.org/geo/1.0"

@dataclass
class WeatherData:
    """Structured weather data container"""
    location: str
    temperature: float
    humidity: int
    pressure: float
    wind_speed: float
    wind_direction: int
    description: str
    timestamp: datetime
    visibility: float = 0
    uv_index: float = 0
    precipitation: float = 0

class WeatherTools:
    """Comprehensive weather tools for AI agents"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.session = requests.Session()
        self.retry_attempts = 3
        self.retry_delay = 1
        
        # Historical data cache (in-memory)
        self.weather_cache = {}
        self.anomaly_thresholds = {
            'temperature': {'min': -50, 'max': 60},
            'humidity': {'min': 0, 'max': 100},
            'pressure': {'min': 800, 'max': 1200},
            'wind_speed': {'min': 0, 'max': 200}
        }
    
    # =================== DATA INGESTION TOOLS ===================
    
    def fetch_weather_data(self, location: str, units: str = "metric") -> Dict[str, Any]:
        """Fetch current weather data with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                url = f"{BASE_URL}/weather"
                params = {
                    'q': location,
                    'appid': self.api_key,
                    'units': units
                }
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Transform to structured format
                weather_data = WeatherData(
                    location=f"{data['name']}, {data['sys']['country']}",
                    temperature=data['main']['temp'],
                    humidity=data['main']['humidity'],
                    pressure=data['main']['pressure'],
                    wind_speed=data['wind'].get('speed', 0),
                    wind_direction=data['wind'].get('deg', 0),
                    description=data['weather'][0]['description'],
                    timestamp=datetime.now(),
                    visibility=data.get('visibility', 0) / 1000,  # Convert to km
                    precipitation=data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0)
                )
                
                # Cache the data
                self.weather_cache[location] = {
                    'data': weather_data,
                    'timestamp': datetime.now(),
                    'raw': data
                }
                
                return {
                    'status': 'success',
                    'data': weather_data.__dict__,
                    'raw_data': data
                }
                
            except requests.exceptions.RequestException as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return {'status': 'error', 'message': f"Failed to fetch data: {str(e)}"}
        
        return {'status': 'error', 'message': "Max retry attempts exceeded"}
    
    def fetch_forecast_data(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Fetch weather forecast data"""
        try:
            url = f"{BASE_URL}/forecast"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            forecasts = []
            for item in data['list']:
                forecast = {
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind'].get('speed', 0),
                    'description': item['weather'][0]['description'],
                    'precipitation_probability': item.get('pop', 0) * 100
                }
                forecasts.append(forecast)
            
            return {
                'status': 'success',
                'location': f"{data['city']['name']}, {data['city']['country']}",
                'forecasts': forecasts
            }
            
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f"Failed to fetch forecast: {str(e)}"}
    
    def validate_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize weather data"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_data': data.copy()
        }
        
        if 'data' not in data:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No weather data found")
            return validation_results
        
        weather_data = data['data']
        
        # Temperature validation
        if 'temperature' in weather_data:
            temp = weather_data['temperature']
            if temp < -100 or temp > 70:
                validation_results['warnings'].append(f"Extreme temperature detected: {temp}°C")
            if temp < -273.15:  # Below absolute zero
                validation_results['errors'].append("Temperature below absolute zero")
                validation_results['is_valid'] = False
        
        # Humidity validation
        if 'humidity' in weather_data:
            humidity = weather_data['humidity']
            if humidity < 0 or humidity > 100:
                validation_results['errors'].append(f"Invalid humidity: {humidity}%")
                validation_results['sanitized_data']['data']['humidity'] = max(0, min(100, humidity))
        
        # Pressure validation
        if 'pressure' in weather_data:
            pressure = weather_data['pressure']
            if pressure < 800 or pressure > 1200:
                validation_results['warnings'].append(f"Extreme pressure detected: {pressure} hPa")
        
        # Wind speed validation
        if 'wind_speed' in weather_data:
            wind_speed = weather_data['wind_speed']
            if wind_speed < 0:
                validation_results['errors'].append("Negative wind speed")
                validation_results['sanitized_data']['data']['wind_speed'] = 0
            elif wind_speed > 150:
                validation_results['warnings'].append(f"Extreme wind speed: {wind_speed} m/s")
        
        return validation_results
    
    # =================== ANOMALY DETECTION TOOLS ===================
    
    def detect_anomalies(self, location: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect statistical outliers in weather patterns"""
        anomalies = {
            'detected': False,
            'anomalies': [],
            'severity': 'normal'
        }
        
        if 'data' not in current_data:
            return anomalies
        
        weather_data = current_data['data']
        
        # Check against absolute thresholds
        for param, thresholds in self.anomaly_thresholds.items():
            if param in weather_data:
                value = weather_data[param]
                if value < thresholds['min'] or value > thresholds['max']:
                    anomalies['detected'] = True
                    anomalies['anomalies'].append({
                        'parameter': param,
                        'value': value,
                        'expected_range': thresholds,
                        'type': 'absolute_threshold'
                    })
        
        # Check for rapid changes (if historical data available)
        if location in self.weather_cache:
            cached_data = self.weather_cache[location]['data']
            time_diff = (datetime.now() - self.weather_cache[location]['timestamp']).total_seconds() / 3600
            
            if time_diff < 24:  # Within 24 hours
                temp_change = abs(weather_data['temperature'] - cached_data.temperature)
                pressure_change = abs(weather_data['pressure'] - cached_data.pressure)
                
                if temp_change > 15:  # >15°C change
                    anomalies['detected'] = True
                    anomalies['anomalies'].append({
                        'parameter': 'temperature_change',
                        'value': temp_change,
                        'type': 'rapid_change'
                    })
                
                if pressure_change > 20:  # >20 hPa change
                    anomalies['detected'] = True
                    anomalies['anomalies'].append({
                        'parameter': 'pressure_change',
                        'value': pressure_change,
                        'type': 'rapid_change'
                    })
        
        # Determine severity
        if len(anomalies['anomalies']) >= 3:
            anomalies['severity'] = 'critical'
        elif len(anomalies['anomalies']) >= 2:
            anomalies['severity'] = 'high'
        elif len(anomalies['anomalies']) == 1:
            anomalies['severity'] = 'moderate'
        
        return anomalies
    
    def analyze_threats(self, weather_data: Dict[str, Any], forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emerging weather threats"""
        threats = {
            'level': 'low',
            'threats': [],
            'recommendations': []
        }
        
        if 'data' not in weather_data:
            return threats
        
        current = weather_data['data']
        
        # Analyze current conditions
        if current.get('wind_speed', 0) > 25:  # >25 m/s (90 km/h)
            threats['threats'].append({
                'type': 'high_winds',
                'severity': 'high',
                'description': f"Dangerous winds: {current['wind_speed']} m/s"
            })
        
        if current.get('temperature', 0) > 40:
            threats['threats'].append({
                'type': 'extreme_heat',
                'severity': 'high',
                'description': f"Extreme heat: {current['temperature']}°C"
            })
        
        if current.get('temperature', 0) < -20:
            threats['threats'].append({
                'type': 'extreme_cold',
                'severity': 'high',
                'description': f"Extreme cold: {current['temperature']}°C"
            })
        
        # Analyze forecast trends
        if forecast_data.get('status') == 'success':
            forecasts = forecast_data['forecasts'][:8]  # Next 24 hours
            
            temps = [f['temperature'] for f in forecasts]
            if max(temps) - min(temps) > 20:
                threats['threats'].append({
                    'type': 'temperature_swing',
                    'severity': 'moderate',
                    'description': f"Large temperature swing expected: {min(temps):.1f}°C to {max(temps):.1f}°C"
                })
            
            precip_prob = max([f['precipitation_probability'] for f in forecasts])
            if precip_prob > 80:
                threats['threats'].append({
                    'type': 'heavy_precipitation',
                    'severity': 'moderate',
                    'description': f"High precipitation probability: {precip_prob}%"
                })
        
        # Determine overall threat level
        high_threats = sum(1 for t in threats['threats'] if t['severity'] == 'high')
        moderate_threats = sum(1 for t in threats['threats'] if t['severity'] == 'moderate')
        
        if high_threats >= 2:
            threats['level'] = 'critical'
        elif high_threats >= 1:
            threats['level'] = 'high'
        elif moderate_threats >= 2:
            threats['level'] = 'moderate'
        
        return threats
    
    # =================== FORECAST OPTIMIZATION TOOLS ===================
    
    def model_terrain_effects(self, location: str, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Model terrain effects on weather patterns"""
        # Simplified terrain modeling (in real app, you'd use elevation APIs)
        terrain_effects = {
            'location': location,
            'adjustments': {},
            'confidence': 0.7
        }
        
        if 'data' not in weather_data:
            return terrain_effects
        
        current = weather_data['data']
        
        # Simulate terrain-based adjustments
        # In a real implementation, you'd use actual elevation data
        location_lower = location.lower()
        
        # Mountain areas - temperature drops with altitude
        if any(term in location_lower for term in ['mountain', 'hill', 'peak', 'alps', 'rocky']):
            terrain_effects['adjustments']['temperature'] = current.get('temperature', 0) - 2
            terrain_effects['adjustments']['wind_speed'] = current.get('wind_speed', 0) * 1.3
            terrain_effects['adjustments']['pressure'] = current.get('pressure', 1013) - 20
        
        # Coastal areas - moderated temperatures
        elif any(term in location_lower for term in ['beach', 'coast', 'bay', 'port', 'marina']):
            terrain_effects['adjustments']['temperature'] = current.get('temperature', 0) * 0.95
            terrain_effects['adjustments']['humidity'] = min(100, current.get('humidity', 50) + 10)
        
        # Valley areas - temperature inversions possible
        elif any(term in location_lower for term in ['valley', 'basin', 'hollow']):
            terrain_effects['adjustments']['temperature'] = current.get('temperature', 0) - 1
            terrain_effects['adjustments']['wind_speed'] = current.get('wind_speed', 0) * 0.7
        
        # Urban areas - heat island effect
        elif any(term in location_lower for term in ['city', 'downtown', 'urban', 'metro']):
            terrain_effects['adjustments']['temperature'] = current.get('temperature', 0) + 2
            terrain_effects['adjustments']['humidity'] = max(0, current.get('humidity', 50) - 5)
        
        return terrain_effects
    
    def apply_fractal_correction(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fractal-based error correction to forecasts"""
        if forecast_data.get('status') != 'success':
            return forecast_data
        
        corrected_forecast = forecast_data.copy()
        forecasts = corrected_forecast['forecasts']
        
        # Apply fractal noise for micro-variations
        for i, forecast in enumerate(forecasts):
            # Simple fractal correction using sine waves
            time_factor = i * 0.1
            
            # Temperature correction
            temp_noise = math.sin(time_factor * 2) * 0.5 + math.sin(time_factor * 5) * 0.2
            forecast['temperature'] += temp_noise
            
            # Humidity correction
            humidity_noise = math.cos(time_factor * 3) * 2
            forecast['humidity'] = max(0, min(100, forecast['humidity'] + humidity_noise))
            
            # Pressure correction
            pressure_noise = math.sin(time_factor * 1.5) * 3
            forecast['pressure'] += pressure_noise
        
        corrected_forecast['correction_applied'] = True
        corrected_forecast['correction_type'] = 'fractal_noise'
        
        return corrected_forecast
    
    # =================== DECISION ENGINE TOOLS ===================
    
    def generate_recommendations(self, weather_data: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate personalized weather recommendations"""
        if user_preferences is None:
            user_preferences = {}
        
        recommendations = {
            'general': [],
            'clothing': [],
            'activities': [],
            'travel': [],
            'safety': []
        }
        
        if 'data' not in weather_data:
            return recommendations
        
        current = weather_data['data']
        temp = current.get('temperature', 20)
        humidity = current.get('humidity', 50)
        wind_speed = current.get('wind_speed', 0)
        precipitation = current.get('precipitation', 0)
        
        # Temperature-based recommendations
        if temp < 0:
            recommendations['clothing'].append("Wear heavy winter clothing, gloves, and hat")
            recommendations['safety'].append("Be cautious of icy conditions")
            recommendations['activities'].append("Indoor activities recommended")
        elif temp < 10:
            recommendations['clothing'].append("Wear warm jacket and layers")
            recommendations['activities'].append("Good for winter sports")
        elif temp < 20:
            recommendations['clothing'].append("Light jacket or sweater recommended")
            recommendations['activities'].append("Perfect for hiking and outdoor walks")
        elif temp < 30:
            recommendations['clothing'].append("Light, comfortable clothing")
            recommendations['activities'].append("Great weather for all outdoor activities")
        else:
            recommendations['clothing'].append("Light, breathable clothing and sun protection")
            recommendations['safety'].append("Stay hydrated and seek shade regularly")
            recommendations['activities'].append("Early morning or evening activities recommended")
        
        # Wind-based recommendations
        if wind_speed > 15:
            recommendations['travel'].append("Be cautious when driving, especially high-profile vehicles")
            recommendations['safety'].append("Secure loose objects outdoors")
        
        # Precipitation recommendations
        if precipitation > 0:
            recommendations['general'].append("Carry an umbrella or raincoat")
            recommendations['travel'].append("Allow extra travel time due to wet conditions")
        
        # Humidity recommendations
        if humidity > 80:
            recommendations['general'].append("High humidity - stay hydrated")
        elif humidity < 30:
            recommendations['general'].append("Low humidity - consider using moisturizer")
        
        return recommendations
    
    def simulate_scenarios(self, base_conditions: Dict[str, Any], scenario_types: List[str]) -> Dict[str, Any]:
        """Run what-if storm simulations"""
        scenarios = {}
        
        if 'data' not in base_conditions:
            return scenarios
        
        base = base_conditions['data']
        
        for scenario_type in scenario_types:
            if scenario_type == 'storm_intensification':
                scenarios[scenario_type] = {
                    'temperature': base.get('temperature', 20) - 5,
                    'wind_speed': base.get('wind_speed', 0) * 2.5,
                    'pressure': base.get('pressure', 1013) - 30,
                    'precipitation': 25,
                    'impact_level': 'high',
                    'duration_hours': 6
                }
            
            elif scenario_type == 'heat_wave':
                scenarios[scenario_type] = {
                    'temperature': base.get('temperature', 20) + 15,
                    'humidity': max(20, base.get('humidity', 50) - 20),
                    'uv_index': 11,
                    'impact_level': 'high',
                    'duration_hours': 72
                }
            
            elif scenario_type == 'cold_snap':
                scenarios[scenario_type] = {
                    'temperature': base.get('temperature', 20) - 20,
                    'wind_speed': base.get('wind_speed', 0) + 10,
                    'wind_chill': base.get('temperature', 20) - 30,
                    'impact_level': 'moderate',
                    'duration_hours': 48
                }
        
        return scenarios
    
    # =================== TREND ANALYSIS TOOLS ===================
    
    def detect_patterns(self, location: str, days_back: int = 30) -> Dict[str, Any]:
        """Detect long-term climate patterns and trends"""
        # Simulated pattern detection (in real app, you'd use historical data)
        patterns = {
            'location': location,
            'analysis_period': f"{days_back} days",
            'trends': {},
            'seasonal_patterns': {},
            'confidence': 0.6
        }
        
        # Simulate temperature trends
        patterns['trends']['temperature'] = {
            'direction': random.choice(['warming', 'cooling', 'stable']),
            'rate_per_day': round(random.uniform(-0.1, 0.1), 3),
            'confidence': random.uniform(0.5, 0.9)
        }
        
        # Simulate precipitation patterns
        patterns['trends']['precipitation'] = {
            'direction': random.choice(['increasing', 'decreasing', 'stable']),
            'rate_per_day': round(random.uniform(-0.05, 0.05), 4),
            'confidence': random.uniform(0.4, 0.8)
        }
        
        # Seasonal patterns
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:
            patterns['seasonal_patterns']['expected'] = 'winter_conditions'
        elif current_month in [3, 4, 5]:
            patterns['seasonal_patterns']['expected'] = 'spring_transition'
        elif current_month in [6, 7, 8]:
            patterns['seasonal_patterns']['expected'] = 'summer_peak'
        else:
            patterns['seasonal_patterns']['expected'] = 'autumn_cooling'
        
        return patterns
    
    def compress_trends(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress long-term trends into actionable insights"""
        insights = {
            'key_insights': [],
            'weekly_outlook': {},
            'monthly_outlook': {},
            'actionable_items': []
        }
        
        if 'trends' not in trend_data:
            return insights
        
        trends = trend_data['trends']
        
        # Temperature insights
        if 'temperature' in trends:
            temp_trend = trends['temperature']
            if temp_trend['direction'] == 'warming' and temp_trend['confidence'] > 0.7:
                insights['key_insights'].append("Warming trend detected with high confidence")
                insights['actionable_items'].append("Consider adjusting clothing choices for warmer weather")
            elif temp_trend['direction'] == 'cooling' and temp_trend['confidence'] > 0.7:
                insights['key_insights'].append("Cooling trend detected with high confidence")
                insights['actionable_items'].append("Prepare for cooler weather conditions")
        
        # Precipitation insights
        if 'precipitation' in trends:
            precip_trend = trends['precipitation']
            if precip_trend['direction'] == 'increasing':
                insights['key_insights'].append("Increasing precipitation pattern")
                insights['actionable_items'].append("Keep rain gear readily available")
        
        # Weekly outlook
        insights['weekly_outlook'] = {
            'temperature_trend': trends.get('temperature', {}).get('direction', 'stable'),
            'precipitation_trend': trends.get('precipitation', {}).get('direction', 'stable'),
            'confidence': round(sum([trends.get('temperature', {}).get('confidence', 0.5),
                                   trends.get('precipitation', {}).get('confidence', 0.5)]) / 2, 2)
        }
        
        return insights
    
    # =================== VISUALIZATION TOOLS ===================
    
    def generate_chart_data(self, weather_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Generate data structures for weather visualizations"""
        chart_data = {
            'type': chart_type,
            'data': {},
            'config': {}
        }
        
        if chart_type == 'temperature_trend':
            # Generate sample temperature data for the last 24 hours
            now = datetime.now()
            hours = [(now - timedelta(hours=i)).strftime('%H:%M') for i in range(24, 0, -1)]
            
            base_temp = weather_data.get('data', {}).get('temperature', 20)
            temperatures = [base_temp + random.uniform(-3, 3) for _ in range(24)]
            
            chart_data['data'] = {
                'labels': hours,
                'datasets': [{
                    'label': 'Temperature (°C)',
                    'data': temperatures,
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)'
                }]
            }
            
        elif chart_type == 'weather_radar':
            # Generate radar chart data for weather parameters
            current = weather_data.get('data', {})
            
            chart_data['data'] = {
                'labels': ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Visibility'],
                'datasets': [{
                    'label': 'Current Conditions',
                    'data': [
                        min(100, max(0, (current.get('temperature', 0) + 50) / 100 * 100)),
                        current.get('humidity', 50),
                        min(100, max(0, (current.get('pressure', 1013) - 900) / 300 * 100)),
                        min(100, current.get('wind_speed', 0) * 5),
                        min(100, current.get('visibility', 10) * 10)
                    ],
                    'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                    'borderColor': 'rgb(54, 162, 235)'
                }]
            }
        
        return chart_data
    
    def build_visual_narrative(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build visual narratives from weather data"""
        narrative = {
            'title': '',
            'story_elements': [],
            'color_scheme': {},
            'mood': ''
        }
        
        if 'data' not in weather_data:
            return narrative
        
        current = weather_data['data']
        temp = current.get('temperature', 20)
        description = current.get('description', 'clear sky')
        
        # Determine mood and color scheme
        if 'storm' in description or 'thunder' in description:
            narrative['mood'] = 'dramatic'
            narrative['color_scheme'] = {'primary': '#2c3e50', 'secondary': '#e74c3c'}
            narrative['title'] = 'Storm Watch'
        elif 'clear' in description or 'sunny' in description:
            narrative['mood'] = 'bright'
            narrative['color_scheme'] = {'primary': '#f39c12', 'secondary': '#3498db'}
            narrative['title'] = 'Clear Skies Ahead'
        elif temp < 0:
            narrative['mood'] = 'cold'
            narrative['color_scheme'] = {'primary': '#3498db', 'secondary': '#ecf0f1'}
            narrative['title'] = 'Winter Conditions'
        else:
            narrative['mood'] = 'mild'
            narrative['color_scheme'] = {'primary': '#27ae60', 'secondary': '#2ecc71'}
            narrative['title'] = 'Pleasant Weather'
        
        # Story elements
        narrative['story_elements'] = [
            f"Current temperature: {temp}°C",
            f"Conditions: {description}",
            f"Humidity at {current.get('humidity', 50)}%",
            f"Wind speed: {current.get('wind_speed', 0)} m/s"
        ]
        
        return narrative
    
    # =================== ALERT MANAGEMENT TOOLS ===================
    
    def prioritize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prioritize weather alerts based on severity and impact"""
        prioritized = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'total_count': len(alerts)
        }
        
        for alert in alerts:
            severity = alert.get('severity', 'low').lower()
            impact = alert.get('impact', 'local')
            urgency = alert.get('urgency', 'future')
            
            # Calculate priority score
            priority_score = 0
            
            if severity == 'critical':
                priority_score += 10
            elif severity == 'high':
                priority_score += 7
            elif severity == 'moderate':
                priority_score += 4
            
            if impact == 'widespread':
                priority_score += 5
            elif impact == 'regional':
                priority_score += 3
            
            if urgency == 'immediate':
                priority_score += 8
            elif urgency == 'expected':
                priority_score += 5
            
            alert['priority_score'] = priority_score
            
            # Categorize by priority
            if priority_score >= 15:
                prioritized['critical'].append(alert)
            elif priority_score >= 10:
                prioritized['high'].append(alert)
            elif priority_score >= 5:
                prioritized['medium'].append(alert)
            else:
                prioritized['low'].append(alert)
        
        # Sort within each category by score
        for category in prioritized:
            if isinstance(prioritized[category], list):
                prioritized[category].sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return prioritized
    
    def manage_notifications(self, alerts: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage notification streams during extreme events"""
        if user_preferences is None:
            user_preferences = {'max_notifications_per_hour': 5, 'severity_threshold': 'medium'}
        
        notifications = {
            'immediate': [],
            'scheduled': [],
            'suppressed': [],
            'rate_limited': False
        }
        
        current_hour_notifications = 0
        max_per_hour = user_preferences.get('max_notifications_per_hour', 5)
        threshold = user_preferences.get('severity_threshold', 'medium')
        
        # Process alerts by priority
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity not in alerts:
                continue
                
            for alert in alerts[severity]:
                if current_hour_notifications >= max_per_hour:
                    notifications['rate_limited'] = True
                    notifications['suppressed'].append(alert)
                    continue
                
                # Check if alert meets user threshold
                severity_levels = ['low', 'medium', 'high', 'critical']
                if severity_levels.index(severity) >= severity_levels.index(threshold):
                    if severity in ['critical', 'high']:
                        notifications['immediate'].append({
                            'alert': alert,
                            'delivery_method': 'push_notification',
                            'timestamp': datetime.now()
                        })
                    else:
                        notifications['scheduled'].append({
                            'alert': alert,
                            'delivery_method': 'email',
                            'scheduled_time': datetime.now() + timedelta(minutes=15)
                        })
                    current_hour_notifications += 1
                else:
                    notifications['suppressed'].append(alert)
        
        return notifications
    
    # =================== GEOSPATIAL TOOLS ===================
    
    def map_terrain_impacts(self, location: str, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map weather impacts to geographical features"""
        terrain_impacts = {
            'location': location,
            'geographical_features': [],
            'impact_zones': {},
            'risk_assessment': {}
        }
        
        if 'data' not in weather_data:
            return terrain_impacts
        
        current = weather_data['data']
        
        # Simulate geographical feature detection
        location_lower = location.lower()
        
        # Identify terrain features
        if any(term in location_lower for term in ['river', 'creek', 'stream']):
            terrain_impacts['geographical_features'].append('waterway')
            terrain_impacts['impact_zones']['flood_risk'] = {
                'level': 'moderate' if current.get('precipitation', 0) > 5 else 'low',
                'factors': ['proximity_to_water', 'precipitation_rate']
            }
        
        if any(term in location_lower for term in ['mountain', 'hill', 'peak']):
            terrain_impacts['geographical_features'].append('elevation')
            terrain_impacts['impact_zones']['avalanche_risk'] = {
                'level': 'high' if current.get('temperature', 0) > 0 and 'snow' in current.get('description', '') else 'low',
                'factors': ['temperature_fluctuation', 'snow_conditions']
            }
            terrain_impacts['impact_zones']['wind_exposure'] = {
                'level': 'high',
                'wind_multiplier': 1.5
            }
        
        if any(term in location_lower for term in ['forest', 'woods', 'tree']):
            terrain_impacts['geographical_features'].append('forest')
            terrain_impacts['impact_zones']['fire_risk'] = {
                'level': 'high' if current.get('temperature', 0) > 30 and current.get('humidity', 50) < 30 else 'low',
                'factors': ['temperature', 'humidity', 'wind_speed']
            }
        
        if any(term in location_lower for term in ['coast', 'beach', 'shore']):
            terrain_impacts['geographical_features'].append('coastal')
            terrain_impacts['impact_zones']['storm_surge'] = {
                'level': 'moderate' if current.get('wind_speed', 0) > 15 else 'low',
                'factors': ['wind_speed', 'pressure_drop']
            }
        
        # Risk assessment
        high_risk_zones = sum(1 for zone in terrain_impacts['impact_zones'].values() if zone['level'] == 'high')
        moderate_risk_zones = sum(1 for zone in terrain_impacts['impact_zones'].values() if zone['level'] == 'moderate')
        
        if high_risk_zones >= 2:
            terrain_impacts['risk_assessment']['overall'] = 'high'
        elif high_risk_zones >= 1 or moderate_risk_zones >= 2:
            terrain_impacts['risk_assessment']['overall'] = 'moderate'
        else:
            terrain_impacts['risk_assessment']['overall'] = 'low'
        
        return terrain_impacts
    
    def calculate_wind_effects(self, terrain_data: Dict[str, Any], weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate wind effects using fluid dynamics principles"""
        wind_effects = {
            'base_wind_speed': 0,
            'adjusted_wind_speed': 0,
            'wind_patterns': {},
            'turbulence_zones': []
        }
        
        if 'data' not in weather_data:
            return wind_effects
        
        base_wind = weather_data['data'].get('wind_speed', 0)
        wind_direction = weather_data['data'].get('wind_direction', 0)
        
        wind_effects['base_wind_speed'] = base_wind
        
        # Apply terrain modifications
        multiplier = 1.0
        
        geographical_features = terrain_data.get('geographical_features', [])
        
        for feature in geographical_features:
            if feature == 'elevation':
                # Wind speed increases with elevation
                multiplier *= 1.3
                wind_effects['wind_patterns']['orographic_lift'] = True
                wind_effects['turbulence_zones'].append('leeward_slope')
                
            elif feature == 'forest':
                # Forest reduces wind speed
                multiplier *= 0.6
                wind_effects['wind_patterns']['canopy_friction'] = True
                
            elif feature == 'coastal':
                # Coastal areas can have sea/land breezes
                multiplier *= 1.1
                wind_effects['wind_patterns']['thermal_circulation'] = True
                
            elif feature == 'waterway':
                # Open water allows unobstructed flow
                multiplier *= 1.05
        
        # Calculate Beaufort scale equivalent
        adjusted_wind = base_wind * multiplier
        wind_effects['adjusted_wind_speed'] = round(adjusted_wind, 1)
        
        # Beaufort scale classification
        if adjusted_wind < 0.5:
            beaufort = 0  # Calm
        elif adjusted_wind < 1.5:
            beaufort = 1  # Light air
        elif adjusted_wind < 3.3:
            beaufort = 2  # Light breeze
        elif adjusted_wind < 5.5:
            beaufort = 3  # Gentle breeze
        elif adjusted_wind < 7.9:
            beaufort = 4  # Moderate breeze
        elif adjusted_wind < 10.7:
            beaufort = 5  # Fresh breeze
        elif adjusted_wind < 13.8:
            beaufort = 6  # Strong breeze
        elif adjusted_wind < 17.1:
            beaufort = 7  # Near gale
        elif adjusted_wind < 20.7:
            beaufort = 8  # Gale
        elif adjusted_wind < 24.4:
            beaufort = 9  # Strong gale
        elif adjusted_wind < 28.4:
            beaufort = 10  # Storm
        elif adjusted_wind < 32.6:
            beaufort = 11  # Violent storm
        else:
            beaufort = 12  # Hurricane
        
        wind_effects['beaufort_scale'] = beaufort
        
        # Wind direction effects
        wind_effects['wind_patterns']['direction'] = wind_direction
        if 'elevation' in geographical_features:
            # Orographic effects
            if 270 <= wind_direction <= 90:  # Westerly to Easterly
                wind_effects['wind_patterns']['orographic_enhancement'] = 'windward'
            else:
                wind_effects['wind_patterns']['orographic_enhancement'] = 'leeward'
        
        return wind_effects
    
    # =================== ENERGY IMPACT TOOLS ===================
    
    def forecast_energy_generation(self, weather_data: Dict[str, Any], forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast renewable energy generation based on weather"""
        energy_forecast = {
            'solar': {'current': 0, 'forecast': []},
            'wind': {'current': 0, 'forecast': []},
            'efficiency_factors': {}
        }
        
        if 'data' not in weather_data:
            return energy_forecast
        
        current = weather_data['data']
        
        # Solar energy calculation
        temp = current.get('temperature', 20)
        cloud_cover = 0.5  # Simplified - would derive from weather description
        if 'clear' in current.get('description', ''):
            cloud_cover = 0.1
        elif 'cloud' in current.get('description', ''):
            cloud_cover = 0.7
        elif 'overcast' in current.get('description', ''):
            cloud_cover = 0.9
        
        # Solar efficiency decreases with high temperature and cloud cover
        solar_efficiency = max(0, (1 - cloud_cover) * (1 - max(0, (temp - 25) * 0.004)))
        energy_forecast['solar']['current'] = round(solar_efficiency * 100, 1)
        
        # Wind energy calculation
        wind_speed = current.get('wind_speed', 0)
        # Wind turbines typically start generating at 3 m/s, optimal at 12-15 m/s
        if wind_speed < 3:
            wind_efficiency = 0
        elif wind_speed > 25:  # Cut-off speed
            wind_efficiency = 0
        else:
            # Simplified power curve
            wind_efficiency = min(1.0, (wind_speed - 3) / 12)
        
        energy_forecast['wind']['current'] = round(wind_efficiency * 100, 1)
        
        # Forecast calculations
        if forecast_data.get('status') == 'success':
            for forecast in forecast_data['forecasts'][:24]:  # Next 24 hours
                f_temp = forecast['temperature']
                f_wind = forecast['wind_speed']
                f_description = forecast['description']
                
                # Solar forecast
                f_cloud_cover = 0.5
                if 'clear' in f_description:
                    f_cloud_cover = 0.1
                elif 'cloud' in f_description:
                    f_cloud_cover = 0.7
                
                f_solar_eff = max(0, (1 - f_cloud_cover) * (1 - max(0, (f_temp - 25) * 0.004)))
                
                # Wind forecast
                if f_wind < 3 or f_wind > 25:
                    f_wind_eff = 0
                else:
                    f_wind_eff = min(1.0, (f_wind - 3) / 12)
                
                energy_forecast['solar']['forecast'].append({
                    'datetime': forecast['datetime'],
                    'efficiency': round(f_solar_eff * 100, 1)
                })
                
                energy_forecast['wind']['forecast'].append({
                    'datetime': forecast['datetime'],
                    'efficiency': round(f_wind_eff * 100, 1)
                })
        
        # Efficiency factors
        energy_forecast['efficiency_factors'] = {
            'temperature_impact': 'negative' if temp > 25 else 'neutral',
            'cloud_impact': 'high' if cloud_cover > 0.7 else 'low',
            'wind_consistency': 'good' if 5 < wind_speed < 20 else 'poor'
        }
        
        return energy_forecast
    
    def optimize_energy_consumption(self, weather_data: Dict[str, Any], consumption_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize energy consumption based on weather conditions"""
        if consumption_profile is None:
            consumption_profile = {
                'heating': 40,  # % of total consumption
                'cooling': 30,
                'lighting': 15,
                'other': 15
            }
        
        optimization = {
            'recommendations': [],
            'potential_savings': {},
            'load_adjustments': {}
        }
        
        if 'data' not in weather_data:
            return optimization
        
        current = weather_data['data']
        temp = current.get('temperature', 20)
        humidity = current.get('humidity', 50)
        wind_speed = current.get('wind_speed', 0)
        
        # Temperature-based optimizations
        if temp < 15:
            # Heating recommendations
            optimization['recommendations'].append("Increase insulation efficiency")
            optimization['recommendations'].append("Use programmable thermostats")
            optimization['load_adjustments']['heating'] = 1.2  # 20% increase
            optimization['potential_savings']['heating'] = "10-15% with smart scheduling"
            
        elif temp > 25:
            # Cooling recommendations
            optimization['recommendations'].append("Use natural ventilation when possible")
            optimization['recommendations'].append("Close blinds during peak sun hours")
            optimization['load_adjustments']['cooling'] = 1.3  # 30% increase
            optimization['potential_savings']['cooling'] = "15-20% with efficient settings"
        
        # Wind-based optimizations
        if wind_speed > 8:
            optimization['recommendations'].append("Natural ventilation available - reduce AC usage")
            optimization['load_adjustments']['cooling'] = 0.8  # 20% reduction
        
        # Humidity-based optimizations
        if humidity > 70:
            optimization['recommendations'].append("Use dehumidifiers efficiently")
            optimization['load_adjustments']['other'] = 1.1  # 10% increase
        
        # Calculate total load factor
        total_load_factor = sum([
            optimization['load_adjustments'].get('heating', 1.0) * consumption_profile['heating'],
            optimization['load_adjustments'].get('cooling', 1.0) * consumption_profile['cooling'],
            optimization['load_adjustments'].get('lighting', 1.0) * consumption_profile['lighting'],
            optimization['load_adjustments'].get('other', 1.0) * consumption_profile['other']
        ]) / 100
        
        optimization['total_load_factor'] = round(total_load_factor, 2)
        
        return optimization
    
    # =================== ROUTE PLANNING TOOLS ===================
    
    def optimize_routes(self, origin: str, destination: str, weather_data: Dict[str, Any], preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize routes based on weather conditions"""
        if preferences is None:
            preferences = {'avoid_rain': True, 'avoid_snow': True, 'avoid_high_winds': True}
        
        route_optimization = {
            'recommended_route': 'standard',
            'weather_hazards': [],
            'travel_recommendations': [],
            'estimated_delays': 0,
            'alternative_routes': []
        }
        
        if 'data' not in weather_data:
            return route_optimization
        
        current = weather_data['data']
        temp = current.get('temperature', 20)
        precipitation = current.get('precipitation', 0)
        wind_speed = current.get('wind_speed', 0)
        description = current.get('description', '')
        
        # Analyze weather hazards
        if precipitation > 2:
            route_optimization['weather_hazards'].append({
                'type': 'heavy_precipitation',
                'severity': 'moderate' if precipitation < 10 else 'high',
                'impact': 'reduced_visibility_wet_roads'
            })
            route_optimization['estimated_delays'] += 15  # minutes
        
        if temp < 2 and precipitation > 0:
            route_optimization['weather_hazards'].append({
                'type': 'icing_conditions',
                'severity': 'high',
                'impact': 'dangerous_road_conditions'
            })
            route_optimization['estimated_delays'] += 30
        
        if wind_speed > 15:
            route_optimization['weather_hazards'].append({
                'type': 'high_winds',
                'severity': 'moderate' if wind_speed < 25 else 'high',
                'impact': 'vehicle_stability_issues'
            })
            route_optimization['estimated_delays'] += 10
        
        if 'fog' in description:
            route_optimization['weather_hazards'].append({
                'type': 'reduced_visibility',
                'severity': 'high',
                'impact': 'limited_sight_distance'
            })
            route_optimization['estimated_delays'] += 20
        
        # Generate recommendations
        if route_optimization['weather_hazards']:
            route_optimization['travel_recommendations'].extend([
                "Allow extra travel time",
                "Check road conditions before departure",
                "Ensure vehicle is properly equipped"
            ])
            
            if any(h['type'] == 'icing_conditions' for h in route_optimization['weather_hazards']):
                route_optimization['travel_recommendations'].append("Use winter tires or chains")
                route_optimization['recommended_route'] = 'winter_alternative'
            
            if any(h['severity'] == 'high' for h in route_optimization['weather_hazards']):
                route_optimization['travel_recommendations'].append("Consider delaying travel if possible")
                route_optimization['alternative_routes'].append({
                    'type': 'delayed_departure',
                    'delay_hours': 2,
                    'benefit': 'improved_conditions'
                })
        
        # Route alternatives
        if route_optimization['weather_hazards']:
            route_optimization['alternative_routes'].extend([
                {
                    'type': 'highway_alternative',
                    'benefit': 'better_road_maintenance',
                    'trade_off': 'longer_distance'
                },
                {
                    'type': 'scenic_route',
                    'benefit': 'avoid_high_exposure_areas',
                    'trade_off': 'increased_travel_time'
                }
            ])
        
        return route_optimization
    
    def analyze_elevation_profiles(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze elevation profiles for route planning"""
        elevation_analysis = {
            'elevation_changes': [],
            'steep_grades': [],
            'weather_impact_zones': [],
            'recommendations': []
        }
        
        # Simulate elevation profile (in real app, you'd use elevation APIs)
        # Generate sample elevation points
        distance_points = list(range(0, 101, 10))  # 0 to 100 km in 10km intervals
        elevations = []
        
        # Generate realistic elevation profile
        base_elevation = 200
        for i, dist in enumerate(distance_points):
            # Simulate some hills and valleys
            elevation = base_elevation + 50 * math.sin(dist * 0.1) + 100 * math.sin(dist * 0.03)
            elevations.append(round(elevation))
        
        # Analyze elevation changes
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i-1]
            distance_segment = distance_points[i] - distance_points[i-1]
            grade = (elevation_change / (distance_segment * 1000)) * 100  # Grade in %
            
            elevation_analysis['elevation_changes'].append({
                'segment': f"{distance_points[i-1]}-{distance_points[i]}km",
                'elevation_change': elevation_change,
                'grade': round(grade, 2),
                'difficulty': 'steep' if abs(grade) > 6 else 'moderate' if abs(grade) > 3 else 'gentle'
            })
            
            if abs(grade) > 6:
                elevation_analysis['steep_grades'].append({
                    'location': f"km {distance_points[i]}",
                    'grade': round(grade, 2),
                    'direction': 'uphill' if grade > 0 else 'downhill'
                })
        
        # Weather impact zones
        for i, elevation in enumerate(elevations):
            if elevation > 800:  # High elevation
                elevation_analysis['weather_impact_zones'].append({
                    'location': f"km {distance_points[i]}",
                    'elevation': elevation,
                    'impacts': ['temperature_drop', 'increased_wind', 'precipitation_change'],
                    'severity': 'high' if elevation > 1200 else 'moderate'
                })
        
        # Generate recommendations
        if elevation_analysis['steep_grades']:
            elevation_analysis['recommendations'].extend([
                "Prepare for steep grades - check brakes and engine cooling",
                "Allow extra time for climbing",
                "Be cautious on downhill sections"
            ])
        
        if elevation_analysis['weather_impact_zones']:
            elevation_analysis['recommendations'].extend([
                "Expect weather changes at higher elevations",
                "Pack appropriate clothing for temperature variations",
                "Monitor conditions for mountain weather"
            ])
        
        return elevation_analysis

# Utility functions for coordinate conversion and distance calculation
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def get_coordinates(location: str) -> Tuple[float, float]:
    """Get coordinates for a location (simplified - would use geocoding API)"""
    # Simplified coordinate lookup (in real app, use geocoding)
    location_coords = {
        'london': (51.5074, -0.1278),
        'paris': (48.8566, 2.3522),
        'new york': (40.7128, -74.0060),
        'tokyo': (35.6762, 139.6503),
        'sydney': (-33.8688, 151.2093)
    }
    
    location_lower = location.lower()
    for city, coords in location_coords.items():
        if city in location_lower:
            return coords
    
    # Default coordinates if not found
    return (0.0, 0.0)