from crewai import Crew, Process
from agents import WeatherAgents
from tasks import WeatherTasks, WeatherWorkflows
from config import config
import json
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force environment setup to prevent OpenAI usage
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_MODEL_NAME"] = ""
os.environ["OPENAI_API_BASE"] = ""

class EnhancedRateLimitManager:
    """Enhanced rate limit manager that respects API-suggested delays and implements minute-based waiting"""
    
    def __init__(self, requests_per_minute: int = 5, requests_per_hour: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
        self.last_request_time = 0
        self.min_delay = 15  # Increased minimum delay to 15 seconds
        self.quota_exceeded_until = None  # Track when quota will reset
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
    def extract_retry_delay_from_error(self, error_message: str) -> int:
        """Extract the retry delay suggested by the API from error message"""
        try:
            # Look for retry_delay { seconds: X } pattern
            match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_message)
            if match:
                return int(match.group(1))
            
            # Fallback: look for other patterns
            match = re.search(r'retry after (\d+) seconds?', error_message, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        except Exception as e:
            logger.warning(f"Could not extract retry delay from error: {e}")
        
        return 60  # Default to 60 seconds if we can't parse
    
    def handle_quota_exceeded(self, error_message: str):
        """Handle quota exceeded error by setting appropriate wait time"""
        suggested_delay = self.extract_retry_delay_from_error(error_message)
        
        # Add buffer time to the suggested delay
        buffer_time = max(30, suggested_delay * 0.5)  # At least 30 seconds buffer
        total_delay = suggested_delay + buffer_time
        
        self.quota_exceeded_until = datetime.now() + timedelta(seconds=total_delay)
        self.consecutive_failures += 1
        
        logger.warning(f"Quota exceeded. Suggested delay: {suggested_delay}s, Total delay with buffer: {total_delay}s")
        logger.warning(f"Will resume requests after: {self.quota_exceeded_until}")
        
        # If we have too many consecutive failures, implement exponential backoff
        if self.consecutive_failures >= self.max_consecutive_failures:
            additional_delay = 2 ** self.consecutive_failures * 60  # Exponential backoff in minutes
            self.quota_exceeded_until += timedelta(seconds=additional_delay)
            logger.warning(f"Multiple consecutive failures detected. Extended delay until: {self.quota_exceeded_until}")
    
    def is_quota_exceeded(self) -> bool:
        """Check if we're still in quota exceeded state"""
        if self.quota_exceeded_until is None:
            return False
        
        if datetime.now() < self.quota_exceeded_until:
            return True
        else:
            # Reset quota exceeded state
            self.quota_exceeded_until = None
            self.consecutive_failures = 0
            return False
    
    def can_make_request(self) -> bool:
        """Check if we can make a request based on rate limits and quota status"""
        # First check if quota is exceeded
        if self.is_quota_exceeded():
            return False
        
        now = time.time()
        
        # Clean old requests
        self.minute_requests = [req_time for req_time in self.minute_requests if now - req_time < 60]
        self.hour_requests = [req_time for req_time in self.hour_requests if now - req_time < 3600]
        
        # Check rate limits
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        if len(self.hour_requests) >= self.requests_per_hour:
            return False
        
        # Check minimum delay
        if now - self.last_request_time < self.min_delay:
            return False
        
        return True
    
    def wait_for_rate_limit(self):
        """Wait until we can make a request, with special handling for quota exceeded"""
        while not self.can_make_request():
            if self.is_quota_exceeded():
                remaining_time = (self.quota_exceeded_until - datetime.now()).total_seconds()
                if remaining_time > 0:
                    logger.info(f"Quota exceeded. Waiting {remaining_time:.0f} seconds until {self.quota_exceeded_until}")
                    
                    # If we need to wait more than 5 minutes, log periodic updates
                    if remaining_time > 300:
                        # Wait in chunks and log progress
                        while remaining_time > 0:
                            sleep_time = min(60, remaining_time)  # Wait max 1 minute at a time
                            time.sleep(sleep_time)
                            remaining_time = (self.quota_exceeded_until - datetime.now()).total_seconds()
                            if remaining_time > 0:
                                logger.info(f"Still waiting... {remaining_time:.0f} seconds remaining")
                    else:
                        time.sleep(remaining_time)
                continue
            
            # Check if we need to wait for the next minute
            now = time.time()
            minute_requests_count = len(self.minute_requests)
            
            if minute_requests_count >= self.requests_per_minute:
                # Wait until the next minute
                oldest_request = min(self.minute_requests) if self.minute_requests else now
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    logger.info(f"Per-minute limit reached ({minute_requests_count}/{self.requests_per_minute}). Waiting {wait_time:.0f} seconds for next minute.")
                    time.sleep(wait_time)
                continue
            
            # Check minimum delay
            if now - self.last_request_time < self.min_delay:
                wait_time = self.min_delay - (now - self.last_request_time)
                logger.info(f"Minimum delay not met. Waiting {wait_time:.0f} seconds.")
                time.sleep(wait_time)
                continue
            
            # General wait
            time.sleep(1)
    
    def record_request(self):
        """Record that a request was made"""
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.last_request_time = now
    
    def record_success(self):
        """Record a successful request to reset failure counter"""
        self.consecutive_failures = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of rate limiter"""
        now = time.time()
        return {
            'can_make_request': self.can_make_request(),
            'minute_requests_count': len(self.minute_requests),
            'hour_requests_count': len(self.hour_requests),
            'requests_per_minute_limit': self.requests_per_minute,
            'requests_per_hour_limit': self.requests_per_hour,
            'last_request_time': self.last_request_time,
            'time_since_last_request': now - self.last_request_time,
            'quota_exceeded': self.is_quota_exceeded(),
            'quota_reset_time': self.quota_exceeded_until.isoformat() if self.quota_exceeded_until else None,
            'consecutive_failures': self.consecutive_failures
        }

def rate_limited_execution(rate_limiter: EnhancedRateLimitManager):
    """Decorator to add rate limiting to any function"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get rate limiter from self if not provided
            limiter = rate_limiter if rate_limiter is not None else getattr(self, 'rate_limiter', None)
            
            if limiter is None:
                raise ValueError("Rate limiter not available - check initialization")
            
            # Wait for rate limit before executing
            limiter.wait_for_rate_limit()
            
            try:
                # Record the request
                limiter.record_request()
                
                # Execute the function
                result = func(self, *args, **kwargs)
                
                # Record success
                limiter.record_success()
                
                return result
                
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a quota/rate limit error
                if any(keyword in error_message.lower() for keyword in 
                      ['429', 'quota', 'rate limit', 'resourceexhausted', 'exceeded']):
                    logger.error(f"Rate limit/quota error detected: {error_message}")
                    limiter.handle_quota_exceeded(error_message)
                
                raise e
        
        return wrapper
    return decorator

class WeatherCrew:
    """Orchestrates weather intelligence operations with enhanced rate limiting"""
    
    def __init__(self):
        # Ensure config is loaded and display status
        config.display_status()
        
        # Get the configured LLM
        self.llm = config.get_llm()
        
        # Initialize enhanced rate limit manager with very conservative limits
        self.rate_limiter = EnhancedRateLimitManager(
            requests_per_minute=3,  # Very conservative
            requests_per_hour=50    # Very conservative
        )
        
        # Initialize agents and tasks with the LLM
        self.weather_agents = WeatherAgents()
        self.weather_tasks = WeatherTasks()
        self.workflows = WeatherWorkflows()
        self.crews = {}
        self._initialize_crews()
    
    def _initialize_crews(self):
        """Initialize minimal crews with very low RPM limits"""
        
        # Ultra-lightweight crew with minimal agents
        self.crews['minimal_weather'] = Crew(
            agents=[
                self.weather_agents.get_agent('data_ingestion'),
                self.weather_agents.get_agent('decision_engine')  # Only 2 agents
            ],
            tasks=[
                self.weather_tasks.get_task('data_collection'),
                self.weather_tasks.get_task('recommendations')    # Only 2 tasks
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=1,  # Only 1 request per minute
            share_crew=True,
            manager_llm=self.llm,
            function_calling_llm=self.llm
        )
        
        # Emergency-only crew for critical requests
        self.crews['emergency_weather'] = Crew(
            agents=[
                self.weather_agents.get_agent('alert_triaging')  # Single agent
            ],
            tasks=[
                self.weather_tasks.get_task('alert_management')  # Single task
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=1,  # Only 1 request per minute
            share_crew=True,
            manager_llm=self.llm,
            function_calling_llm=self.llm
        )

class WeatherCrewManager:
    """Manages and executes weather crew operations with enhanced rate limiting"""
    
    def __init__(self):
        self.weather_crew = WeatherCrew()
        self.execution_history = []
        self.active_crews = {}
        self.rate_limiter = self.weather_crew.rate_limiter
    
    @rate_limited_execution(None)  # Rate limiter will be accessed from self
    def _execute_crew_internal(self, crew_name: str, inputs: Dict[str, Any]) -> Any:
        """Internal crew execution with rate limiting"""
        if crew_name not in self.weather_crew.crews:
            raise ValueError(f"Crew '{crew_name}' not found")
        
        crew = self.weather_crew.crews[crew_name]
        return crew.kickoff(inputs=inputs)
    
    @retry(
        stop=stop_after_attempt(5),  # Increased retry attempts
        wait=wait_exponential(multiplier=2, min=10, max=300),  # Longer waits
        retry=retry_if_exception_type(Exception)
    )
    async def execute_crew_with_retry(self, crew_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute crew with enhanced retry logic"""
        
        execution_start = datetime.now()
        
        try:
            logger.info(f"Executing crew: {crew_name}")
            
            # Check rate limit status before execution
            status = self.rate_limiter.get_status()
            logger.info(f"Rate limit status: {status['minute_requests_count']}/{status['requests_per_minute_limit']} this minute")
            
            # Execute the crew with rate limiting
            result = self._execute_crew_internal(crew_name, inputs)
            
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()
            
            # Log successful execution
            execution_record = {
                'crew_name': crew_name,
                'inputs': inputs,
                'result': result,
                'execution_time': execution_time,
                'timestamp': execution_start.isoformat(),
                'status': 'success'
            }
            self.execution_history.append(execution_record)
            
            logger.info(f"Successfully executed {crew_name} in {execution_time:.2f}s")
            
            return {
                'success': True,
                'crew_name': crew_name,
                'result': result,
                'execution_time': execution_time,
                'timestamp': execution_start.isoformat()
            }
            
        except Exception as e:
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()
            error_message = str(e)
            
            logger.error(f"Error executing {crew_name}: {error_message}")
            
            # Log error
            error_record = {
                'crew_name': crew_name,
                'inputs': inputs,
                'error': error_message,
                'execution_time': execution_time,
                'timestamp': execution_start.isoformat(),
                'status': 'error'
            }
            self.execution_history.append(error_record)
            
            # Handle specific error types
            if any(keyword in error_message.lower() for keyword in 
                  ['429', 'quota', 'rate limit', 'resourceexhausted', 'exceeded']):
                logger.warning("Quota/rate limit error detected, implementing extended delay")
                self.rate_limiter.handle_quota_exceeded(error_message)
                
                # For quota errors, wait before retrying
                if self.rate_limiter.is_quota_exceeded():
                    remaining_time = (self.rate_limiter.quota_exceeded_until - datetime.now()).total_seconds()
                    logger.info(f"Waiting {remaining_time:.0f} seconds before retry due to quota exceeded")
                    time.sleep(min(remaining_time, 60))  # Wait up to 1 minute per retry attempt
            
            raise e
    
    async def execute_crew(self, crew_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific crew with given inputs"""
        try:
            return await self.execute_crew_with_retry(crew_name, inputs)
        except Exception as e:
            return {
                'success': False,
                'crew_name': crew_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'rate_limit_status': self.rate_limiter.get_status()
            }
    
    def get_minimal_weather_report(self, location: str, user_preferences: Dict = None) -> Dict[str, Any]:
        """Get minimal weather report using ultra-lightweight crew"""
        inputs = {
            'location': location,
            'user_preferences': user_preferences or {},
            'request_type': 'minimal_weather',
            'timestamp': datetime.now().isoformat()
        }
        
        return asyncio.run(self.execute_crew('minimal_weather', inputs))
    
    def get_emergency_weather_alert(self, location: str) -> Dict[str, Any]:
        """Get emergency weather alerts only"""
        inputs = {
            'location': location,
            'request_type': 'emergency_alert',
            'timestamp': datetime.now().isoformat()
        }
        
        return asyncio.run(self.execute_crew('emergency_weather', inputs))
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get detailed rate limit status"""
        return self.rate_limiter.get_status()
    
    def wait_for_next_available_slot(self):
        """Manually wait for next available request slot"""
        logger.info("Waiting for next available request slot...")
        self.rate_limiter.wait_for_rate_limit()
        logger.info("Ready to make next request")

# Weather App Integration with Enhanced Rate Limiting
class WeatherAppCrew:
    """Main interface for weather app integration with enhanced rate limiting"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or getattr(config, 'openweather_api_key', None)
        if not self.api_key:
            logger.warning("No OpenWeather API key provided")
        
        self.crew_manager = WeatherCrewManager()
        self.default_preferences = {
            'units': 'metric',
            'language': 'en',
            'alerts_enabled': True,
            'detailed_forecasts': False,
            'minimal_mode': True  # Force minimal mode
        }
    
    def get_weather_for_location(self, location: str, preferences: Dict = None) -> Dict[str, Any]:
        """Get minimal weather data for a location with enhanced rate limiting"""
        user_prefs = {**self.default_preferences, **(preferences or {})}
        
        # Always check rate limits first
        rate_status = self.crew_manager.get_rate_limit_status()
        
        if not rate_status['can_make_request']:
            wait_message = "Rate limit exceeded."
            if rate_status['quota_exceeded']:
                wait_message += f" Quota reset at: {rate_status['quota_reset_time']}"
            else:
                wait_message += f" Next slot available in {rate_status['time_since_last_request']:.0f} seconds"
            
            return {
                'success': False,
                'error': wait_message,
                'rate_limit_status': rate_status,
                'suggested_action': 'wait_and_retry'
            }
        
        logger.info(f"Making weather request for {location} (Rate status: {rate_status['minute_requests_count']}/{rate_status['requests_per_minute_limit']} this minute)")
        
        return self.crew_manager.get_minimal_weather_report(location, user_prefs)
    
    def get_emergency_alerts(self, location: str) -> Dict[str, Any]:
        """Get emergency weather alerts only"""
        rate_status = self.crew_manager.get_rate_limit_status()
        
        if not rate_status['can_make_request']:
            return {
                'success': False,
                'error': 'Rate limit exceeded for emergency alerts',
                'rate_limit_status': rate_status
            }
        
        return self.crew_manager.get_emergency_weather_alert(location)
    
    def wait_for_next_request(self):
        """Wait until next request can be made"""
        self.crew_manager.wait_for_next_available_slot()
    
    def get_detailed_rate_info(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting information"""
        status = self.crew_manager.get_rate_limit_status()
        
        # Add user-friendly information
        status['user_friendly'] = {
            'can_make_request_now': status['can_make_request'],
            'requests_remaining_this_minute': max(0, status['requests_per_minute_limit'] - status['minute_requests_count']),
            'requests_remaining_this_hour': max(0, status['requests_per_hour_limit'] - status['hour_requests_count']),
            'quota_exceeded': status['quota_exceeded'],
            'estimated_wait_time_seconds': self._estimate_wait_time(status)
        }
        
        return status
    
    def _estimate_wait_time(self, status: Dict[str, Any]) -> float:
        """Estimate wait time until next request can be made"""
        if status['quota_exceeded'] and status['quota_reset_time']:
            quota_reset = datetime.fromisoformat(status['quota_reset_time'])
            return max(0, (quota_reset - datetime.now()).total_seconds())
        
        if status['minute_requests_count'] >= status['requests_per_minute_limit']:
            return max(0, 60 - status['time_since_last_request'])
        
        if status['time_since_last_request'] < 15:  # min_delay
            return 15 - status['time_since_last_request']
        
        return 0

def get_user_location() -> str:
    """Get user location through input or automatic detection"""
    import requests
    
    # First, try to get location from user input
    try:
        user_input = input("\n📍 Enter your location (city, state/country) or press Enter for auto-detection: ").strip()
        
        if user_input:
            print(f"📍 Using location: {user_input}")
            return user_input
        
        # If no input, try to auto-detect location
        print("🔍 Auto-detecting your location...")
        
        # Try to get location from IP address
        try:
            # Using a free IP geolocation service
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    city = data.get('city', '')
                    region = data.get('regionName', '')
                    country = data.get('country', '')
                    
                    # Format location string
                    if city and region and country:
                        if country == 'United States':
                            location = f"{city}, {region}"
                        else:
                            location = f"{city}, {country}"
                        print(f"📍 Auto-detected location: {location}")
                        return location
        except Exception as e:
            print(f"⚠️  Auto-detection failed: {str(e)}")
        
        # Fallback to asking user again
        print("❌ Could not auto-detect location.")
        fallback_location = input("📍 Please enter your location manually: ").strip()
        if fallback_location:
            return fallback_location
        else:
            print("⚠️  No location provided, using default: New York, NY")
            return "New York, NY"
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        exit(0)
    except Exception as e:
        print(f"⚠️  Error getting location: {str(e)}")
        return "New York, NY"

def format_weather_output(weather_result: Dict[str, Any]) -> str:
    """Format weather result into human-readable output, showing only current weather recommendations."""
    if not weather_result.get('success', False):
        return f"❌ Weather request failed: {weather_result.get('error', 'Unknown error')}"

    result_data = weather_result.get('result', {})

    # Handle case where result is a string
    if isinstance(result_data, str):
        return f"🌤 Weather Information:\n{result_data}"

    # Handle dictionary result (expected case from the crew)
    if isinstance(result_data, dict):
        formatted_output = "🌤 Weather Report for Your Location:\n"

        # Extract and display current weather conditions
        if 'weather_data' in result_data and isinstance(result_data['weather_data'], dict):
            weather_data = result_data['weather_data']
            formatted_output += "Current Conditions:\n"
            if 'temperature' in weather_data:
                formatted_output += f"🌡 Temperature: {weather_data['temperature']}°C\n"
            if 'condition' in weather_data:
                formatted_output += f"☁️ Condition: {weather_data['condition']}\n"
            if 'humidity' in weather_data:
                formatted_output += f"💧 Humidity: {weather_data['humidity']}%\n"
            if 'wind' in weather_data:
                formatted_output += f"💨 Wind: {weather_data['wind']}\n"

        # Display only relevant recommendations
        if 'recommendations' in result_data and isinstance(result_data['recommendations'], dict):
            formatted_output += "\nBased on the current weather at your location, here are your recommendations:\n"
            recommendations = result_data['recommendations']
            has_recommendations = False
            for category, recs in recommendations.items():
                if recs:  # Only include categories with actual recommendations
                    formatted_output += f"\n{category.capitalize()}:\n"
                    for rec in recs:
                        formatted_output += f"- {rec}\n"
                    has_recommendations = True
            if not has_recommendations:
                formatted_output += "- No specific recommendations for current conditions.\n"

        return formatted_output

    return f"🌤 Weather Information:\n{str(result_data)}"

def interactive_weather_session():
    """Run an interactive weather session with location input"""
    print("🌤️ Welcome to Enhanced Weather Intelligence System")
    print("=" * 50)
    
    # Initialize the weather app crew system
    weather_app = WeatherAppCrew()
    
    while True:
        try:
            # Get user location
            location = get_user_location()
            
            # Get user preferences
            print("\n⚙️  Weather Preferences:")
            units = input("Units (metric/imperial) [metric]: ").strip().lower() or "metric"
            
            preferences = {
                "units": units,
                "minimal_mode": True
            }
            
            # Check rate limits
            print("\n=== Rate Limit Status ===")
            rate_info = weather_app.get_detailed_rate_info()
            print(f"Ready to make request: {rate_info['user_friendly']['can_make_request_now']}")
            print(f"Requests remaining this minute: {rate_info['user_friendly']['requests_remaining_this_minute']}")
            
            if not rate_info['user_friendly']['can_make_request_now']:
                wait_time = rate_info['user_friendly']['estimated_wait_time_seconds']
                print(f"⏳ Need to wait {wait_time:.0f} seconds before next request")
                
                wait_choice = input("Wait automatically? (y/n) [y]: ").strip().lower()
                if wait_choice != 'n':
                    print("⏳ Waiting for next available request slot...")
                    weather_app.wait_for_next_request()
                else:
                    print("❌ Skipping weather request due to rate limits")
                    continue
            
            # Make weather request
            print(f"\n=== Weather Request for {location} ===")
            weather_result = weather_app.get_weather_for_location(location, preferences)
            
            # Display results
            weather_output = format_weather_output(weather_result)
            print(weather_output)
            
            # Show execution time if available
            if weather_result.get('success') and 'execution_time' in weather_result:
                print(f"\n⏱️  Request completed in {weather_result['execution_time']:.2f} seconds")
            
            # Ask if user wants to continue
            print("\n" + "=" * 50)
            continue_choice = input("🔄 Get weather for another location? (y/n) [n]: ").strip().lower()
            if continue_choice != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n👋 Thanks for using Weather Intelligence System!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            retry_choice = input("🔄 Try again? (y/n) [y]: ").strip().lower()
            if retry_choice == 'n':
                break
    
    # Show final rate limit status
    print(f"\n=== Session Summary ===")
    final_status = weather_app.get_detailed_rate_info()
    print(f"Requests remaining this minute: {final_status['user_friendly']['requests_remaining_this_minute']}")
    print(f"Requests remaining this hour: {final_status['user_friendly']['requests_remaining_this_hour']}")
    print("👋 Goodbye!")

# Usage example with user location input and interactive session
if __name__ == "__main__":
    # Run interactive weather session
    interactive_weather_session()