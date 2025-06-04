"""
Configuration module for the Stock Analysis CrewAI project.
Handles environment setup and API key validation.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for managing API keys and settings."""
    
    def __init__(self):
        self._setup_environment()
        self._validate_keys()
    
    def _setup_environment(self):
        """Setup environment variables to force Gemini usage."""
        # Force CrewAI to avoid OpenAI
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_MODEL_NAME"] = ""
        os.environ["OPENAI_API_BASE"] = ""
        
        # Set API keys from environment
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
    
    def _validate_keys(self):
        """Validate required API keys."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")

        if not self.openweather_api_key:
            print("Warning: OPENWEATHER_API_KEY not found. Weather data retrieval will be limited.")

    def get_llm(self):
        """Get configured Gemini LLM instance."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.google_api_key,
                temperature=0.1,  # Lower temperature for more consistent tool usage
                verbose=False,
                max_output_tokens=2048,
                # Removed convert_system_message_to_human as it's deprecated
                # Removed safety_settings to use default values
            )
            # Test connection
            test_response = llm.invoke("Hello")
            print(f"✅ Gemini LLM connection successful: {len(str(test_response))} chars")
            return llm
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Gemini LLM: {e}")
    
    def display_status(self):
        """Display configuration status."""
        print("🔧 Marketing Analysis AI Configuration")
        print(f"📍 Google API Key: {'✅ Found' if self.google_api_key else '❌ Missing'}")
        print(f"🌦️ OpenWeather API Key: {'✅ Found' if self.openweather_api_key else '❌ Missing'}")
        print("✅ Configuration loaded - CrewAI will use Gemini exclusively")

# Global config instance
config = Config()