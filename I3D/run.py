#!/usr/bin/env python3
"""
Startup script for the Gem Infer FastAPI microservice
"""
import os
import sys
import uvicorn
from pathlib import Path

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        "checkpoint/nslt_100_005624_0.756.pt",
        "weights/rgb_imagenet.pt", 
        "preprocess/wlasl_class_list.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all model files are in the correct locations.")
        return False
    
    print("✅ All required model files found")
    return True

def check_environment():
    """Check environment variables"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️  WARNING: GEMINI_API_KEY not found in environment")
        print("   Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        print("   Get your API key at: https://makersuite.google.com/app/apikey")
    else:
        print("✅ Gemini API key configured")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Gem Infer I3D FastAPI Service")
    print("=" * 50)
    
    # Check requirements
    if not check_required_files():
        sys.exit(1)
    
    check_environment()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    min_glosses = int(os.getenv("MIN_GLOSSES_FOR_GEMINI", "5"))
    
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Min glosses for Gemini: {min_glosses}")
    
    try:
        # Import the app from the modified gem_infer
        from gem_infer import app
        
        print(f"\n🎯 Starting service on http://{host}:{port}")
        print("📚 API docs will be available at: http://localhost:5000/docs")
        print("🔍 Health check: http://localhost:5000/health")
        print("\nPress Ctrl+C to stop the service\n")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Gem Infer service...")
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        sys.exit(1)