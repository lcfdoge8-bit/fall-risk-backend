import sys
import subprocess

# 1. Define packages
REQUIRED_PACKAGES = [
    ("fastapi[standard]", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("hashlib", "hashlib"),
    ("secrets", "secrets"),
    ("gradio", "gradio"),
    ("sqlalchemy", "sqlalchemy"),
    ("aiosqlite", "aiosqlite"),
    ("pyjwt", "jwt"),
    ("passlib[bcrypt]", "passlib"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("faker", "faker"),
    ("lime", "lime"),
]

print("Checking dependencies...")

def libmain():
    for package_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            print(f"{import_name} is already available.")
        except ImportError:
            print(f"{package_name} not found. Installing now...")
            # Quotes are automatically handled safely by passing as a list item to subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} successfully installed!")

