#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python environment
check_python_env() {
    echo -e "${BLUE}Checking Python environment...${NC}"
    
    # Check if Python is installed
    if ! command_exists python3; then
        echo "Python 3 is not installed. Please install it first."
        exit 1
    fi
    
    # Check if virtual environment exists, create if it doesn't
    if [ ! -d ".venv" ]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
    
    # Check if requirements are installed
    if [ ! -f ".venv/.requirements_installed" ]; then
        echo -e "${BLUE}Installing requirements...${NC}"
        pip install -r requirements.txt
        touch .venv/.requirements_installed
    fi
}

# Function to check .env file
check_env_file() {
    echo -e "${BLUE}Checking .env file...${NC}"
    if [ ! -f ".env" ]; then
        echo "Creating .env file..."
        echo "ANTHROPIC_API_KEY=your_key_here" > .env
        echo "ADMIN_PASSWORD=your_password_here" >> .env
        echo "Please edit the .env file with your actual API key and admin password."
        exit 1
    fi
}

# Function to run the main application
run_geddesghost() {
    echo -e "${GREEN}Starting GeddesGhost...${NC}"
    streamlit run geddesghost.py
}

# Function to run the admin dashboard
run_admin_dashboard() {
    echo -e "${GREEN}Starting Admin Dashboard...${NC}"
    streamlit run admin_dashboard_temp.py
}

# Main script
echo -e "${BLUE}=== GeddesGhost Launcher ===${NC}"

# Check Python environment
check_python_env

# Check .env file
check_env_file

# Menu
while true; do
    echo -e "\n${BLUE}What would you like to run?${NC}"
    echo "1) GeddesGhost (Main Application)"
    echo "2) Admin Dashboard"
    echo "3) Run Both (in separate terminals)"
    echo "4) Exit"
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            run_geddesghost
            ;;
        2)
            run_admin_dashboard
            ;;
        3)
            # Check if lxterminal is installed for Raspberry Pi
            if command_exists lxterminal; then
                lxterminal -e "bash -c 'source .venv/bin/activate && streamlit run geddesghost.py'" &
                lxterminal -e "bash -c 'source .venv/bin/activate && streamlit run admin_dashboard_temp.py'" &
            else
                echo "lxterminal is not installed. Installing..."
                sudo apt-get update
                sudo apt-get install -y lxterminal
                lxterminal -e "bash -c 'source .venv/bin/activate && streamlit run geddesghost.py'" &
                lxterminal -e "bash -c 'source .venv/bin/activate && streamlit run admin_dashboard_temp.py'" &
            fi
            ;;
        4)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done 