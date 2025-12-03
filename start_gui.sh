#!/bin/bash

# Quick Start Script for Neural Image Authentication GUI
# This script helps you launch the GUI applications easily

echo "=================================="
echo "Neural Image Authentication System"
echo "=================================="
echo ""
echo "Which GUI would you like to use?"
echo ""
echo "1) Web GUI (Browser-based) - Recommended"
echo "2) Desktop GUI (Tkinter)"
echo "3) Train models first"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Web GUI..."
        echo "Open your browser to: http://localhost:5000"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        python web_gui.py
        ;;
    2)
        echo ""
        echo "Starting Desktop GUI..."
        echo ""
        python gui_app.py
        ;;
    3)
        echo ""
        echo "Starting training (this may take 1-2 hours)..."
        echo ""
        read -p "How many epochs? [default: 50]: " epochs
        epochs=${epochs:-50}
        echo "Training for $epochs epochs..."
        python -c "from neural_image_auth.main import train_and_save; train_and_save(num_epochs=$epochs)"
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
