from app import app, socketio  # Import the app and socketio instance

if __name__ == "__main__":
    socketio.run(app, debug=True)  # Use socketio.run to start the server
