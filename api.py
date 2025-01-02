
import requests
from pymongo import MongoClient
from datetime import datetime
import pytz

apisend = False


def notify_person_event(event_type, timestamp, original_count, new_count):
    """
    Sends a notification to the API about a person entering or exiting and logs the event in the database.
    :param event_type: 'Enter' for entering the frame, 'Exit' for exiting the frame.
    :param timestamp: The timestamp of the event.
    :param original_count: The initial count of people before the event.
    :param new_count: The new count of people after the event.
    """
    # 1. API-Benachrichtigung senden
    url = "https://ntfy.sh/Sagasu"
    data = f"{event_type}: {original_count} -> {new_count} at {timestamp}"  # Klarere Datenstruktur

    #print("Notify_person")
    save_event(new_count)
    if apisend:
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            print(f"Notification sent: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send notification: {e}")

# Save event in a MonogDb database

def save_event(new_count):
    """
    Saves the event in a MongoDB database.
    :param event_type: 'Enter' for entering the frame, 'Exit' for exiting the frame.
    :param original_count: The initial count of people before the event.
    :param new_count: The new count of people after the event.
    """
    # Verbindung zur MongoDB herstellen
    uri = "mongodb+srv://htl3r:ILoveZAI123@statisticdata.i2ntz.mongodb.net/?retryWrites=true&w=majority&appName=StatisticData"
    client = MongoClient(uri)
    db = client['Data']
    collection = db['PeopleCount']


    vienna_tz = pytz.timezone("Europe/Vienna")

    # Aktuelle Uhrzeit in der Zeitzone von Wien abrufen
    current_time_vienna = datetime.now(vienna_tz)

    # Dokument erstellen
    event_document = {
        "currentCount": new_count,
        "LastChange": current_time_vienna  # Aktueller Zeitstempel
    }

    # Dokument einfügen
    collection.insert_one(event_document)

    existing_document = collection.find_one({"unique_id": 0})

    if existing_document:
        # Prepare the update
        update = {
            "$set": { # Set past to current value
                "currentCount": new_count,           # Set current to new value
                "LastChange": current_time_vienna              # Update the date to current Vienna time
            }
        }

        # Perform the update
        result = collection.update_one({"unique_id": 0}, update)

    # Verbindung schließen
    client.close()
