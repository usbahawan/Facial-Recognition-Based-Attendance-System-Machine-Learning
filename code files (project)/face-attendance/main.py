from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from plyer import camera
import requests
import os

# ------------------------
# Define Screens
# ------------------------
class MainScreen(Screen):
    def capture_and_send(self):
        filename = "captured_face.jpg"
        # Capture using mobile camera
        camera.take_picture(filename, self.send_to_server)

    def send_to_server(self, filepath):
        url = "http://192.168.100.21:5000/recognize"
        with open(filepath, "rb") as f:
            files = {"image": f}
            try:
                response = requests.post(url, files=files)
                data = response.json()
                self.ids.result_label.text = f"Name: {data['name']}\nConfidence: {data['confidence']*100:.1f}%\nTime: {data['timestamp']}"
            except Exception as e:
                self.ids.result_label.text = f"Error: {e}"

# ------------------------
# Kivy UI
# ------------------------
KV = """
ScreenManager:
    MainScreen:
        name: "main"

<MainScreen>:
    BoxLayout:
        orientation: "vertical"
        padding: 20
        spacing: 20

        MDRaisedButton:
            text: "Capture & Mark Attendance"
            on_release: root.capture_and_send()

        MDLabel:
            id: result_label
            text: "Result will appear here"
            halign: "center"
"""

class AttendanceApp(MDApp):
    def build(self):
        return Builder.load_string(KV)

if __name__ == "__main__":
    AttendanceApp().run()
