"""
Locust load testing scenario for Pet Breed Classification API.
"""

import io
from locust import HttpUser, between, task
from PIL import Image


class PetClassificationUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        # Generate dummy 224x224 JPEG image payload
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        self.image_bytes = buf.getvalue()

    @task(5)
    def predict_image(self):
        """Simulate user uploading a pet photo."""
        files = {"file": ("pet.jpg", self.image_bytes, "image/jpeg")}
        self.client.post("/predict", files=files)

    @task(1)
    def health_check(self):
        """Simulate container load balancer health probe."""
        self.client.get("/health")
