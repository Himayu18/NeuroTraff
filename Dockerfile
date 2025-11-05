# Step 1: Use a lightweight Python base image
FROM python:3.13-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only requirements first for caching
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your app
COPY . .

# Step 6: Expose the port your Flask app will run on
EXPOSE 5000

# Step 7: Start the app with Gunicorn
CMD ["gunicorn", "app:app", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:5000"]
