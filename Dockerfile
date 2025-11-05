# Step 1: Use a lightweight Python base image
FROM python:3.13-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy setup.py and requirements.txt first
COPY setup.py requirements.txt ./

# Step 4: Copy your package code (so -e . works)
COPY src/ ./src  

# Step 5: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your app (templates, static, etc.)
COPY . .

# Step 7: Expose the port your Flask app will run on
EXPOSE 5000

# Step 8: Start the app with Gunicorn
CMD ["gunicorn", "app:app", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:5000"]
