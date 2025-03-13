# Project Setup Guide

## Overview
This project is a FastAPI-based web application designed to manage employee attendance and conference planning. It supports authentication, user role management, database interaction, JWT handling, email notifications, and data security. 

### Key Features:
- **User Authentication**: Secure login with JWT-based authentication.
- **Travel Details**: Users can input their travel method, flight numbers, and arrival times.
- **Dietary Requirements**: Attendees can specify food preferences and allergies.
- **Session Attendance**: Users can register for specific technical sessions.
- **Expense Tracking**: Attendees can log their expenses.
- **VIP Flight Restriction**: Ensures no more than one VIP from each region (US, Europe, Asia) is on the same flight.
- **Conference Planning Dashboard**: Provides planners with access to all stored information.

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- PostgreSQL (recommended) or SQLite (for development)
- A virtual environment tool (optional but recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to configure your database and email settings:
     ```ini
     DATABASE_URL=postgresql://user:password@localhost/dbname
     SECRET_KEY=your_secret_key_here
     ALGORITHM=HS256
     ACCESS_TOKEN_EXPIRE_MINUTES=30
     SMTP_SERVER=smtp.example.com
     SMTP_PORT=587
     SMTP_USERNAME=your_email@example.com
     SMTP_PASSWORD=your_email_password
     ```

## Database Setup

1. **Apply Migrations** (if using PostgreSQL or SQLite):
   ```bash
   alembic upgrade head
   ```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage
### User Registration & Authentication
- Register via `/register`.
- Authenticate via `/token` to receive a JWT.
- Use JWT for accessing protected endpoints.

### Conference Features
- **Submit Travel Details**: Users provide flight numbers, arrival times, and travel methods.
- **Dietary Preferences**: Specify food restrictions and allergies.
- **Session Registration**: Select from available technical sessions.
- **Expense Logging**: Enter and track conference-related expenses.
- **VIP Flight Restriction**: System prevents multiple VIPs from the same region on the same flight.
- **Admin Dashboard**: Planners access and manage conference data.

## Security Considerations
- **JWT-Based Authentication**: Ensures secure access.
- **Role-Based Access Control (RBAC)**: Limits planner access to administrative data.
- **Encryption & Hashing**: User data is securely stored.
- **Input Validation & Sanitization**: Prevents security vulnerabilities.

## Logging
Logs are stored in `app.log`. Modify logging settings in the script as needed.

## Troubleshooting
- Ensure your `.env` file is correctly configured.
- If using PostgreSQL, verify that the database service is running.
- Check logs for errors: `tail -f app.log`.

## Contributing
Feel free to fork the repository, create a branch, and submit a pull request with improvements.

## License
This project is licensed under the MIT License.

