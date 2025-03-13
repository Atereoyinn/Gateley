from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, constr, field_validator
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, Float, Enum, Index
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum as PyEnum
import re
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Constants and Configurations ---
class UserRole(str, PyEnum):
    ATTENDEE = "attendee"
    SECRETARY = "secretary"
    PLANNER = "planner"

class Region(str, PyEnum):
    US = "US"
    EUROPE = "Europe"
    ASIA = "Asia"

class ExpenseCategory(str, PyEnum):
    TRAVEL = "travel"
    ACCOMMODATION = "accommodation"
    FOOD = "food"
    OTHER = "other"

app = FastAPI(title="Conference Attendance System")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./conference.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True)
    company = Column(String(100), nullable=False)
    password = Column(String(200))
    role = Column(Enum(UserRole), nullable=False)
    region = Column(Enum(Region), nullable=False)
    is_vip = Column(Boolean, default=False)
    
    travel = relationship("TravelDetails", back_populates="user", cascade="all, delete")
    dietary = relationship("DietaryRequirement", back_populates="user", cascade="all, delete")
    sessions = relationship("SessionAttendance", back_populates="user", cascade="all, delete")
    expenses = relationship("Expense", back_populates="user", cascade="all, delete")

    __table_args__ = (
        Index("ix_users_company", "company"),
        Index("ix_users_region", "region"),
    )

class TravelDetails(Base):
    __tablename__ = "travel_details"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    method = Column(String(50), nullable=False)
    flight_number = Column(String(20))
    arrival = Column(DateTime, nullable=False)
    user = relationship("User", back_populates="travel")

class DietaryRequirement(Base):
    __tablename__ = "dietary_requirements"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    requirement = Column(String(100), nullable=False)
    user = relationship("User", back_populates="dietary")


class ConferenceSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    capacity = Column(Integer)
    attendees = relationship("SessionAttendance", back_populates="session")

class SessionAttendance(Base):
    __tablename__ = "session_attendance"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"))
    user = relationship("User", back_populates="sessions")
    session = relationship("ConferenceSession", back_populates="attendees")


class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float, nullable=False)
    category = Column(Enum(ExpenseCategory), nullable=False)
    description = Column(String(200))
    user = relationship("User", back_populates="expenses")

Base.metadata.create_all(bind=engine)

# --- Pydantic Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    full_name: constr(min_length=2, max_length=100)
    email: EmailStr
    company: constr(min_length=2, max_length=100)
    region: Region

class UserCreate(UserBase):
    password: constr(min_length=8)
    role: UserRole = UserRole.ATTENDEE
    is_vip: bool = False

    @field_validator("is_vip")
    def validate_vip(cls, v, values):
        if v and values.get("role") != UserRole.ATTENDEE:
            raise ValueError("VIP status only applicable for attendees")
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    role: UserRole
    is_vip: bool

    class Config:
        orm_mode = True

class TravelCreate(BaseModel):
    method: str
    flight_number: Optional[str] = None
    arrival: datetime

    @field_validator("flight_number")
    def validate_flight_number(cls, v):
        if v and not re.match(r"^[A-Z]{2}\d{3,4}$", v):
            raise ValueError("Invalid flight number format (e.g., AA1234)")
        return v

    @field_validator("arrival")
    def validate_arrival(cls, v):
        if v < datetime.utcnow() + timedelta(hours=2):
            raise ValueError("Arrival time must be at least 2 hours from now")
        return v


class SessionCreate(BaseModel):
    name: str
    start_time: datetime
    capacity: int

class DietaryCreate(BaseModel):
    requirement: str

class ExpenseCreate(BaseModel):
    amount: float
    category: ExpenseCategory
    description: Optional[str] = None

    @field_validator("amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 10000:
            raise ValueError("Amount exceeds maximum allowed $10,000")
        return v

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    company: str
    role: UserRole
    region: Region
    is_vip: bool

    class Config:
        orm_mode = True

class SessionResponse(BaseModel):
    id: int
    name: str
    start_time: datetime
    capacity: int
    attendees: List[UserResponse]

    class Config:
        orm_mode = True

class TravelDetailsResponse(BaseModel):
    id: int
    user_id: int
    method: str
    flight_number: Optional[str]
    arrival: datetime

    class Config:
        orm_mode = True  # Enable ORM mode for SQLAlchemy compatibility

class DietaryRequirementResponse(BaseModel):
    id: int
    user_id: int
    requirement: str

    class Config:
        orm_mode = True

class SessionAttendanceResponse(BaseModel):
    id: int
    user_id: int
    session_id: int

    class Config:
        orm_mode = True

class ExpenseResponse(BaseModel):
    id: int
    user_id: int
    amount: float
    category: str
    description: Optional[str]

    class Config:
        orm_mode = True

# --- Helper Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def send_welcome_email(email: str, name: str, company: str):
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = email
        msg["Subject"] = "Conference Registration Confirmation"
        
        body = f"""
        <h2>Welcome to the Annual Conference, {name}!</h2>
        <p><strong>Company:</strong> {company}</p>
        <p>Your registration has been successfully processed.</p>
        <h3>Next Steps:</h3>
        <ol>
            <li>Complete your travel details</li>
            <li>Submit dietary requirements</li>
            <li>Select conference sessions</li>
        </ol>
        <p>Access your portal: <a href="https://conference.example.com">Login Here</a></p>
        """
        
        msg.attach(MIMEText(body, "html"))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            
    except Exception as e:
        logging.error(f"Email sending failed: {str(e)}")

def verify_vip_flight_constraint(db: Session, user_id: int, flight: str):
    if not flight: return
    
    user = db.get(User, user_id)
    if not (user and user.is_vip):
        return

    conflict = db.query(User.id).join(TravelDetails).filter(
        User.region == user.region,
        User.is_vip == True,
        TravelDetails.flight_number == flight,
        User.id != user_id
    ).first()
    
    if conflict:
        raise HTTPException(
            status_code=400,
            detail=f"Another VIP from {user.region} is on flight {flight}"
        )

# --- Authentication Endpoints ---
@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Validate VIP uniqueness
    if user.is_vip:
        existing_vip = db.query(User).filter(
            User.region == user.region,
            User.is_vip == True
        ).first()
        if existing_vip:
            raise HTTPException(400, detail="Only one VIP allowed per region")

    hashed_pw = pwd_context.hash(user.password)
    db_user = User(
        full_name=user.full_name,
        email=user.email,
        company=user.company,
        password=hashed_pw,
        role=user.role,
        region=user.region,
        is_vip=user.is_vip
    )
    
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        send_welcome_email(user.email, user.full_name, user.company)
    except Exception as e:
        db.rollback()
        raise HTTPException(400, detail="Registration failed: Email exists")
    
    return db_user


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = jwt.encode(
        {"sub": db_user.email, "exp": datetime.utcnow() + access_token_expires},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"access_token": access_token, "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


@app.get("/user/me", response_model=UserResponse)
def get_current_user_details(current_user: User = Depends(get_current_user)):
    return current_user

# --- Attendance Endpoints ---
@app.post("/travel")
def add_travel(
    data: TravelCreate,
    target_user_id: Optional[int] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user.role == UserRole.SECRETARY:
        if not target_user_id:
            raise HTTPException(400, detail="Must specify user ID")
        target = db.get(User, target_user_id)
        if not target or target.company != user.company:
            raise HTTPException(403, detail="Unauthorized user access")
    elif user.role == UserRole.ATTENDEE:
        target_user_id = user.id
    else:
        raise HTTPException(403, detail="Unauthorized action")

    verify_vip_flight_constraint(db, target_user_id, data.flight_number)
    
    travel = TravelDetails(
        method=data.method,
        flight_number=data.flight_number,
        arrival=data.arrival,
        user_id=target_user_id
    )
    db.add(travel)
    db.commit()
    return {"message": "Travel details saved"}

@app.post("/dietary")
def add_dietary(
    data: DietaryCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dietary = DietaryRequirement(requirement=data.requirement, user_id=user.id)
    db.add(dietary)
    db.commit()
    return {"message": "Dietary requirement saved"}

@app.post("/expenses")
def add_expense(
    data: ExpenseCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expense = Expense(
        amount=data.amount,
        category=data.category,
        description=data.description,
        user_id=user.id
    )
    db.add(expense)
    db.commit()
    return {"message": "Expense recorded"}

# --- Session Management ---
@app.post("/sessions")
def create_session(
    data: SessionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user.role != UserRole.PLANNER:
        raise HTTPException(403, detail="Planner access required")
    
    if db.query(ConferenceSession).filter(ConferenceSession.name == data.name).first():
        raise HTTPException(400, detail="Session exists")
    
    session = ConferenceSession(
        name=data.name,
        start_time=data.start_time,
        capacity=data.capacity
    )
    db.add(session)
    db.commit()
    return {"id": session.id}



@app.post("/sessions/register/{session_id}")
def register_for_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.get(ConferenceSession, session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    
    if datetime.utcnow() > session.start_time:
        raise HTTPException(400, detail="Session expired")
    
    if len(session.attendees) >= session.capacity:
        raise HTTPException(400, detail="Session full")
    
    existing = db.query(SessionAttendance).filter(
        SessionAttendance.user_id == user.id,
        SessionAttendance.session_id == session_id
    ).first()
    if existing:
        raise HTTPException(400, detail="Already registered")
    
    attendance = SessionAttendance(user_id=user.id, session_id=session_id)
    db.add(attendance)
    db.commit()
    return {"message": "Registration successful"}

@app.get("/users/{user_id}/travel-details", response_model=TravelDetailsResponse)
def get_travel_details(user_id: int, db: Session = Depends(get_db)):
    travel_details = db.query(TravelDetails).filter(TravelDetails.user_id == user_id).first()
    if not travel_details:
        raise HTTPException(status_code=404, detail="Travel details not found")
    return travel_details

@app.get("/users/{user_id}/dietary-requirements", response_model=List[DietaryRequirementResponse])
def get_dietary_requirements(user_id: int, db: Session = Depends(get_db)):
    dietary_requirements = db.query(DietaryRequirement).filter(DietaryRequirement.user_id == user_id).all()
    return dietary_requirements

@app.get("/users/{user_id}/session-attendance", response_model=List[SessionAttendanceResponse])
def get_session_attendance(user_id: int, db: Session = Depends(get_db)):
    session_attendance = db.query(SessionAttendance).filter(SessionAttendance.user_id == user_id).all()
    return session_attendance

@app.get("/users/{user_id}/expenses", response_model=List[ExpenseResponse])
def get_expenses(user_id: int, db: Session = Depends(get_db)):
    expenses = db.query(Expense).filter(Expense.user_id == user_id).all()
    return expenses

@app.get("/attendees", response_model=List[UserResponse])
def get_attendees(
    region: Optional[Region] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user.role != UserRole.PLANNER:
        raise HTTPException(403, detail="Only planners can view attendees")
    
    query = db.query(User).filter(User.role != UserRole.PLANNER)
    if region:
        query = query.filter(User.region == region)
    
    return query.all()

@app.get("/sessions", response_model=List[SessionResponse])
def get_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(ConferenceSession).all()

@app.get("/planner/attendees", response_model=List[UserResponse])
def get_attendees(
    region: Optional[Region] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user.role != UserRole.PLANNER:
        raise HTTPException(403, detail="Access denied")
    
    query = db.query(User).filter(User.role != UserRole.PLANNER)
    if region:
        query = query.filter(User.region == region)
    
    return query.all()

@app.get("/planner/dashboard")
def planner_dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.role != UserRole.PLANNER:
        raise HTTPException(403, detail="Access denied")
    
    stats = {
        "total_attendees": db.query(User).filter(User.role != UserRole.PLANNER).count(),
        "vip_count": db.query(User).filter(User.is_vip == True).count(),
        "sessions": [
            {
                "name": s.name,
                "start_time": s.start_time,
                "registered": len(s.attendees),
                "capacity": s.capacity
            }
            for s in db.query(ConferenceSession).all()
        ]
    }
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)