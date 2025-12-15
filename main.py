from fastapi import FastAPI, HTTPException, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import jwt
from passlib.context import CryptContext

from fastapi_mcp import FastApiMCP


load_dotenv()

# ==================== FASTAPI INSTANCE ====================
app = FastAPI(title="Hotel Booking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# ==================== JWT CONFIG ====================
SECRET_KEY = "MY_SECRET_KEY"
ALGORITHM = "HS256"



def create_token(username: str):
    expiration = datetime.utcnow() + timedelta(hours=5)
    return jwt.encode({"sub": username, "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)


# ==================== DATABASE CONFIG ====================
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "denisons_beach_resort"
ROOM_TYPES_COLLECTION = "RoomTypes"
BOOKINGS_COLLECTION = "Bookings"
ADMINS_COLLECTION = "Admins"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not MONGO_URI:
    raise ValueError("Missing MONGO_URI env variable")

client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi("1"))
db = client[DATABASE_NAME]

room_types_collection = db[ROOM_TYPES_COLLECTION]
bookings_collection = db[BOOKINGS_COLLECTION]
admins_collection = db[ADMINS_COLLECTION]  # ✅ Now db exists


# ==================== MODELS ====================
class Capacity(BaseModel):
    adults: int
    children: int


class Pricing(BaseModel):
    base_price: float
    tax_price: float
    total_price: float
    currency: str = "INR"
    pricing_type: str = "per night"


class RoomNumber(BaseModel):
    room_no: int


class RoomType(BaseModel):
    id: int
    name: str
    description: Optional[str] = ""
    capacity: Capacity
    amenities: List[str]
    min_days: int = Field(1)
    max_days: int = Field(30)
    pricing: Pricing
    room_numbers: List[RoomNumber]
    image_url: Optional[str] = ""
    banner_image: Optional[str] = ""
    refund_policy: Optional[str] = ""


class BookingRequest(BaseModel):
    room_type_id: int
    check_in_date: str
    check_out_date: str
    user_name: str
    email: EmailStr
    adults: int
    children: int


class Booking(BaseModel):
    booking_id: int
    room_type_id: int
    room_name: str
    room_no: int
    check_in_date: str
    check_out_date: str
    total_price: float
    stay_days: int
    user_name: str
    email: str
    adults: int
    children: int
    status: str
    created_at: str


class BookingResponse(BaseModel):
    message: str
    booking: Booking


# ==================== LIFECYCLE ====================
@app.on_event("startup")
async def start_db():
    await room_types_collection.create_index("id", unique=True)
    await bookings_collection.create_index("booking_id", unique=True)
    await admins_collection.create_index("username", unique=True)
    
    # Create default admin from env variables if not exists
    existing_admin = await admins_collection.find_one({"username": ADMIN_USERNAME})
    if not existing_admin:
        default_admin = {
            "username": ADMIN_USERNAME,
            "password": hash_password(ADMIN_PASSWORD),
            "created_at": datetime.utcnow().isoformat()
        }
        await admins_collection.insert_one(default_admin)
        print(f"✅ Default admin created (username: {ADMIN_USERNAME})")
    
    print("Database connected 🔥")

@app.on_event("shutdown")
async def stop_db():
    client.close()
    print("Database connection closed")


# ==================== UTIL FUNCTIONS ====================
def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")


async def is_room_type_available(
    room_data,
    checkin_dt: datetime,
    checkout_dt: datetime,
    adults: int,
    children: int,
) -> bool:
    """
    Returns True only if:
      - adults/children fit capacity
      - at least one room_no is free for the date range
    """

    # Basic validation
    if adults <= 0:
        return False
    if children < 0:
        return False

    # Capacity checks
    if adults > room_data["capacity"]["adults"]:
        return False
    if children > room_data["capacity"]["children"]:
        return False

    # Date availability: check any room_no in this room_type is free
    for rn in room_data.get("room_numbers", []):
        room_no = rn["room_no"]
        cursor = bookings_collection.find(
            {
                "room_no": room_no,
                "status": "confirmed",
            }
        )

        is_free = True
        async for b in cursor:
            try:
                db_ci = datetime.strptime(b["check_in_date"], "%Y-%m-%d")
                db_co = datetime.strptime(b["check_out_date"], "%Y-%m-%d")
            except Exception:
                continue

            # overlap condition
            if db_ci < checkout_dt and db_co >= checkin_dt:
                is_free = False
                break

        if is_free:
            return True

    return False


# ==================== CRUD ROOM TYPES ====================


@app.get("/room/{room_type_id}", response_model=RoomType)
async def get_room(room_type_id: int):
    data = await room_types_collection.find_one({"id": room_type_id})
    if not data:
        raise HTTPException(status_code=404, detail="Room Type Not Found")
    data.pop("_id", None)
    return RoomType(**data)


# ==================== SEARCH ROOM TYPES BY DATE ====================
@app.get("/room-types/available", response_model=List[RoomType])
async def available_rooms(
    check_in_date: str = Query(..., description="Check-in date in YYYY-MM-DD format"),
    check_out_date: str = Query(..., description="Check-out date in YYYY-MM-DD format"),
    adults: int = Query(..., ge=1, description="Number of adults"),
    children: int = Query(..., ge=0, description="Number of children"),
):
    # Parse & validate dates
    try:
        checkin_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(check_out_date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    today = datetime.utcnow().date()
    max_date = today + timedelta(days=30)

    if checkin_dt.date() < today or checkout_dt.date() < today:
        raise HTTPException(status_code=400, detail="Dates cannot be in the past")

    if checkin_dt.date() > max_date or checkout_dt.date() > max_date:
        raise HTTPException(status_code=400, detail="Dates cannot be more than 30 days ahead")

    if checkout_dt <= checkin_dt:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    # Validate adults/children inputs
    if adults <= 0:
        raise HTTPException(status_code=400, detail="Adults must be at least 1")
    if children < 0:
        raise HTTPException(status_code=400, detail="Children count cannot be negative")

    clean_rooms: List[RoomType] = []
    cursor = room_types_collection.find()

    async for room in cursor:
        # Capacity filter
        if room["capacity"]["adults"] < adults:
            continue
        if room["capacity"]["children"] < children:
            continue

        available_room_numbers: List[RoomNumber] = []

        for rn in room.get("room_numbers", []):
            room_no = rn["room_no"]

            # Keep only room numbers with no overlap in date range
            overlap = await bookings_collection.find_one(
                {
                    "room_no": room_no,
                    "status": "confirmed",
                    "check_in_date": {"$lt": check_out_date},
                    "check_out_date": {"$gte": check_in_date},
                }
            )

            if not overlap:
                available_room_numbers.append(RoomNumber(room_no=room_no))

        if available_room_numbers:
            room.pop("_id", None)
            room["room_numbers"] = [r.model_dump() for r in available_room_numbers]
            clean_rooms.append(RoomType(**room))

    return clean_rooms


@app.post("/bookings", response_model=BookingResponse)
async def make_booking(data: BookingRequest):
    # Basic log
    print("BOOKING REQUEST:", data.model_dump())

    # Parse dates
    try:
        checkin_dt = parse_date(data.check_in_date)
        checkout_dt = parse_date(data.check_out_date)
    except HTTPException:
        raise

    today = datetime.utcnow().date()
    max_date = today + timedelta(days=30)

    if checkin_dt.date() < today or checkout_dt.date() < today:
        raise HTTPException(status_code=400, detail="Dates cannot be in the past")
    if checkin_dt.date() > max_date or checkout_dt.date() > max_date:
        raise HTTPException(status_code=400, detail="Dates cannot be more than 30 days ahead")
    if checkout_dt <= checkin_dt:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    # Fetch room type once
    room_data = await room_types_collection.find_one({"id": data.room_type_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room type not found")

    stay_days = (checkout_dt - checkin_dt).days
    if stay_days < room_data.get("min_days", 1) or stay_days > room_data.get("max_days", 30):
        raise HTTPException(
            status_code=400,
            detail=f"You must stay between {room_data.get('min_days',1)} - {room_data.get('max_days',30)} days"
        )

    # Capacity validation
    if data.adults > room_data["capacity"]["adults"]:
        raise HTTPException(status_code=400, detail=f"Max adults allowed for this room: {room_data['capacity']['adults']}")
    if data.children > room_data["capacity"]["children"]:
        raise HTTPException(status_code=400, detail=f"Max children allowed for this room: {room_data['capacity']['children']}")

    # Use is_room_type_available to check both capacity and date availability
    available = await is_room_type_available(
        room_data,
        checkin_dt,
        checkout_dt,
        data.adults,
        data.children,
    )
    if not available:
        raise HTTPException(status_code=400, detail="This room type is not available for the selected dates or capacity")

    # Select a free room_no
    selected_room = None
    for rn in room_data.get("room_numbers", []):
        room_no = rn["room_no"]
        cursor = bookings_collection.find({
            "room_no": room_no,
            "status": "confirmed"
        })

        free = True
        async for b in cursor:
            try:
                db_checkin = datetime.strptime(b["check_in_date"], "%Y-%m-%d")
                db_checkout = datetime.strptime(b["check_out_date"], "%Y-%m-%d")
            except Exception:
                continue

            if db_checkin < checkout_dt and db_checkout >= checkin_dt:
                free = False
                break

        if free:
            selected_room = room_no
            break

    if not selected_room:
        raise HTTPException(status_code=400, detail="No rooms available")

    # Create booking ID
    last = await bookings_collection.find_one(sort=[("booking_id", -1)])
    new_id = 1 if not last else last["booking_id"] + 1

    night_price = room_data["pricing"]["total_price"]
    total_price = night_price * stay_days

    record = {
        "booking_id": new_id,
        "room_type_id": room_data["id"],
        "room_name": room_data["name"],
        "room_no": selected_room,
        "check_in_date": data.check_in_date,
        "check_out_date": data.check_out_date,
        "stay_days": stay_days,
        "user_name": data.user_name,
        "email": data.email.lower(),
        "adults": data.adults,
        "children": data.children,
        "status": "confirmed",
        "created_at": datetime.utcnow().isoformat(),
        "total_price": total_price
    }

    await bookings_collection.insert_one(record)
    record.pop("_id", None)

    print("BOOKING CREATED:", record["booking_id"], "ROOM_NO:", selected_room)
    return {"message": "Room Booked Successfully!", "booking": Booking(**record)}

@app.get("/bookings/email/{email}", response_model=List[Booking])
async def get_bookings_by_email(email: str):
    """
    Get all bookings for a specific email address
    """

    bookings = []
    cursor = bookings_collection.find({"email": email.lower()})
    async for item in cursor:
        item.pop("_id", None)
        bookings.append(Booking(**item))
    
    if not bookings:
        raise HTTPException(status_code=404, detail="No bookings found for this email")
    
    return bookings

@app.get("/room-types/{room_type_name}/image")
async def get_room_image(room_type_name: str):
    """
    Return the image URL of a room type by name (case-insensitive match)
    """
    room = await room_types_collection.find_one({
        "name": {"$regex": f"^{room_type_name}$", "$options": "i"}
    })
    if not room:
        raise HTTPException(status_code=404, detail="Room type not found")
    return {"name": room["name"], "image_url": room.get("image_url", "")}


@app.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: int):
    booking = await bookings_collection.find_one({"booking_id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    await bookings_collection.update_one({"booking_id": booking_id}, {"$set": {"status": "cancelled"}})
    print("BOOKING CANCELLED:", booking_id)
    return {"message": "Booking cancelled successfully"}


# ==================== MCP ENABLE ====================
mcp = FastApiMCP(app)
mcp.mount_http()


# ==================== LOGIN ====================
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    admin = await admins_collection.find_one({"username": form_data.username})
    
    if not admin or not verify_password(form_data.password, admin["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/room-types", response_model=RoomType)
async def add_room(room_data: RoomType):
    exists = await room_types_collection.find_one({"id": room_data.id})
    if exists:
        raise HTTPException(status_code=400, detail="Room ID already exists")
    await room_types_collection.insert_one(room_data.model_dump())
    return room_data


@app.get("/")
def root():
    return {"message": "Hotel Booking API Running 🏨"}


@app.get("/room-types", response_model=List[RoomType])
async def get_all_room_types():
    rooms = []
    cursor = room_types_collection.find()
    async for doc in cursor:
        doc.pop("_id", None)
        rooms.append(RoomType(**doc))
    return rooms


@app.post("/room-types", response_model=RoomType)
async def create_room_type(room_data: RoomType):
    existing = await room_types_collection.find_one({"id": room_data.id})
    if existing:
        raise HTTPException(status_code=400, detail="Room type with this ID already exists")
    await room_types_collection.insert_one(room_data.model_dump())
    return room_data


@app.put("/room-types/{room_type_id}", response_model=RoomType)
async def update_room_type(room_type_id: int, updated: RoomType):
    existing = await room_types_collection.find_one({"id": room_type_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Room type not found")
    await room_types_collection.update_one({"id": room_type_id}, {"$set": updated.model_dump()})
    return updated


@app.delete("/room-types/{room_type_id}")
async def delete_room_type(room_type_id: int):
    existing = await room_types_collection.find_one({"id": room_type_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Room type not found")
    await room_types_collection.delete_one({"id": room_type_id})
    return {"message": "Room type deleted successfully"}

@app.get("/bookings", response_model=List[Booking])
async def all_bookings():
    bookings = []
    cursor = bookings_collection.find()
    async for item in cursor:
        item.pop("_id", None)
        bookings.append(Booking(**item))
    return bookings

def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()