from fastapi import FastAPI, HTTPException, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import jwt

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


# ==================== JWT CONFIG ====================
SECRET_KEY = "MY_SECRET_KEY"
ALGORITHM = "HS256"

DUMMY_USER = {
    "username": "admin",
    "password": "password123"
}

def create_token(username: str):
    expiration = datetime.utcnow() + timedelta(hours=5)
    return jwt.encode({"sub": username, "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)


# ==================== DATABASE CONFIG ====================
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "denisons_beach_resort"
ROOM_TYPES_COLLECTION = "RoomTypes"
BOOKINGS_COLLECTION = "Bookings"

if not MONGO_URI:
    raise ValueError("Missing MONGO_URI env variable")

client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi("1"))
db = client[DATABASE_NAME]

room_types_collection = db[ROOM_TYPES_COLLECTION]
bookings_collection = db[BOOKINGS_COLLECTION]


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
    email: str


class Booking(BaseModel):
    booking_id: int
    room_type_id: int
    room_name: str
    room_no: int
    check_in_date: str
    check_out_date: str
    stay_days: int
    user_name: str
    email: str
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
    print("Database connected 🔥")


@app.on_event("shutdown")
async def stop_db():
    client.close()





# ==================== UTILS ====================
def parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")


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
    check_in_date: str,
    check_out_date: str,
    adults: Optional[int] = None,
    children: Optional[int] = None,
):
    # basic sanity check
    checkin_dt = parse_date(check_in_date)
    checkout_dt = parse_date(check_out_date)
    if checkout_dt <= checkin_dt:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    clean_rooms: List[RoomType] = []
    cursor = room_types_collection.find()

    async for room in cursor:
        # capacity filter (use is not None so 0 is allowed)
        if adults is not None and room["capacity"]["adults"] < adults:
            continue
        if children is not None and room["capacity"]["children"] < children:
            continue

        available_room_numbers: List[RoomNumber] = []

        for rn in room.get("room_numbers", []):
            room_no = rn["room_no"]

            # check if this room_no is already booked in that range
            overlap = await bookings_collection.find_one(
                {
                    "room_no": room_no,
                    "status": "confirmed",
                    "check_in_date": {"$lt": check_out_date},
                    "check_out_date": {"$gt": check_in_date},
                }
            )

            # only keep room numbers with NO overlap
            if not overlap:
                available_room_numbers.append(RoomNumber(room_no=room_no))

        # only include room types that have at least 1 free room number
        if available_room_numbers:
            room.pop("_id", None)
            room["room_numbers"] = [r.model_dump() for r in available_room_numbers]
            clean_rooms.append(RoomType(**room))

    return clean_rooms


# ==================== BOOK ROOM ====================
@app.post("/bookings", response_model=BookingResponse)
async def make_booking(data: BookingRequest):

    room_data = await room_types_collection.find_one({"id": data.room_type_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room type not found")

    checkin_dt = parse_date(data.check_in_date)
    checkout_dt = parse_date(data.check_out_date)

    stay_days = (checkout_dt - checkin_dt).days
    if stay_days <= 0:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    if stay_days < room_data["min_days"] or stay_days > room_data["max_days"]:
        raise HTTPException(
            status_code=400,
            detail=f"You must stay between {room_data['min_days']} - {room_data['max_days']} days"
        )

    selected_room = None

    for rn in room_data["room_numbers"]:
        room_no = rn["room_no"]

        overlap = await bookings_collection.find_one(
            {
                "room_no": room_no,
                "status": "confirmed",
                "check_in_date": {"$lt": data.check_out_date},
                "check_out_date": {"$gt": data.check_in_date},
            }
        )

        if not overlap:
            selected_room = room_no
            break

    if not selected_room:
        raise HTTPException(status_code=400, detail="No rooms available")

    last = await bookings_collection.find_one(sort=[("booking_id", -1)])
    new_id = 1 if not last else last["booking_id"] + 1

    record = {
        "booking_id": new_id,
        "room_type_id": room_data["id"],
        "room_name": room_data["name"],
        "room_no": selected_room,
        "check_in_date": data.check_in_date,
        "check_out_date": data.check_out_date,
        "stay_days": stay_days,
        "user_name": data.user_name,
        "email": data.email,
        "status": "confirmed",
        "created_at": datetime.utcnow().isoformat()
    }

    await bookings_collection.insert_one(record)
    record.pop("_id", None)

    return {"message": "Room Booked Successfully!", "booking": Booking(**record)}


@app.get("/bookings", response_model=List[Booking])
async def all_bookings():
    bookings = []
    cursor = bookings_collection.find()

    async for item in cursor:
        item.pop("_id", None)
        bookings.append(Booking(**item))

    return bookings


@app.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: int):
    booking = await bookings_collection.find_one({"booking_id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    await bookings_collection.update_one(
        {"booking_id": booking_id},
        {"$set": {"status": "cancelled"}}
    )

    return {"message": "Booking cancelled successfully"}


# ==================== MCP ENABLE ====================
mcp = FastApiMCP(app)
mcp.mount_http()


# ==================== LOGIN ====================
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != DUMMY_USER["username"] or form_data.password != DUMMY_USER["password"]:
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
    """
    Create a new room type along with room numbers and pricing.
    """
    # Check if room type already exists
    existing = await room_types_collection.find_one({"id": room_data.id})
    if existing:
        raise HTTPException(status_code=400, detail="Room type with this ID already exists")

    # Insert into database
    await room_types_collection.insert_one(room_data.model_dump())

    return room_data


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
