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
    total_price: float       # <-- ADD THIS BACK
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
    try:
        checkin_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(check_out_date, "%Y-%m-%d")
    except:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    if checkout_dt <= checkin_dt:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    today = datetime.utcnow().date()
    max_date = today + timedelta(days=30)

    if checkin_dt.date() < today or checkout_dt.date() < today:
        raise HTTPException(status_code=400, detail="Dates cannot be in the past")

    if checkin_dt.date() > max_date or checkout_dt.date() > max_date:
        raise HTTPException(status_code=400, detail="Dates cannot be more than 30 days ahead")

    clean_rooms: List[RoomType] = []
    cursor = room_types_collection.find()

    async for room in cursor:
        if adults is not None and room["capacity"]["adults"] < adults:
            continue
        if children is not None and room["capacity"]["children"] < children:
            continue

        available_room_numbers: List[RoomNumber] = []

        for rn in room.get("room_numbers", []):
            room_no = rn["room_no"]

            # Fetch all confirmed bookings for that room
            bookings_cursor = bookings_collection.find({
                "room_no": room_no,
                "status": "confirmed"
            })

            is_available = True

            async for b in bookings_cursor:
                # Convert stored date strings to datetime objects
                db_checkin = datetime.strptime(b["check_in_date"], "%Y-%m-%d")
                db_checkout = datetime.strptime(b["check_out_date"], "%Y-%m-%d")

                # Overlap check in Python (correct comparison)
                if db_checkin < checkout_dt and db_checkout > checkin_dt:
                    is_available = False
                    break

            if is_available:
                available_room_numbers.append(RoomNumber(room_no=room_no))

        if available_room_numbers:
            room.pop("_id", None)
            room["room_numbers"] = [r.model_dump() for r in available_room_numbers]
            clean_rooms.append(RoomType(**room))

    return clean_rooms

# ==================== BOOK ROOM ====================
@app.post("/bookings", response_model=BookingResponse)
async def make_booking(data: BookingRequest):

    # Parse dates
    try:
        checkin_dt = datetime.strptime(data.check_in_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(data.check_out_date, "%Y-%m-%d")
    except:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    today = datetime.utcnow().date()
    max_date = today + timedelta(days=30)

    if checkin_dt.date() < today or checkout_dt.date() < today:
        raise HTTPException(status_code=400, detail="Dates cannot be in the past")

    if checkin_dt.date() > max_date or checkout_dt.date() > max_date:
        raise HTTPException(status_code=400, detail="Dates cannot be more than 30 days ahead")

    if checkout_dt <= checkin_dt:
        raise HTTPException(status_code=400, detail="check_out_date must be after check_in_date")

    room_data = await room_types_collection.find_one({"id": data.room_type_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room type not found")

    stay_days = (checkout_dt - checkin_dt).days

    if stay_days < room_data["min_days"] or stay_days > room_data["max_days"]:
        raise HTTPException(
            status_code=400,
            detail=f"You must stay between {room_data['min_days']} - {room_data['max_days']} days"
        )

    selected_room = None

    for rn in room_data["room_numbers"]:
        room_no = rn["room_no"]

        # Fetch all bookings for that room
        bookings_cursor = bookings_collection.find({
            "room_no": room_no,
            "status": "confirmed"
        })

        is_available = True

        async for b in bookings_cursor:
            db_checkin = datetime.strptime(b["check_in_date"], "%Y-%m-%d")
            db_checkout = datetime.strptime(b["check_out_date"], "%Y-%m-%d")

            # Overlap check
            if db_checkin < checkout_dt and db_checkout > checkin_dt:
                is_available = False
                break

        if is_available:
            selected_room = room_no
            break

    if not selected_room:
        raise HTTPException(status_code=400, detail="No rooms available")

    last = await bookings_collection.find_one(sort=[("booking_id", -1)])
    new_id = 1 if not last else last["booking_id"] + 1

    night_price = room_data["pricing"]["total_price"]
    total_price = night_price * stay_days

    # Store as STRING (your requirement)
    record = {
        "booking_id": new_id,
        "room_type_id": room_data["id"],
        "room_name": room_data["name"],
        "room_no": selected_room,
        "check_in_date": data.check_in_date,   # store string
        "check_out_date": data.check_out_date, # store string
        "stay_days": stay_days,
        "user_name": data.user_name,
        "email": data.email,
        "status": "confirmed",
        "created_at": datetime.utcnow().isoformat(),
        "total_price": total_price
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

    return {
        "name": room["name"],
        "image_url": room.get("image_url", "")
    }


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


@app.put("/room-types/{room_type_id}", response_model=RoomType)
async def update_room_type(room_type_id: int, updated: RoomType):
    existing = await room_types_collection.find_one({"id": room_type_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Room type not found")

    await room_types_collection.update_one(
        {"id": room_type_id},
        {"$set": updated.model_dump()}
    )

    return updated

@app.delete("/room-types/{room_type_id}")
async def delete_room_type(room_type_id: int):
    existing = await room_types_collection.find_one({"id": room_type_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Room type not found")

    await room_types_collection.delete_one({"id": room_type_id})

    return {"message": "Room type deleted successfully"}



def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
