from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi_mcp import FastApiMCP
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
from datetime import date
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB SETUP
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "Denisons")
ROOMS_COLLECTION = "Rooms"
BOOKINGS_COLLECTION = "Bookings"

client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi('1'))
db = client[DATABASE_NAME]
rooms_collection = db[ROOMS_COLLECTION]
bookings_collection = db[BOOKINGS_COLLECTION]


# ==================== MODELS ========================

class RoomSize(BaseModel):
    area: float
    unit: str


class Availability(BaseModel):
    available_rooms: int
    status: str


class Pricing(BaseModel):
    base_price: float
    currency: str
    tax_price: float
    total_price: float
    pricing_type: str


class Room(BaseModel):
    id: int
    name: str
    adults: int
    children: int
    guests: int
    description: Optional[str] = ""
    size: RoomSize
    amenities: List[str]
    availability: Availability
    pricing: Pricing
    package_name: Optional[str] = None
    refund_policy: Optional[str] = None
    banner_image: Optional[str] = ""

    image_url: Optional[str] = ""   # ⭐ NEW CODE

    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None


class BookingRequest(BaseModel):
    check_in_date: str
    check_out_date: str
    user_name: str
    email: str


class Booking(BaseModel):
    booking_id: int
    room_id: int
    room_name: str
    total_price: float
    currency: str
    booking_date: str
    status: str
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    user_name: str
    email: str


class BookingResponse(BaseModel):
    message: str
    booking: Booking


# ==================== EVENTS ========================

@app.on_event("startup")
async def startup_db():
    await rooms_collection.create_index("id", unique=True)
    await bookings_collection.create_index("booking_id", unique=True)
    print("Database connected")


@app.on_event("shutdown")
async def shutdown_db():
    client.close()


# ==================== APIS ========================

# GET ALL ROOMS
@app.get("/rooms", response_model=List[Room])
async def get_all_rooms():
    rooms = []
    cursor = rooms_collection.find({"availability.available_rooms": {"$gt": 0}})

    async for room in cursor:
        room.pop("_id", None)
        rooms.append(Room(**room))

    return rooms


# ⭐ NEW CODE ─ Filter Rooms API
@app.get("/rooms/filter", response_model=List[Room])
async def filter_rooms(
    check_in_date: str,
    check_out_date: str,
    adults: int,
    children: int
):
    """
    Filters rooms based on EXACT matching of DB stored date values
    and guest capacity.
    """
    query = {
        "availability.available_rooms": {"$gt": 0},
        "adults": {"$gte": adults},
        "children": {"$gte": children},
        "check_in_date": check_in_date,
        "check_out_date": check_out_date
    }

    rooms = []
    cursor = rooms_collection.find(query)

    async for room in cursor:
        room.pop("_id", None)
        rooms.append(Room(**room))

    return rooms


# BOOK ROOM
# BOOK ROOM USING NAME
@app.post("/rooms/{room_name}/book", response_model=BookingResponse)
async def book_room(room_name: str, booking_data: BookingRequest = Body(...)):
    # Find room based on name (case-insensitive)
    room = await rooms_collection.find_one({
        "name": {"$regex": f"^{room_name}$", "$options": "i"}
    })

    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    current_availability = room["availability"]["available_rooms"]

    if current_availability <= 0:
        raise HTTPException(status_code=400, detail="Room not available")

    # Reduce room availability
    await rooms_collection.update_one(
        {"id": room["id"]},
        {"$set": {"availability.available_rooms": current_availability - 1}}
    )

    # Booking ID logic
    last_booking = await bookings_collection.find_one(sort=[("booking_id", -1)])
    new_booking_id = (last_booking["booking_id"] + 1) if last_booking else 1

    today = date.today().strftime("%Y-%m-%d")

    booking_record = {
        "booking_id": new_booking_id,
        "room_id": room["id"],      # keep id internally
        "room_name": room["name"],  # using name now
        "total_price": room["pricing"]["total_price"],
        "currency": room["pricing"]["currency"],
        "booking_date": today,
        "status": "confirmed",
        "check_in_date": booking_data.check_in_date,
        "check_out_date": booking_data.check_out_date,
        "user_name": booking_data.user_name,
        "email": booking_data.email
    }

    await bookings_collection.insert_one(booking_record)
    booking_record.pop("_id", None)

    return {
        "message": "Room booked successfully!",
        "booking": Booking(**booking_record)
    }


# CANCEL BOOKING
@app.delete("/bookings/{booking_id}/cancel")
async def cancel_booking(booking_id: int):
    booking = await bookings_collection.find_one({"booking_id": booking_id})

    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if booking["status"] == "cancelled":
        raise HTTPException(status_code=400, detail="Already cancelled")

    room = await rooms_collection.find_one({"id": booking["room_id"]})

    await rooms_collection.update_one(
        {"id": booking["room_id"]},
        {"$set": {"availability.available_rooms": room["availability"]["available_rooms"] + 1}}
    )

    await bookings_collection.update_one(
        {"booking_id": booking_id},
        {"$set": {"status": "cancelled"}}
    )

    return {"message": "Booking cancelled successfully!"}

@app.get("/bookings", response_model=List[Booking])
async def get_all_bookings():
    bookings = []
    cursor = bookings_collection.find()

    async for booking in cursor:
        booking.pop("_id", None)
        bookings.append(Booking(**booking))

    return bookings

@app.get("/rooms/image")
async def get_image_by_name(name: str):
    room = await rooms_collection.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})

    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    image_url = room.get("image_url", "")

    return {
        "name": room["name"],
        "image_url": image_url
    }
    
# MCP CONFIG
mcp = FastApiMCP(app)
mcp.mount_http()


def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



if __name__ == "__main__":
    main()
