from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi_mcp import FastApiMCP
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
from datetime import date

load_dotenv()

app = FastAPI()

# DATABASE CONNECTION SETUP
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "Denisons")
ROOMS_COLLECTION = "Rooms"
BOOKINGS_COLLECTION = "Bookings"

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi('1'))

db = client[DATABASE_NAME]
rooms_collection = db[ROOMS_COLLECTION]
bookings_collection = db[BOOKINGS_COLLECTION]


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

    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None


# UPDATED BOOKING REQUEST MODEL
class BookingRequest(BaseModel):
    check_in_date: str
    check_out_date: str
    user_name: str       # 👈 NEW
    email: str           # 👈 NEW


# UPDATED BOOKING MODEL
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


@app.on_event("startup")
async def startup_db():
    try:
        await client.admin.command("ping")
        await rooms_collection.create_index("id", unique=True)
        await bookings_collection.create_index("booking_id", unique=True)
        print("🚀 Connected to database successfully!")
    except Exception as e:
        print("❌ Could not connect:", str(e))


@app.on_event("shutdown")
async def shutdown_db():
    client.close()


# GET ROOMS
@app.get("/rooms", response_model=List[Room], operation_id="get_all_rooms",description="Display all available rooms along with their full details such as name, guest capacity, amenities, pricing, availability, refund policy, and check-in/check-out dates so the user can compare rooms and choose a suitable one as per their requirement")
async def get_all_rooms():
    rooms = []
    cursor = rooms_collection.find({"availability.available_rooms": {"$gt": 0}})

    async for room in cursor:
        room.pop("_id", None)
        rooms.append(Room(**room))

    return rooms


# BOOK ROOM
@app.post("/rooms/{room_id}/book", response_model=BookingResponse, operation_id="book_room")
async def book_room(room_id: int, booking_data: BookingRequest = Body(...)):
    room = await rooms_collection.find_one({"id": room_id})

    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    current_availability = room["availability"]["available_rooms"]

    if current_availability <= 0:
        raise HTTPException(status_code=400, detail="Room is fully booked")

    await rooms_collection.update_one(
        {"id": room_id},
        {"$set": {"availability.available_rooms": current_availability - 1}}
    )

    last_booking = await bookings_collection.find_one(sort=[("booking_id", -1)])
    new_booking_id = (last_booking["booking_id"] + 1) if last_booking else 1

    today = date.today().strftime("%Y-%m-%d")

    # STORE NEW USER VALUES
    booking_record = {
        "booking_id": new_booking_id,
        "room_id": room_id,
        "room_name": room["name"],
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


# GET BOOKINGS
@app.get("/bookings", response_model=List[Booking], operation_id="get_all_bookings")
async def get_all_bookings():
    bookings = []
    cursor = bookings_collection.find()

    async for booking in cursor:
        booking.pop("_id", None)
        bookings.append(Booking(**booking))

    return bookings


# GET BOOKING BY ID
@app.get("/bookings/{booking_id}", response_model=Booking, operation_id="get_booking_by_id")
async def get_booking(booking_id: int):
    booking = await bookings_collection.find_one({"booking_id": booking_id})

    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking.pop("_id", None)
    return Booking(**booking)


# CANCEL BOOKING
@app.delete("/bookings/{booking_id}/cancel", operation_id="cancel_booking")
async def cancel_booking(booking_id: int):
    booking = await bookings_collection.find_one({"booking_id": booking_id})

    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if booking["status"] == "cancelled":
        raise HTTPException(status_code=400, detail="Already cancelled")

    room_id = booking["room_id"]
    room = await rooms_collection.find_one({"id": room_id})

    await rooms_collection.update_one(
        {"id": room_id},
        {"$set": {"availability.available_rooms": room["availability"]["available_rooms"] + 1}}
    )

    await bookings_collection.update_one(
        {"booking_id": booking_id},
        {"$set": {"status": "cancelled"}}
    )

    return {"message": "Booking cancelled successfully!"}


# MCP CONFIG
mcp = FastApiMCP(
    app,
    include_operations=[
        "get_all_rooms",
        "book_room",
        "get_all_bookings",
        "get_booking_by_id",
        "cancel_booking"
    ]
)

mcp.mount()


def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



if __name__ == "__main__":
    main()
