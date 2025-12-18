const API_BASE = "http://127.0.0.1:8000";

// State
let token = "";
let roomTypes = [];
let bookings = [];
let roomNumbers = [];
let isEditing = false;
let editingRoomId = null;
let currentCalendarDate = new Date();
let allRoomNumbers = []; // Store all room numbers from all room types
let selectedCalendarRoomType = "all";
let currentPage = 1;
let totalPages = 1;

// DOM Elements
const loginScreen = document.getElementById("loginScreen");
const dashboard = document.getElementById("dashboard");
const loginForm = document.getElementById("loginForm");
const loginError = document.getElementById("loginError");
const logoutBtn = document.getElementById("logoutBtn");

// Tabs
const navTabs = document.querySelectorAll(".nav-item");
const tabContents = document.querySelectorAll(".tab-content");

// Modal
const roomModal = document.getElementById("roomModal");
const addRoomBtn = document.getElementById("addRoomBtn");
const modalClose = document.querySelector(".modal-close");
const modalCancel = document.querySelector(".modal-cancel");
const roomTypeForm = document.getElementById("roomTypeForm");
const bookingModal = document.getElementById("bookingModal");
const addBookingBtn = document.getElementById("addBookingBtn");
const bookingForm = document.getElementById("bookingForm");
const bookingModalClose = document.querySelector(".booking-modal-close");
const bookingModalCancel = document.querySelector(".booking-modal-cancel");

// Search and Filter
const searchBookings = document.getElementById("searchBookings");
const filterStatus = document.getElementById("filterStatus");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  checkLoginStatus();
  setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
  loginForm.addEventListener("submit", handleLogin);
  logoutBtn.addEventListener("click", handleLogout);

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  addRoomBtn.addEventListener("click", () => {
    isEditing = false;
    editingRoomId = null;
    document.querySelector(".modal-header h3").textContent = "Create Room Type";
    document.querySelector(".modal-footer .btn-primary").textContent =
      "Create Room Type";
    roomModal.classList.add("active");
    resetRoomForm();
  });

  modalClose.addEventListener("click", () =>
    roomModal.classList.remove("active")
  );
  modalClose.addEventListener("click", () =>
    roomModal.classList.remove("active")
  );
  modalCancel.addEventListener("click", () =>
    roomModal.classList.remove("active")
  );
  document.addEventListener("input", (e) => {
    if (e.target.id !== "calendarRoomTypeInput") return;

    const query = e.target.value.toLowerCase();
    const list = document.getElementById("calendarRoomTypeList");
    list.style.display = "block";

    Array.from(list.children).forEach((item) => {
      item.style.display = item.textContent.toLowerCase().includes(query)
        ? "block"
        : "none";
    });
  });
  const calendarInput = document.getElementById("calendarRoomTypeInput");
  const clearBtn = document.getElementById("calendarRoomTypeClear");

  if (calendarInput && clearBtn) {
    calendarInput.addEventListener("input", () => {
      clearBtn.style.display = calendarInput.value ? "block" : "none";
    });

    clearBtn.addEventListener("click", () => {
      calendarInput.value = "";
      clearBtn.style.display = "none";
      selectedCalendarRoomType = "all";
      renderCalendar();
    });
  }
  const filterFromDate = document.getElementById("filterFromDate");
  const filterToDate = document.getElementById("filterToDate");

  if (filterFromDate)
    filterFromDate.addEventListener("change", filterBookingsTable);

  if (filterToDate)
    filterToDate.addEventListener("change", filterBookingsTable);

  document.addEventListener("click", (e) => {
    const wrapper = document.getElementById("calendarRoomTypeWrapper");
    const list = document.getElementById("calendarRoomTypeList");

    if (!wrapper.contains(e.target)) {
      list.style.display = "none";
    }
  });
  document.getElementById("prevPageBtn").addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      loadRoomTypes(currentPage);
    }
  });

  document.getElementById("nextPageBtn").addEventListener("click", () => {
    if (currentPage < totalPages) {
      currentPage++;
      loadRoomTypes(currentPage);
    }
  });

  // Booking Modal Listeners
  if (addBookingBtn) {
    addBookingBtn.addEventListener("click", () => {
      bookingModal.classList.add("active");
      populateBookingRoomTypes();
    });
  }

  if (bookingModalClose) {
    bookingModalClose.addEventListener("click", () =>
      bookingModal.classList.remove("active")
    );
  }

  if (bookingModalCancel) {
    bookingModalCancel.addEventListener("click", () =>
      bookingModal.classList.remove("active")
    );
  }

  if (bookingForm) {
    bookingForm.addEventListener("submit", handleCreateBooking);
  }

  const bookingDetailsModal = document.getElementById("bookingDetailsModal");
  const bookingDetailsCloseButtons = document.querySelectorAll(
    ".booking-details-close"
  );

  if (bookingDetailsCloseButtons) {
    bookingDetailsCloseButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        bookingDetailsModal.classList.remove("active");
      });
    });
  }

  roomTypeForm.addEventListener("submit", handleCreateRoomType);

  document
    .getElementById("addRoomNumber")
    .addEventListener("click", addRoomNumberField);
  document
    .getElementById("basePrice")
    .addEventListener("input", calculatePricing);

  searchBookings.addEventListener("input", filterBookingsTable);
  filterStatus.addEventListener("change", filterBookingsTable);

  // Calendar Navigation
  const prevMonthBtn = document.getElementById("prevMonth");
  const nextMonthBtn = document.getElementById("nextMonth");
  const todayBtn = document.getElementById("todayBtn");

  if (prevMonthBtn)
    prevMonthBtn.addEventListener("click", () => changeMonth(-1));
  if (nextMonthBtn)
    nextMonthBtn.addEventListener("click", () => changeMonth(1));
  if (todayBtn)
    todayBtn.addEventListener("click", () => {
      currentCalendarDate = new Date();
      updateYearSelect();
      renderCalendar();
    });

  // Year and month select change handlers
  const yearSelect = document.getElementById("yearSelect");
  const monthSelect = document.getElementById("monthSelect");

  if (yearSelect && monthSelect) {
    yearSelect.addEventListener("change", (e) => {
      changeYear(parseInt(e.target.value));
    });

    monthSelect.addEventListener("change", (e) => {
      changeMonth(parseInt(e.target.value) - currentCalendarDate.getMonth());
    });

    // Initialize selects
    updateYearSelect();
    updateMonthSelect();
  }

  // Handle children age inputs
  const bookingChildrenInput = document.getElementById("bookingChildren");
  if (bookingChildrenInput) {
    bookingChildrenInput.addEventListener("input", handleChildrenCountChange);
  }

  // Refresh Buttons
  const refreshRoomsBtn = document.getElementById("refreshRoomsBtn");
  if (refreshRoomsBtn) {
    refreshRoomsBtn.addEventListener("click", async () => {
      const icon = refreshRoomsBtn.querySelector("svg");
      icon.style.animation = "spin 1s linear infinite";
      refreshRoomsBtn.disabled = true;

      try {
        await loadRoomTypes();
      } finally {
        // Ensure a minimum spin time or just stop when done, here we stop when done but strictly remove animation
        setTimeout(() => {
          icon.style.animation = "";
          refreshRoomsBtn.disabled = false;
        }, 500);
      }
    });
  }

  const refreshBookingsBtn = document.getElementById("refreshBookingsBtn");
  if (refreshBookingsBtn) {
    refreshBookingsBtn.addEventListener("click", async () => {
      const icon = refreshBookingsBtn.querySelector("svg");
      icon.style.animation = "spin 1s linear infinite";
      refreshBookingsBtn.disabled = true;

      try {
        await loadBookings();
      } finally {
        setTimeout(() => {
          icon.style.animation = "";
          refreshBookingsBtn.disabled = false;
        }, 500);
      }
    });
  }

  // Stat Card Navigation
  const statCards = document.querySelectorAll(".stat-card[data-navigate]");
  statCards.forEach((card) => {
    card.addEventListener("click", () => {
      const targetTab = card.getAttribute("data-navigate");
      if (targetTab) {
        switchTab(targetTab);
      }
    });
  });

  // Theme Toggle
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", toggleTheme);

    // Initialize theme
    if (
      savedTheme === "dark" ||
      (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches)
    ) {
      document.body.classList.add("dark-mode");
      themeToggle.textContent = "☀️";
    }
  }

  // Sidebar Toggle
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      document.querySelector(".sidebar").classList.toggle("active");
    });
  }
}

function toggleTheme() {
  const body = document.body;
  const themeToggle = document.getElementById("themeToggle");

  body.classList.toggle("dark-mode");

  if (body.classList.contains("dark-mode")) {
    localStorage.setItem("theme", "dark");
    themeToggle.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sun"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41-1.41"/><path d="m19.07 4.93-1.41-1.41"/></svg>';
  } else {
    localStorage.setItem("theme", "light");
    themeToggle.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-moon"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';
  }
}
function toggleClear(input) {
  const clearBtn = input.nextElementSibling;
  if (!clearBtn) return;

  clearBtn.style.display = input.value ? "block" : "none";
}

function clearSearch(inputId) {
  const input = document.getElementById(inputId);
  if (!input) return;

  input.value = "";

  // Hide clear button
  const clearBtn = input.nextElementSibling;
  if (clearBtn) clearBtn.style.display = "none";

  // Trigger relevant refresh
  if (inputId === "roomSearchInput") {
    renderRoomTypes(roomTypes);
  } else if (inputId === "searchBookings") {
    filterBookingsTable();
  }
}

// Login
async function handleLogin(e) {
  e.preventDefault();
  loginError.style.display = "none";

  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;

  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);

  try {
    const response = await fetch(`${API_BASE}/login`, {
      method: "POST",
      body: formData,
      headers: {
        "ngrok-skip-browser-warning": "true",
      },
    });

    if (!response.ok) {
      throw new Error("Invalid credentials");
    }

    const data = await response.json();
    token = data.access_token;
    localStorage.setItem("auth_token", token);

    loginScreen.style.display = "none";
    dashboard.style.display = "block";

    loadDashboardData();
  } catch (error) {
    loginError.textContent = "Invalid username or password";
    loginError.style.display = "block";
  }
}

// Logout
function handleLogout() {
  token = "";
  localStorage.removeItem("auth_token");
  loginScreen.style.display = "flex";
  dashboard.style.display = "none";
  loginForm.reset();
}

function checkLoginStatus() {
  const savedToken = localStorage.getItem("auth_token");
  if (savedToken) {
    token = savedToken;
    loginScreen.style.display = "none";
    dashboard.style.display = "block";
    loadDashboardData();
  }
}

// Switch Tabs
function switchTab(tabName) {
  navTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });

  tabContents.forEach((content) => {
    content.classList.toggle("active", content.id === `${tabName}Tab`);
  });

  // Update Page Title
  const titleMap = {
    overview: "Overview",
    rooms: "Room Management",
    bookings: "Booking Management",
    calendar: "Room Calendar",
  };
  const pageTitle = document.getElementById("pageTitle");
  if (pageTitle) {
    pageTitle.textContent = titleMap[tabName] || "Dashboard";
  }

  if (tabName === "overview") {
    loadDashboardData();
  } else if (tabName === "rooms") {
    loadRoomTypes();
  } else if (tabName === "bookings") {
    loadBookings();
  } else if (tabName === "calendar") {
    loadCalendarData();
  }
}

// Load Dashboard Data
async function loadDashboardData() {
  await Promise.all([loadRoomTypes(), loadBookings()]);
  updateStatistics();
  renderRecentBookings();
}

// Load Room Types
// Load Room Types
async function loadRoomTypes(page = 1, limit = 10) {
  try {
    const response = await fetch(
      `${API_BASE}/room-types?page=${page}&limit=${limit}`,
      {
        headers: {
          "ngrok-skip-browser-warning": "true",
          Authorization: `Bearer ${token}`,
        },
      }
    );
    document.getElementById(
      "pageInfo"
    ).textContent = `Page ${currentPage} of ${totalPages}`;

    document.getElementById("prevPageBtn").disabled = currentPage === 1;
    document.getElementById("nextPageBtn").disabled =
      currentPage === totalPages;

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const result = await response.json();

    roomTypes = result.data; // ✅ important
    totalPages = result.total_pages;

    renderRoomTypes();
    updateStatistics();
    populateCalendarRoomTypes();
  } catch (error) {
    console.error("Error loading room types:", error);
  }
}

function populateCalendarRoomTypes() {
  const list = document.getElementById("calendarRoomTypeList");
  if (!list) return;

  list.innerHTML = "";

  // All option
  const all = document.createElement("div");
  all.className = "dropdown-item";
  all.textContent = "All Room Types";
  all.onclick = () => selectCalendarRoomType("all", "All Room Types");
  list.appendChild(all);

  roomTypes.forEach((room) => {
    const item = document.createElement("div");
    item.className = "dropdown-item";
    item.textContent = room.name;
    item.onclick = () => selectCalendarRoomType(room.id, room.name);
    list.appendChild(item);
  });
}
function selectCalendarRoomType(value, label) {
  const calendarInput = document.getElementById("calendarRoomTypeInput");
  const clearBtn = document.getElementById("calendarRoomTypeClear");
  const list = document.getElementById("calendarRoomTypeList");

  selectedCalendarRoomType = value;
  calendarInput.value = label;
  clearBtn.style.display = value === "all" ? "none" : "block";
  list.style.display = "none";

  renderCalendar();
}

// Load Bookings
async function loadBookings() {
  try {
    const response = await fetch(`${API_BASE}/bookings`, {
      headers: {
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();

    bookings = [...data].reverse(); // 🔁 reverse order safely

    renderBookingsTable();
    updateStatistics();
  } catch (error) {
    console.error("Error loading bookings:", error);
  }
}

// Load Calendar Data
async function loadCalendarData() {
  try {
    // Load both room types and bookings
    await Promise.all([loadRoomTypes(), loadBookings()]);
    renderCalendar();
  } catch (error) {
    console.error("Error loading calendar data:", error);
  }
}

// Update Statistics
async function updateStatistics() {
  try {
    const response = await fetch(`${API_BASE}/dashboard/stats`, {
      headers: {
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
    });

    const stats = await response.json();

    document.getElementById("totalRooms").textContent = stats.total_rooms;

    document.getElementById("totalBookings").textContent = stats.total_bookings;

    document.getElementById("confirmedBookings").textContent =
      stats.confirmed_bookings;

    document.getElementById(
      "revenue"
    ).textContent = `₹${stats.revenue.toLocaleString()}`;

    // Optional: show month label
    document.getElementById(
      "currentDate"
    ).textContent = `Stats for ${stats.month}`;
  } catch (error) {
    console.error("Dashboard stats error:", error);
  }
}

// Render Recent Bookings
function renderRecentBookings() {
  const tbody = document.getElementById("recentBookingsTable");

  const recentBookings = [...bookings] // copy array
    // reverse order (latest first)
    .slice(0, 5); // take last 5 bookings

  tbody.innerHTML = recentBookings
    .map(
      (booking) => `
        <tr>
            <td>#${booking.booking_id}</td>
            <td>${booking.user_name}</td>
            <td>${booking.room_name} (#${booking.room_no})</td>
            <td>${booking.check_in_date}</td>
            <td>
                <span class="status-badge ${booking.status}">
                    ${booking.status}
                </span>
            </td>
        </tr>
    `
    )
    .join("");
}

// Render Room Types
function renderRoomTypes(data = roomTypes) {
  const grid = document.getElementById("roomsGrid");

  grid.innerHTML = data
    .map(
      (room) => `
        <div class="room-card">
            ${
              room.image_url
                ? `<img src="${room.image_url}" alt="${room.name}" class="room-image">`
                : ""
            }
            <div class="room-content">
                <h3 class="room-title">${room.name}</h3>
                <p class="room-description">${room.description || ""}</p>
                
                <div class="room-details">
                    <div class="room-detail-row">
                        <span class="room-detail-label">Capacity:</span>
                        <span class="room-detail-value">${
                          room.capacity.adults
                        }A + ${room.capacity.children}C</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Price:</span>
                        <span class="room-detail-value" style="color: #4f46e5;">₹${
                          room.pricing.total_price
                        }/night</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Total Rooms:</span>
                        <span class="room-detail-value">${
                          room.room_numbers.length
                        }</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Stay Range:</span>
                        <span class="room-detail-value">${room.min_days}-${
        room.max_days
      } days</span>
                    </div>
                </div>
                
                ${
                  room.amenities.length > 0
                    ? `
                    <div class="amenities-list">
                        ${room.amenities
                          .slice(0, 3)
                          .map(
                            (amenity) =>
                              `<span class="amenity-tag">${amenity}</span>`
                          )
                          .join("")}
                        ${
                          room.amenities.length > 3
                            ? `<span class="amenity-tag">+${
                                room.amenities.length - 3
                              } more</span>`
                            : ""
                        }
                    </div>
                `
                    : ""
                }
                
                <div class="room-actions">
                    <button class="btn-edit" onclick="editRoom(${
                      room.id
                    })" style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-pencil"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>
                        Edit
                    </button>
                    <button class="btn-delete" onclick="deleteRoom(${
                      room.id
                    })" style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
                        Delete
                    </button>
                </div>
            </div>
        </div>
    `
    )
    .join("");
}
function searchRoomTypes() {
  const query = document
    .getElementById("roomSearchInput")
    .value.toLowerCase()
    .trim();

  if (!query) {
    renderRoomTypes(roomTypes);
    return;
  }

  const filtered = roomTypes.filter((room) => {
    // Room name
    if (room.name.toLowerCase().includes(query)) return true;

    // Description
    if ((room.description || "").toLowerCase().includes(query)) return true;

    // Amenities
    if (
      room.amenities &&
      room.amenities.some((a) => a.toLowerCase().includes(query))
    )
      return true;

    // Room numbers
    if (
      room.room_numbers &&
      room.room_numbers.some((rn) =>
        String(typeof rn === "object" ? rn.room_no : rn).includes(query)
      )
    )
      return true;

    // Price
    if (String(room.pricing.total_price).includes(query)) return true;

    return false;
  });

  renderRoomTypes(filtered);
}

// Render Bookings Table
function renderBookingsTable() {
  filterBookingsTable();
}

// Filter Bookings Table
function filterBookingsTable() {
  const searchTerm = searchBookings.value.toLowerCase();
  const status = filterStatus.value;
  const fromDate = document.getElementById("filterFromDate").value;
  const toDate = document.getElementById("filterToDate").value;

  const filtered = bookings.filter((booking) => {
    // 🔍 Text search
    const matchesSearch =
      booking.user_name.toLowerCase().includes(searchTerm) ||
      booking.email.toLowerCase().includes(searchTerm) ||
      booking.booking_id.toString().includes(searchTerm);

    // 📌 Status filter
    const matchesStatus = status === "all" || booking.status === status;

    // 📅 Date filter
    const bookingDate = booking.check_in_date;

    const matchesFrom = !fromDate || bookingDate >= fromDate;

    const matchesTo = !toDate || bookingDate <= toDate;

    return matchesSearch && matchesStatus && matchesFrom && matchesTo;
  });

  const tbody = document.getElementById("bookingsTable");

  tbody.innerHTML = filtered
    .map(
      (booking) => `
        <tr onclick="showBookingDetails(${booking.booking_id})">
          <td>#${booking.booking_id}</td>
          <td>${booking.user_name}</td>
          <td>${booking.room_name} (#${booking.room_no})</td>
          <td>
            <span class="status-badge ${booking.status}">
              ${booking.status}
            </span>
          </td>
          <td onclick="event.stopPropagation()">
            ${
              booking.status === "confirmed"
                ? `<button class="btn-cancel-booking" onclick="cancelBooking(${booking.booking_id})">Cancel</button>`
                : "-"
            }
          </td>
        </tr>
      `
    )
    .join("");
}

// Show Booking Details
function showBookingDetails(bookingId) {
  const booking = bookings.find((b) => b.booking_id === bookingId);
  if (!booking) return;

  document.getElementById("detailId").textContent = `#${booking.booking_id}`;
  document.getElementById("detailName").textContent = booking.user_name;
  document.getElementById("detailEmail").textContent = booking.email;
  document.getElementById(
    "detailRoom"
  ).textContent = `${booking.room_name} (Room ${booking.room_no})`;
  document.getElementById("detailCheckIn").textContent = booking.check_in_date;
  document.getElementById("detailCheckOut").textContent =
    booking.check_out_date;
  document.getElementById(
    "detailDays"
  ).textContent = `${booking.stay_days} nights`;

  const statusEl = document.getElementById("detailStatus");
  statusEl.textContent = booking.status;
  statusEl.className = `status-badge ${booking.status}`;

  // Additional details
  const guestsText =
    `${booking.adults} Adult${booking.adults > 1 ? "s" : ""}` +
    (booking.children > 0
      ? `, ${booking.children} Child${booking.children > 1 ? "ren" : ""}`
      : "");
  document.getElementById("detailGuests").textContent = guestsText;

  document.getElementById(
    "detailPrice"
  ).textContent = `₹${booking.total_price.toLocaleString()}`;

  // Format created_at date
  const createdDate = new Date(booking.created_at);
  const formattedDate = createdDate.toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  document.getElementById("detailCreatedAt").textContent = formattedDate;

  document.getElementById("bookingDetailsModal").classList.add("active");
}

// Cancel Booking
async function cancelBooking(bookingId) {
  if (!confirm("Are you sure you want to cancel this booking?")) return;

  try {
    const response = await fetch(`${API_BASE}/bookings/${bookingId}`, {
      method: "DELETE",
      headers: {
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
    });

    if (response.ok) {
      alert("Booking cancelled successfully");
      loadBookings();
    }
  } catch (error) {
    alert("Error cancelling booking");
  }
}

// Room Type Form Functions
function resetRoomForm() {
  roomTypeForm.reset();
  roomNumbers = [];
  document.getElementById("roomNumbersContainer").innerHTML = "";
  document.getElementById("taxPrice").value = "";
  document.getElementById("totalPrice").value = "";
}

function calculatePricing() {
  const basePrice = parseFloat(document.getElementById("basePrice").value) || 0;
  const taxPrice = basePrice * 0.18;
  const totalPrice = basePrice + taxPrice;

  document.getElementById("taxPrice").value = taxPrice.toFixed(2);
  document.getElementById("totalPrice").value = totalPrice.toFixed(2);
}

function addRoomNumberField() {
  const container = document.getElementById("roomNumbersContainer");
  const roomNo = roomNumbers.length + 1;

  const div = document.createElement("div");
  div.className = "room-number-row";
  div.innerHTML = `
        <input type="number" placeholder="Room number" value="${roomNo}" required>
        <button type="button" class="btn-remove" onclick="this.parentElement.remove()" style="display: flex; align-items: center; justify-content: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-x"><path d="M18 6 6 18"/><path d="m6 6 18 12"/></svg>
        </button>
    `;

  container.appendChild(div);
  roomNumbers.push(roomNo);
}

// Handle Room Form Submit (Create or Update)
async function handleCreateRoomType(e) {
  e.preventDefault();

  const roomNumberRows = document.querySelectorAll(".room-number-row");
  const roomNumbersData = Array.from(roomNumberRows).map((row) => ({
    room_no: parseInt(row.querySelector("input").value),
    status: "available",
  }));

  const amenitiesInput = document.getElementById("amenities").value;
  const amenitiesArray = amenitiesInput
    ? amenitiesInput
        .split(",")
        .map((a) => a.trim())
        .filter((a) => a)
    : [];

  const roomTypeData = {
    id: parseInt(document.getElementById("roomId").value), // ID is required for creation, might be ignored on update depending on API
    name: document.getElementById("roomName").value,
    description: document.getElementById("roomDescription").value,
    capacity: {
      adults: parseInt(document.getElementById("adultsCapacity").value),
      children: parseInt(document.getElementById("childrenCapacity").value),
    },
    amenities: amenitiesArray,
    min_days: parseInt(document.getElementById("minDays").value),
    max_days: parseInt(document.getElementById("maxDays").value),
    pricing: {
      base_price: parseFloat(document.getElementById("basePrice").value),
      tax_price: parseFloat(document.getElementById("taxPrice").value),
      total_price: parseFloat(document.getElementById("totalPrice").value),
      currency: "INR",
      pricing_type: "per night",
    },
    room_numbers: roomNumbersData,
    image_url: document.getElementById("imageUrl").value,
    banner_image: "",
    refund_policy: "",
  };

  try {
    const url = isEditing
      ? `${API_BASE}/room-types/${editingRoomId}`
      : `${API_BASE}/room-types`;
    const method = isEditing ? "PUT" : "POST";

    const response = await fetch(url, {
      method: method,
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(roomTypeData),
    });

    if (response.ok) {
      alert(
        isEditing
          ? "Room type updated successfully!"
          : "Room type created successfully!"
      );
      roomModal.classList.remove("active");
      loadRoomTypes();
      resetRoomForm();
    } else {
      const error = await response.json();
      let errorMessage = "Operation failed";

      if (error.detail) {
        if (typeof error.detail === "string") {
          errorMessage = error.detail;
        } else if (Array.isArray(error.detail)) {
          errorMessage = error.detail
            .map((e) => e.msg || JSON.stringify(e))
            .join(", ");
        } else {
          errorMessage = JSON.stringify(error.detail);
        }
      }

      alert(`Error: ${errorMessage}`);
    }
  } catch (error) {
    console.error("Error saving room type:", error);
    alert("Error saving room type: " + error.message);
  }
}

// Edit Room
async function editRoom(roomId) {
  try {
    const response = await fetch(`${API_BASE}/room/${roomId}`, {
      headers: {
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) throw new Error("Failed to fetch room details");
    const room = await response.json(); // Usually returns an array or single object. Assuming single object here given the ID.
    // NOTE: If API returns array like [room], we need room[0].
    // Assuming API implementation returns the object directly.

    isEditing = true;
    editingRoomId = roomId;

    // Populate Form
    document.getElementById("roomId").value = room.id;
    document.getElementById("roomName").value = room.name;
    document.getElementById("roomDescription").value = room.description || "";
    document.getElementById("adultsCapacity").value = room.capacity.adults;
    document.getElementById("childrenCapacity").value = room.capacity.children;
    document.getElementById("basePrice").value = room.pricing.base_price;
    // Trigger calculation for tax and total
    calculatePricing();

    document.getElementById("minDays").value = room.min_days;
    document.getElementById("maxDays").value = room.max_days;
    document.getElementById("imageUrl").value = room.image_url || "";
    document.getElementById("amenities").value = room.amenities.join(", ");

    // Populate Room Numbers
    const container = document.getElementById("roomNumbersContainer");
    container.innerHTML = "";
    roomNumbers = []; // Clear current tracking

    if (room.room_numbers && room.room_numbers.length > 0) {
      room.room_numbers.forEach((rn) => {
        addRoomNumberFieldWithValues(rn.room_no, rn.status);
      });
    }

    // Update Modal UI
    document.querySelector(".modal-header h3").textContent = "Edit Room Type";
    document.querySelector(".modal-footer .btn-primary").textContent =
      "Update Room Type";

    // Show Modal
    roomModal.classList.add("active");
  } catch (error) {
    console.error("Error fetching room details:", error);
    alert("Could not load room details for editing.");
  }
}

function addRoomNumberFieldWithValues(number, status) {
  const container = document.getElementById("roomNumbersContainer");
  // Store only the room number
  roomNumbers.push(number);

  const div = document.createElement("div");
  div.className = "room-number-row";
  div.innerHTML = `
        <input type="number" placeholder="Room number" value="${number}" required>
        <button type="button" class="btn-remove" onclick="this.parentElement.remove()" style="display: flex; align-items: center; justify-content: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-x"><path d="M18 6 6 18"/><path d="m6 6 18 12"/></svg>         
        </button>
    `;
  container.appendChild(div);
}

// Delete Room
async function deleteRoom(roomId) {
  if (
    !confirm(
      "Are you sure you want to delete this room type? This action cannot be undone."
    )
  ) {
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/room-types/${roomId}`, {
      method: "DELETE",
      headers: {
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
    });

    if (response.ok) {
      alert("Room type deleted successfully");
      loadRoomTypes();
    } else {
      const error = await response.text(); // or .json() depending on API
      alert("Error deleting room type. It might be in use.");
      console.error(error);
    }
  } catch (error) {
    console.error("Error deleting room type:", error);
    alert("Error deleting room type");
  }
}

// Populate Booking Room Types
function populateBookingRoomTypes() {
  const select = document.getElementById("bookingRoomType");
  select.innerHTML = '<option value="">Select Room Type</option>';

  roomTypes.forEach((room) => {
    const option = document.createElement("option");
    option.value = room.id;
    option.textContent = `${room.name} (₹${room.pricing.total_price}/night)`;
    select.appendChild(option);
  });
}

// Handle Children Count Change
function handleChildrenCountChange() {
  const childrenCount =
    parseInt(document.getElementById("bookingChildren").value) || 0;
  const container = document.getElementById("childAgesContainer");
  const inputsDiv = document.getElementById("childAgesInputs");

  // Show/hide container
  if (childrenCount > 0) {
    container.style.display = "block";
  } else {
    container.style.display = "none";
    inputsDiv.innerHTML = "";
    return;
  }

  // Clear existing inputs
  inputsDiv.innerHTML = "";

  // Create input fields for each child
  for (let i = 0; i < childrenCount; i++) {
    const ageInput = document.createElement("input");
    ageInput.type = "number";
    ageInput.min = "0";
    ageInput.max = "17";
    ageInput.placeholder = `Age of Child ${i + 1}`;
    ageInput.className = "child-age-input";
    ageInput.required = true;
    ageInput.style.width = "100%";
    inputsDiv.appendChild(ageInput);
  }
}

// Handle Create Booking
async function handleCreateBooking(e) {
  e.preventDefault();

  // Collect child ages
  const childAgesInputs = document.querySelectorAll(".child-age-input");
  const childrenAges = Array.from(childAgesInputs).map((input) =>
    parseInt(input.value)
  );

  const bookingData = {
    user_name: document.getElementById("bookingGuestName").value,
    email: document.getElementById("bookingEmail").value,
    room_type_id: parseInt(document.getElementById("bookingRoomType").value),
    check_in_date: document.getElementById("bookingCheckIn").value,
    check_out_date: document.getElementById("bookingCheckOut").value,
    adults: parseInt(document.getElementById("bookingAdults").value),
    children: parseInt(document.getElementById("bookingChildren").value),
    children_ages: childrenAges,
    // Add defaults or calculation for other fields if API requires them
    // For simplicity, assuming these are the core fields needed
  };

  try {
    const response = await fetch(`${API_BASE}/bookings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(bookingData),
    });

    if (response.ok) {
      alert("Booking created successfully!");
      bookingModal.classList.remove("active");
      document.getElementById("bookingForm").reset();
      loadBookings();
    } else {
      const error = await response.json();
      alert(`Error: ${error.detail || "Failed to create booking"}`);
    }
  } catch (error) {
    console.error("Error creating booking:", error);
    alert("Error creating booking");
  }
}
// Calendar Functions
function changeMonth(direction) {
  if (typeof direction === "number") {
    currentCalendarDate.setMonth(currentCalendarDate.getMonth() + direction);
  } else if (typeof direction === "object" && direction.target) {
    // Handle direct month selection from dropdown
    const selectedMonth = parseInt(direction.target.value);
    currentCalendarDate.setMonth(selectedMonth);
  }

  updateYearSelect();
  updateMonthSelect();
  renderCalendar();
}

function changeYear(selectedYear) {
  const currentMonth = currentCalendarDate.getMonth();
  currentCalendarDate.setFullYear(selectedYear);
  // Handle February 29th edge case for non-leap years
  if (currentMonth === 1 && currentCalendarDate.getDate() > 28) {
    const lastDayOfFeb = new Date(selectedYear, 2, 0).getDate();
    if (currentCalendarDate.getDate() > lastDayOfFeb) {
      currentCalendarDate.setDate(lastDayOfFeb);
    }
  }
  renderCalendar();
}

function updateYearSelect() {
  const yearSelect = document.getElementById("yearSelect");
  if (!yearSelect) return;

  const currentYear = new Date().getFullYear();
  const selectedYear = currentCalendarDate.getFullYear();

  // Clear existing options
  yearSelect.innerHTML = "";

  // Add 5 years before and after current year
  for (let year = currentYear - 5; year <= currentYear + 5; year++) {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    if (year === selectedYear) {
      option.selected = true;
    }
    yearSelect.appendChild(option);
  }
}

function updateMonthSelect() {
  const monthSelect = document.getElementById("monthSelect");
  if (!monthSelect) return;

  const currentMonth = currentCalendarDate.getMonth();

  // Set the selected month in the dropdown
  monthSelect.value = currentMonth;
}

function renderCalendar() {
  const year = currentCalendarDate.getFullYear();
  const month = currentCalendarDate.getMonth();
  const filteredBookings =
    selectedCalendarRoomType === "all"
      ? bookings
      : bookings.filter(
          (b) => String(b.room_type_id) === String(selectedCalendarRoomType)
        );

  /* ======================
     Month Header
  ====================== */
  const monthNames = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  document.getElementById(
    "currentMonth"
  ).textContent = `${monthNames[month]} ${year}`;

  /* ======================
     Date helpers
  ====================== */
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const daysInMonth = lastDay.getDate();

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  function toggleClear(input) {
    const clearBtn = input.nextElementSibling;
    if (!clearBtn) return;

    clearBtn.style.display = input.value ? "block" : "none";
  }

  function clearSearch(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;

    input.value = "";

    // Hide clear button
    const clearBtn = input.nextElementSibling;
    if (clearBtn) clearBtn.style.display = "none";

    // Trigger relevant refresh
    if (inputId === "roomSearchInput") {
      renderRoomTypes(roomTypes);
    } else if (inputId === "searchBookings") {
      filterBookingsTable();
    }
  }

  /* ======================
     Group rooms by type
  ====================== */
  const roomGroups = roomTypes
    .filter(
      (rt) =>
        selectedCalendarRoomType === "all" ||
        String(rt.id) === String(selectedCalendarRoomType)
    )
    .map((rt) => ({
      type_id: rt.id,
      type_name: rt.name,
      rooms: (rt.room_numbers || []).map((rn) =>
        typeof rn === "object" ? rn.room_no : rn
      ),
    }));

  /* ======================
     Build table
  ====================== */
  let html = `
    <table class="calendar-table" style="width:100%; border-collapse:collapse;">
      <thead>
        <tr>
          <th style="
            position:sticky; left:0; z-index:10;
            background:white; border-right:2px solid #e5e7eb;
            padding:0.75rem;
          ">
            Room
          </th>
  `;

  /* ======================
     Header dates
  ====================== */
  for (let day = 1; day <= daysInMonth; day++) {
    const date = new Date(year, month, day);
    const isToday = date.getTime() === today.getTime();
    const dayName = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][
      date.getDay()
    ];

    html += `
      <th style="
        min-width:60px; text-align:center; font-size:0.75rem;
        padding:0.5rem;
        ${isToday ? "background:#eff6ff; font-weight:700;" : ""}
      ">
        <div>${dayName}</div>
        <div style="font-size:1rem">${day}</div>
      </th>
    `;
  }

  html += `</tr></thead><tbody>`;

  /* ======================
     Room type groups
  ====================== */
  roomGroups.forEach((group) => {
    // Type header row
    html += `
      <tr style="background:#f9fafb;">
        <td colspan="${daysInMonth + 1}"
            style="padding:0.75rem; font-weight:700;
                   border-top:2px solid #e5e7eb;">
          ${group.type_name}
        </td>
      </tr>
    `;

    // Rooms under type
    group.rooms.forEach((roomNum) => {
      html += `<tr>`;
      html += `
        <td style="
          position:sticky; left:0; z-index:9;
          background:white; border-right:2px solid #e5e7eb;
          padding:0.75rem; font-weight:600;
        ">
          Room ${roomNum}
        </td>
      `;

      /* ======================
         Day cells
      ====================== */
      for (let day = 1; day <= daysInMonth; day++) {
        const date = new Date(year, month, day);
        const isPast = date < today;

        // Find booking for this room/date
        const booking = filteredBookings.find((b) => {
          if (b.room_no !== roomNum || b.status === "cancelled") return false;

          const checkIn = new Date(b.check_in_date);
          const checkOut = new Date(b.check_out_date);
          checkIn.setHours(0, 0, 0, 0);
          checkOut.setHours(0, 0, 0, 0);

          // checkout day is free
          return date >= checkIn && date < checkOut;
        });

        let bg = "#dcfce7"; // available
        let border = "#86efac";
        let content = "";

        if (booking) {
          bg = "#fee2e2";
          border = "#fca5a5";

          const checkIn = new Date(booking.check_in_date);
          checkIn.setHours(0, 0, 0, 0);

          if (date.getTime() === checkIn.getTime()) {
            content = `
              <div style="font-size:0.7rem; font-weight:600; margin-top:0.25rem;">
                ${booking.user_name.split(" ")[0]}
              </div>
            `;
          }
        } else if (isPast) {
          bg = "#f3f4f6"; // past empty
          border = "#d1d5db";
        }

        html += `
          <td style="
            background:${bg};
            border:1px solid ${border};
            text-align:center;
            vertical-align:top;
            min-height:50px;
          ">
            ${content}
          </td>
        `;
      }

      html += `</tr>`;
    });
  });

  html += `</tbody></table>`;

  document.getElementById("calendarGrid").innerHTML = html;
}

function formatDateForComparison(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}
