const API_BASE = 'https://bookingbe.heykoala.ai';

// State
let token = '';
let roomTypes = [];
let bookings = [];
let roomNumbers = [];
let isEditing = false;
let editingRoomId = null;
let currentCalendarDate = new Date();
let allRoomNumbers = []; // Store all room numbers from all room types

// DOM Elements
const loginScreen = document.getElementById('loginScreen');
const dashboard = document.getElementById('dashboard');
const loginForm = document.getElementById('loginForm');
const loginError = document.getElementById('loginError');
const logoutBtn = document.getElementById('logoutBtn');

// Tabs
const navTabs = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');

// Modal
const roomModal = document.getElementById('roomModal');
const addRoomBtn = document.getElementById('addRoomBtn');
const modalClose = document.querySelector('.modal-close');
const modalCancel = document.querySelector('.modal-cancel');
const roomTypeForm = document.getElementById('roomTypeForm');
const bookingModal = document.getElementById('bookingModal');
const addBookingBtn = document.getElementById('addBookingBtn');
const bookingForm = document.getElementById('bookingForm');
const bookingModalClose = document.querySelector('.booking-modal-close');
const bookingModalCancel = document.querySelector('.booking-modal-cancel');

// Search and Filter
const searchBookings = document.getElementById('searchBookings');
const filterStatus = document.getElementById('filterStatus');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkLoginStatus();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    loginForm.addEventListener('submit', handleLogin);
    logoutBtn.addEventListener('click', handleLogout);

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    addRoomBtn.addEventListener('click', () => {
        isEditing = false;
        editingRoomId = null;
        document.querySelector('.modal-header h3').textContent = 'Create Room Type';
        document.querySelector('.modal-footer .btn-primary').textContent = 'Create Room Type';
        roomModal.classList.add('active');
        resetRoomForm();
    });

    modalClose.addEventListener('click', () => roomModal.classList.remove('active'));
    modalClose.addEventListener('click', () => roomModal.classList.remove('active'));
    modalCancel.addEventListener('click', () => roomModal.classList.remove('active'));

    // Booking Modal Listeners
    if (addBookingBtn) {
        addBookingBtn.addEventListener('click', () => {
            bookingModal.classList.add('active');
            populateBookingRoomTypes();
        });
    }

    if (bookingModalClose) {
        bookingModalClose.addEventListener('click', () => bookingModal.classList.remove('active'));
    }

    if (bookingModalCancel) {
        bookingModalCancel.addEventListener('click', () => bookingModal.classList.remove('active'));
    }

    if (bookingForm) {
        bookingForm.addEventListener('submit', handleCreateBooking);
    }

    const bookingDetailsModal = document.getElementById('bookingDetailsModal');
    const bookingDetailsCloseButtons = document.querySelectorAll('.booking-details-close');

    if (bookingDetailsCloseButtons) {
        bookingDetailsCloseButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                bookingDetailsModal.classList.remove('active');
            });
        });
    }

    roomTypeForm.addEventListener('submit', handleCreateRoomType);

    document.getElementById('addRoomNumber').addEventListener('click', addRoomNumberField);
    document.getElementById('basePrice').addEventListener('input', calculatePricing);

    searchBookings.addEventListener('input', filterBookingsTable);
    filterStatus.addEventListener('change', filterBookingsTable);

    // Calendar Navigation
    const prevMonthBtn = document.getElementById('prevMonth');
    const nextMonthBtn = document.getElementById('nextMonth');
    const todayBtn = document.getElementById('todayBtn');

    if (prevMonthBtn) prevMonthBtn.addEventListener('click', () => changeMonth(-1));
    if (nextMonthBtn) nextMonthBtn.addEventListener('click', () => changeMonth(1));
    if (todayBtn) todayBtn.addEventListener('click', () => {
        currentCalendarDate = new Date();
        renderCalendar();
    });

    // Handle children age inputs
    const bookingChildrenInput = document.getElementById('bookingChildren');
    if (bookingChildrenInput) {
        bookingChildrenInput.addEventListener('input', handleChildrenCountChange);
    }

    // Refresh Buttons
    const refreshRoomsBtn = document.getElementById('refreshRoomsBtn');
    if (refreshRoomsBtn) {
        refreshRoomsBtn.addEventListener('click', async () => {
            const icon = refreshRoomsBtn.querySelector('svg');
            icon.style.animation = 'spin 1s linear infinite';
            refreshRoomsBtn.disabled = true;

            try {
                await loadRoomTypes();
            } finally {
                // Ensure a minimum spin time or just stop when done, here we stop when done but strictly remove animation
                setTimeout(() => {
                    icon.style.animation = '';
                    refreshRoomsBtn.disabled = false;
                }, 500);
            }
        });
    }

    const refreshBookingsBtn = document.getElementById('refreshBookingsBtn');
    if (refreshBookingsBtn) {
        refreshBookingsBtn.addEventListener('click', async () => {
            const icon = refreshBookingsBtn.querySelector('svg');
            icon.style.animation = 'spin 1s linear infinite';
            refreshBookingsBtn.disabled = true;

            try {
                await loadBookings();
            } finally {
                setTimeout(() => {
                    icon.style.animation = '';
                    refreshBookingsBtn.disabled = false;
                }, 500);
            }
        });
    }

    // Stat Card Navigation
    const statCards = document.querySelectorAll('.stat-card[data-navigate]');
    statCards.forEach(card => {
        card.addEventListener('click', () => {
            const targetTab = card.getAttribute('data-navigate');
            if (targetTab) {
                switchTab(targetTab);
            }
        });
    });

    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);

        // Initialize theme
        if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.body.classList.add('dark-mode');
            themeToggle.textContent = '☀️';
        }
    }

    // Sidebar Toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.querySelector('.sidebar').classList.toggle('active');
        });
    }
}

function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');

    body.classList.toggle('dark-mode');

    if (body.classList.contains('dark-mode')) {
        localStorage.setItem('theme', 'dark');
        themeToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sun"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41-1.41"/><path d="m19.07 4.93-1.41-1.41"/></svg>';
    } else {
        localStorage.setItem('theme', 'light');
        themeToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-moon"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';
    }
}

// Login
async function handleLogin(e) {
    e.preventDefault();
    loginError.style.display = 'none';

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
        const response = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            body: formData,
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });

        if (!response.ok) {
            throw new Error('Invalid credentials');
        }

        const data = await response.json();
        token = data.access_token;
        localStorage.setItem('auth_token', token);

        loginScreen.style.display = 'none';
        dashboard.style.display = 'block';

        loadDashboardData();
    } catch (error) {
        loginError.textContent = 'Invalid username or password';
        loginError.style.display = 'block';
    }
}

// Logout
function handleLogout() {
    token = '';
    localStorage.removeItem('auth_token');
    loginScreen.style.display = 'flex';
    dashboard.style.display = 'none';
    loginForm.reset();
}

function checkLoginStatus() {
    const savedToken = localStorage.getItem('auth_token');
    if (savedToken) {
        token = savedToken;
        loginScreen.style.display = 'none';
        dashboard.style.display = 'block';
        loadDashboardData();
    }
}

// Switch Tabs
function switchTab(tabName) {
    navTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}Tab`);
    });

    // Update Page Title
    const titleMap = {
        'overview': 'Overview',
        'rooms': 'Room Management',
        'bookings': 'Booking Management',
        'calendar': 'Room Calendar'
    };
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
        pageTitle.textContent = titleMap[tabName] || 'Dashboard';
    }

    if (tabName === 'overview') {
        loadDashboardData();
    } else if (tabName === 'rooms') {
        loadRoomTypes();
    } else if (tabName === 'bookings') {
        loadBookings();
    } else if (tabName === 'calendar') {
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
async function loadRoomTypes() {
    try {
        const response = await fetch(`${API_BASE}/room-types`, {
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        roomTypes = await response.json();
        renderRoomTypes();
        updateStatistics();
    } catch (error) {
        console.error('Error loading room types:', error);
    }
}

// Load Bookings
async function loadBookings() {
    try {
        const response = await fetch(`${API_BASE}/bookings`, {
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        bookings = await response.json();
        renderBookingsTable();
        updateStatistics();
    } catch (error) {
        console.error('Error loading bookings:', error);
    }
}

// Load Calendar Data
async function loadCalendarData() {
    try {
        // Load both room types and bookings
        await Promise.all([loadRoomTypes(), loadBookings()]);
        renderCalendar();
    } catch (error) {
        console.error('Error loading calendar data:', error);
    }
}

// Update Statistics
function updateStatistics() {
    const totalRooms = roomTypes.reduce((sum, rt) => sum + rt.room_numbers.length, 0);
    const totalBookings = bookings.length;
    const confirmedBookings = bookings.filter(b => b.status === 'confirmed').length;

    const revenue = bookings
        .filter(b => b.status === 'confirmed')
        .reduce((sum, b) => {
            const roomType = roomTypes.find(rt => rt.id === b.room_type_id);
            return sum + (roomType ? roomType.pricing.total_price * b.stay_days : 0);
        }, 0);

    document.getElementById('totalRooms').textContent = totalRooms;
    document.getElementById('totalBookings').textContent = totalBookings;
    document.getElementById('confirmedBookings').textContent = confirmedBookings;
    document.getElementById('revenue').textContent = `₹${revenue.toLocaleString()}`;
}

// Render Recent Bookings
function renderRecentBookings() {
    const tbody = document.getElementById('recentBookingsTable');
    const recentBookings = bookings.slice(0, 5);

    tbody.innerHTML = recentBookings.map(booking => `
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
    `).join('');
}

// Render Room Types
function renderRoomTypes() {
    const grid = document.getElementById('roomsGrid');

    grid.innerHTML = roomTypes.map(room => `
        <div class="room-card">
            ${room.image_url ? `<img src="${room.image_url}" alt="${room.name}" class="room-image">` : ''}
            <div class="room-content">
                <h3 class="room-title">${room.name}</h3>
                <p class="room-description">${room.description || ''}</p>
                
                <div class="room-details">
                    <div class="room-detail-row">
                        <span class="room-detail-label">Capacity:</span>
                        <span class="room-detail-value">${room.capacity.adults}A + ${room.capacity.children}C</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Price:</span>
                        <span class="room-detail-value" style="color: #4f46e5;">₹${room.pricing.total_price}/night</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Total Rooms:</span>
                        <span class="room-detail-value">${room.room_numbers.length}</span>
                    </div>
                    <div class="room-detail-row">
                        <span class="room-detail-label">Stay Range:</span>
                        <span class="room-detail-value">${room.min_days}-${room.max_days} days</span>
                    </div>
                </div>
                
                ${room.amenities.length > 0 ? `
                    <div class="amenities-list">
                        ${room.amenities.slice(0, 3).map(amenity =>
        `<span class="amenity-tag">${amenity}</span>`
    ).join('')}
                        ${room.amenities.length > 3 ?
                `<span class="amenity-tag">+${room.amenities.length - 3} more</span>` : ''
            }
                    </div>
                ` : ''}
                
                <div class="room-actions">
                    <button class="btn-edit" onclick="editRoom(${room.id})" style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-pencil"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>
                        Edit
                    </button>
                    <button class="btn-delete" onclick="deleteRoom(${room.id})" style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
                        Delete
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

// Render Bookings Table
function renderBookingsTable() {
    filterBookingsTable();
}

// Filter Bookings Table
function filterBookingsTable() {
    const searchTerm = searchBookings.value.toLowerCase();
    const status = filterStatus.value;

    const filtered = bookings.filter(booking => {
        const matchesSearch =
            booking.user_name.toLowerCase().includes(searchTerm) ||
            booking.email.toLowerCase().includes(searchTerm) ||
            booking.booking_id.toString().includes(searchTerm);

        const matchesStatus = status === 'all' || booking.status === status;

        return matchesSearch && matchesStatus;
    });

    const tbody = document.getElementById('bookingsTable');

    tbody.innerHTML = filtered.map(booking => `
        <tr onclick="showBookingDetails(${booking.booking_id})" style="cursor: pointer;">
            <td>#${booking.booking_id}</td>
            <td>${booking.user_name}</td>
            <td>${booking.room_name} (#${booking.room_no})</td>
            <td>
                <span class="status-badge ${booking.status}">
                    ${booking.status}
                </span>
            </td>
            <td onclick="event.stopPropagation()">
                ${booking.status === 'confirmed' ?
            `<button class="btn-cancel-booking" onclick="cancelBooking(${booking.booking_id})">Cancel</button>`
            : '-'
        }
            </td>
        </tr>
    `).join('');
}

// Show Booking Details
function showBookingDetails(bookingId) {
    const booking = bookings.find(b => b.booking_id === bookingId);
    if (!booking) return;

    document.getElementById('detailId').textContent = `#${booking.booking_id}`;
    document.getElementById('detailName').textContent = booking.user_name;
    document.getElementById('detailEmail').textContent = booking.email;
    document.getElementById('detailRoom').textContent = `${booking.room_name} (Room ${booking.room_no})`;
    document.getElementById('detailCheckIn').textContent = booking.check_in_date;
    document.getElementById('detailCheckOut').textContent = booking.check_out_date;
    document.getElementById('detailDays').textContent = `${booking.stay_days} nights`;

    const statusEl = document.getElementById('detailStatus');
    statusEl.textContent = booking.status;
    statusEl.className = `status-badge ${booking.status}`;

    // Additional details
    const guestsText = `${booking.adults} Adult${booking.adults > 1 ? 's' : ''}` +
        (booking.children > 0 ? `, ${booking.children} Child${booking.children > 1 ? 'ren' : ''}` : '');
    document.getElementById('detailGuests').textContent = guestsText;

    document.getElementById('detailPrice').textContent = `₹${booking.total_price.toLocaleString()}`;

    // Format created_at date
    const createdDate = new Date(booking.created_at);
    const formattedDate = createdDate.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
    document.getElementById('detailCreatedAt').textContent = formattedDate;

    document.getElementById('bookingDetailsModal').classList.add('active');
}

// Cancel Booking
async function cancelBooking(bookingId) {
    if (!confirm('Are you sure you want to cancel this booking?')) return;

    try {
        const response = await fetch(`${API_BASE}/bookings/${bookingId}`, {
            method: 'DELETE',
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            alert('Booking cancelled successfully');
            loadBookings();
        }
    } catch (error) {
        alert('Error cancelling booking');
    }
}

// Room Type Form Functions
function resetRoomForm() {
    roomTypeForm.reset();
    roomNumbers = [];
    document.getElementById('roomNumbersContainer').innerHTML = '';
    document.getElementById('taxPrice').value = '';
    document.getElementById('totalPrice').value = '';
}

function calculatePricing() {
    const basePrice = parseFloat(document.getElementById('basePrice').value) || 0;
    const taxPrice = basePrice * 0.18;
    const totalPrice = basePrice + taxPrice;

    document.getElementById('taxPrice').value = taxPrice.toFixed(2);
    document.getElementById('totalPrice').value = totalPrice.toFixed(2);
}

function addRoomNumberField() {
    const container = document.getElementById('roomNumbersContainer');
    const roomNo = roomNumbers.length + 1;

    const div = document.createElement('div');
    div.className = 'room-number-row';
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

    const roomNumberRows = document.querySelectorAll('.room-number-row');
    const roomNumbersData = Array.from(roomNumberRows).map(row => ({
        room_no: parseInt(row.querySelector('input').value),
        status: 'available'
    }));

    const amenitiesInput = document.getElementById('amenities').value;
    const amenitiesArray = amenitiesInput
        ? amenitiesInput.split(',').map(a => a.trim()).filter(a => a)
        : [];

    const roomTypeData = {
        id: parseInt(document.getElementById('roomId').value), // ID is required for creation, might be ignored on update depending on API
        name: document.getElementById('roomName').value,
        description: document.getElementById('roomDescription').value,
        capacity: {
            adults: parseInt(document.getElementById('adultsCapacity').value),
            children: parseInt(document.getElementById('childrenCapacity').value)
        },
        amenities: amenitiesArray,
        min_days: parseInt(document.getElementById('minDays').value),
        max_days: parseInt(document.getElementById('maxDays').value),
        pricing: {
            base_price: parseFloat(document.getElementById('basePrice').value),
            tax_price: parseFloat(document.getElementById('taxPrice').value),
            total_price: parseFloat(document.getElementById('totalPrice').value),
            currency: 'INR',
            pricing_type: 'per night'
        },
        room_numbers: roomNumbersData,
        image_url: document.getElementById('imageUrl').value,
        banner_image: '',
        refund_policy: ''
    };

    try {
        const url = isEditing ? `${API_BASE}/room-types/${editingRoomId}` : `${API_BASE}/room-types`;
        const method = isEditing ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(roomTypeData)
        });

        if (response.ok) {
            alert(isEditing ? 'Room type updated successfully!' : 'Room type created successfully!');
            roomModal.classList.remove('active');
            loadRoomTypes();
            resetRoomForm();
        } else {
            const error = await response.json();
            let errorMessage = 'Operation failed';

            if (error.detail) {
                if (typeof error.detail === 'string') {
                    errorMessage = error.detail;
                } else if (Array.isArray(error.detail)) {
                    errorMessage = error.detail.map(e => e.msg || JSON.stringify(e)).join(', ');
                } else {
                    errorMessage = JSON.stringify(error.detail);
                }
            }

            alert(`Error: ${errorMessage}`);
        }
    } catch (error) {
        console.error('Error saving room type:', error);
        alert('Error saving room type: ' + error.message);
    }
}

// Edit Room
async function editRoom(roomId) {
    try {
        const response = await fetch(`${API_BASE}/room/${roomId}`, {
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error('Failed to fetch room details');
        const room = await response.json(); // Usually returns an array or single object. Assuming single object here given the ID.
        // NOTE: If API returns array like [room], we need room[0].
        // Assuming API implementation returns the object directly.

        isEditing = true;
        editingRoomId = roomId;

        // Populate Form
        document.getElementById('roomId').value = room.id;
        document.getElementById('roomName').value = room.name;
        document.getElementById('roomDescription').value = room.description || '';
        document.getElementById('adultsCapacity').value = room.capacity.adults;
        document.getElementById('childrenCapacity').value = room.capacity.children;
        document.getElementById('basePrice').value = room.pricing.base_price;
        // Trigger calculation for tax and total
        calculatePricing();

        document.getElementById('minDays').value = room.min_days;
        document.getElementById('maxDays').value = room.max_days;
        document.getElementById('imageUrl').value = room.image_url || '';
        document.getElementById('amenities').value = room.amenities.join(', ');

        // Populate Room Numbers
        const container = document.getElementById('roomNumbersContainer');
        container.innerHTML = '';
        roomNumbers = []; // Clear current tracking

        if (room.room_numbers && room.room_numbers.length > 0) {
            room.room_numbers.forEach(rn => {
                addRoomNumberFieldWithValues(rn.room_no, rn.status);
            });
        }

        // Update Modal UI
        document.querySelector('.modal-header h3').textContent = 'Edit Room Type';
        document.querySelector('.modal-footer .btn-primary').textContent = 'Update Room Type';

        // Show Modal
        roomModal.classList.add('active');

    } catch (error) {
        console.error('Error fetching room details:', error);
        alert('Could not load room details for editing.');
    }
}

function addRoomNumberFieldWithValues(number, status) {
    const container = document.getElementById('roomNumbersContainer');
    // Store only the room number
    roomNumbers.push(number);

    const div = document.createElement('div');
    div.className = 'room-number-row';
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
    if (!confirm('Are you sure you want to delete this room type? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/room-types/${roomId}`, {
            method: 'DELETE',
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            alert('Room type deleted successfully');
            loadRoomTypes();
        } else {
            const error = await response.text(); // or .json() depending on API
            alert('Error deleting room type. It might be in use.');
            console.error(error);
        }
    } catch (error) {
        console.error('Error deleting room type:', error);
        alert('Error deleting room type');
    }
}


// Populate Booking Room Types
function populateBookingRoomTypes() {
    const select = document.getElementById('bookingRoomType');
    select.innerHTML = '<option value="">Select Room Type</option>';

    roomTypes.forEach(room => {
        const option = document.createElement('option');
        option.value = room.id;
        option.textContent = `${room.name} (₹${room.pricing.total_price}/night)`;
        select.appendChild(option);
    });
}

// Handle Children Count Change
function handleChildrenCountChange() {
    const childrenCount = parseInt(document.getElementById('bookingChildren').value) || 0;
    const container = document.getElementById('childAgesContainer');
    const inputsDiv = document.getElementById('childAgesInputs');

    // Show/hide container
    if (childrenCount > 0) {
        container.style.display = 'block';
    } else {
        container.style.display = 'none';
        inputsDiv.innerHTML = '';
        return;
    }

    // Clear existing inputs
    inputsDiv.innerHTML = '';

    // Create input fields for each child
    for (let i = 0; i < childrenCount; i++) {
        const ageInput = document.createElement('input');
        ageInput.type = 'number';
        ageInput.min = '0';
        ageInput.max = '17';
        ageInput.placeholder = `Age of Child ${i + 1}`;
        ageInput.className = 'child-age-input';
        ageInput.required = true;
        ageInput.style.width = '100%';
        inputsDiv.appendChild(ageInput);
    }
}

// Handle Create Booking
async function handleCreateBooking(e) {
    e.preventDefault();

    // Collect child ages
    const childAgesInputs = document.querySelectorAll('.child-age-input');
    const childrenAges = Array.from(childAgesInputs).map(input => parseInt(input.value));

    const bookingData = {
        user_name: document.getElementById('bookingGuestName').value,
        email: document.getElementById('bookingEmail').value,
        room_type_id: parseInt(document.getElementById('bookingRoomType').value),
        check_in_date: document.getElementById('bookingCheckIn').value,
        check_out_date: document.getElementById('bookingCheckOut').value,
        adults: parseInt(document.getElementById('bookingAdults').value),
        children: parseInt(document.getElementById('bookingChildren').value),
        children_ages: childrenAges,
        // Add defaults or calculation for other fields if API requires them
        // For simplicity, assuming these are the core fields needed
    };

    try {
        const response = await fetch(`${API_BASE}/bookings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(bookingData)
        });

        if (response.ok) {
            alert('Booking created successfully!');
            bookingModal.classList.remove('active');
            document.getElementById('bookingForm').reset();
            loadBookings();
        } else {
            const error = await response.json();
            alert(`Error: ${error.detail || 'Failed to create booking'}`);
        }
    } catch (error) {
        console.error('Error creating booking:', error);
        alert('Error creating booking');
    }
}
// Calendar Functions
function changeMonth(direction) {
    currentCalendarDate.setMonth(currentCalendarDate.getMonth() + direction);
    renderCalendar();
}

function renderCalendar() {
    const year = currentCalendarDate.getFullYear();
    const month = currentCalendarDate.getMonth();

    // Update month display
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'];
    document.getElementById('currentMonth').textContent = `${monthNames[month]} ${year}`;

    // Get all unique room numbers from room types
    allRoomNumbers = [];
    roomTypes.forEach(roomType => {
        if (roomType.room_numbers && Array.isArray(roomType.room_numbers)) {
            roomType.room_numbers.forEach(rn => {
                const roomNum = typeof rn === 'object' ? rn.room_no : rn;
                if (!allRoomNumbers.includes(roomNum)) {
                    allRoomNumbers.push(roomNum);
                }
            });
        }
    });
    allRoomNumbers.sort((a, b) => a - b);

    // Get days in month
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Build calendar grid
    let calendarHTML = '<table class="calendar-table" style="width: 100%; border-collapse: collapse;">';

    // Header row with dates
    calendarHTML += '<thead><tr><th style="padding: 0.75rem; text-align: left; font-weight: 600; position: sticky; left: 0; background: white; z-index: 10; border-right: 2px solid #e5e7eb;">Room</th>';
    for (let day = 1; day <= daysInMonth; day++) {
        const date = new Date(year, month, day);
        const dayName = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][date.getDay()];
        const isToday = date.getTime() === today.getTime();
        calendarHTML += `<th style="padding: 0.5rem; text-align: center; min-width: 60px; font-size: 0.75rem; ${isToday ? 'background: #eff6ff; font-weight: 700;' : ''}">
            <div>${dayName}</div>
            <div style="font-size: 1rem; margin-top: 0.25rem;">${day}</div>
        </th>`;
    }
    calendarHTML += '</tr></thead>';

    // Body rows with room availability
    calendarHTML += '<tbody>';
    allRoomNumbers.forEach(roomNum => {
        calendarHTML += `<tr>`;
        calendarHTML += `<td style="padding: 0.75rem; font-weight: 600; position: sticky; left: 0; background: white; z-index: 9; border-right: 2px solid #e5e7eb;">Room ${roomNum}</td>`;

        for (let day = 1; day <= daysInMonth; day++) {
            const date = new Date(year, month, day);
            const dateStr = formatDateForComparison(date);
            const isPast = date < today;

            // Check if room is booked on this date
            const isBooked = bookings.some(booking => {
                if (booking.room_no !== roomNum || booking.status === 'cancelled') return false;

                const checkIn = new Date(booking.check_in_date);
                const checkOut = new Date(booking.check_out_date);
                checkIn.setHours(0, 0, 0, 0);
                checkOut.setHours(0, 0, 0, 0);

                return date >= checkIn && date <= checkOut;
            });

            let bgColor = '#dcfce7'; // Available (green)
            let borderColor = '#86efac';
            let content = '';
            let isEndDate = false;

            if (isPast) {
                bgColor = '#f3f4f6'; // Past date (gray)
                borderColor = '#d1d5db';
            } else if (isBooked) {
                bgColor = '#fee2e2'; // Booked (red)
                borderColor = '#fca5a5';

                // Find booking details
                const booking = bookings.find(b => {
                    if (b.room_no !== roomNum || b.status === 'cancelled') return false;
                    const checkIn = new Date(b.check_in_date);
                    const checkOut = new Date(b.check_out_date);
                    checkIn.setHours(0, 0, 0, 0);
                    checkOut.setHours(0, 0, 0, 0);
                    return date >= checkIn && date <= checkOut;
                });

                if (booking) {
                    const checkIn = new Date(booking.check_in_date);
                    const checkOut = new Date(booking.check_out_date);
                    checkIn.setHours(0, 0, 0, 0);
                    checkOut.setHours(0, 0, 0, 0);

                    if (date.getTime() === checkIn.getTime()) {
                        content = `<div style="font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem;">${booking.user_name.split(' ')[0]}</div>`;
                    }

                    // Check if this is the end date of the booking
                    if (date.getTime() === checkOut.getTime()) {
                        isEndDate = true;
                    }
                }
            }

            // Add special border for end date
            const borderStyle = isEndDate
                ? `border: 1px solid ${borderColor}; border-right: 3px solid #dc2626;`
                : `border: 1px solid ${borderColor};`;

            calendarHTML += `<td style="padding: 0.5rem; text-align: center; background: ${bgColor}; ${borderStyle} min-height: 50px; vertical-align: top;">
                ${content}
            </td>`;
        }

        calendarHTML += '</tr>';
    });
    calendarHTML += '</tbody></table>';

    document.getElementById('calendarGrid').innerHTML = calendarHTML;
}

function formatDateForComparison(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}
